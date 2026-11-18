"""Audit completed independent batches without opening the sealed test region."""
import argparse
from collections import defaultdict
import json
import os
import gzip
from pathlib import Path

import numpy as np

from .__main__ import tomllib
from .astrometry import project
from .continuous_reference import verify_directory
from .continuous_instruments import chip_receipts,instrument_by_exposure
from .continuous_summary import pool, estimate, METHODS
from .io import digest, save_json
from .nsc_comparison import comparison
from .public_summary import check_aggregate
from .standard_motion import icrs_motion
from .validation_blocks import metrics as block_metrics
from .validation import ratio


def validate_release(state,protocol,release,code_root=Path('.')):
    """Require a recorded frozen decision before opening the reserved window."""
    recorded=json.loads(release.read_text())
    required=dict(selection_sha256=digest(state/'selection.json'),protocol_sha256=digest(protocol),
        method_sources_sha256={p.name:digest(p) for p in sorted((code_root/'src/deep_pm').glob('*.py'))},
        precision_sha256=digest(code_root/'configs/precision-v03.toml'),
        production_sha256=digest(code_root/'configs/spatial-v08.toml'),
        fit_source_sha256=digest(code_root/'src/deep_pm/continuous_fit.py'),
        reference_source_sha256=digest(code_root/'src/deep_pm/continuous_reference.py'),
        final_test_box=tomllib.loads(protocol.read_text())['final_test_box'],
        adopted_models=['C0/W0 r-only'],no_test_retuning=True)
    if any(recorded.get(k)!=v for k,v in required.items()):
        raise ValueError('Test release is missing or does not bind the frozen selection/model')
    decision=Path(recorded['decision_document'])
    if not decision.is_absolute():decision=code_root/decision
    if digest(decision)!=recorded.get('decision_sha256'):
        raise ValueError('Frozen decision document changed')
    return digest(release)


def standard_block(row, center):
    """Reconstruct vectors from retained tangent values, including old receipts.

    Published native axis errors are already ICRS; source C0 errors include the
    full state propagation. Historical private published vectors used an inverse
    Jacobian in error; they are deliberately never consumed here.
    """
    position=project([row['position_deg'][0]],[row['position_deg'][1]],center)[0]
    jac=icrs_motion(position,np.zeros(2),center)['state_jacobian'][np.ix_([1,3],[1,3])]
    motion={key:jac@np.asarray(value) for key,value in row['motion'].items()}
    np.testing.assert_allclose(motion['C0'],row['motion_icrs']['C0'],rtol=1e-9,atol=1e-8)
    return dict(row,motion=motion,nsc_epoch_days=row['nsc_nights_utc18'])


def comparison_cohorts(records):
    settings=dict(min_publish_count=5,tail_masyr=5,bootstrap_draws=800,bootstrap_absolute_zero=True,seed=9137)
    grouped={}
    for low in (21,22):
        selected=[r for r in records if low<=r['magnitude']<low+1]
        common=[r for r in selected if set(METHODS)<=r['motion'].keys()]
        sets={'all':common,
              'one_sdss_run':[r for r in common if len({e for e in r['exposures'] if e.startswith('SDSS:')})==1],
              'multiple_sdss_runs':[r for r in common if len({e for e in r['exposures'] if e.startswith('SDSS:')})>=2],
              'nsc_span_lt_2yr':[r for r in common if r['nsc_span_years']<2],
              'nsc_span_ge_2yr':[r for r in common if r['nsc_span_years']>=2]}
        for instrument in sorted({r['instrument_mix'] for r in common if 'instrument_mix' in r}):
            sets['instrument_'+instrument]=[r for r in common if r.get('instrument_mix')==instrument]
        for component in sorted({r['reference_component'] for r in common if 'reference_component' in r}):
            sets['reference_component_'+component]=[r for r in common if r.get('reference_component')==component]
        if any('median_nsc_r_class_star' in r for r in common):
            def shape_bin(row):
                value=row.get('median_nsc_r_class_star')
                if value is None or not 0<=value<=1:return 'shape_unavailable'
                return 'pointlike_qso' if value>=.8 else 'nonpointlike_qso'
            sets.update({key:[r for r in common if shape_bin(r)==key]
                         for key in ('pointlike_qso','nonpointlike_qso','shape_unavailable')})
        out={}
        for name,rows in sets.items():
            values=[{k:(r['motion'][k],r['error_icrs'][k]) for k in METHODS} for r in rows]
            stats=comparison(values,settings,(('C0','NSC'),('C0','Xiao')))
            out[name]=dict(common=stats,physical_group_sensitivity=estimate(rows,METHODS,9137),
                sampling_columns=['sdss_span_years','nsc_span_years','joint_span_years','calendar_year_count'],
                sampling_p16_p50_p84=np.quantile([[r[k] for k in ('sdss_span_years','nsc_span_years','joint_span_years','calendar_year_count')] for r in rows],[.16,.5,.84],axis=0) if rows else None)
            if name=='all':
                blocks=[dict(r,motion={'C0':r['motion']['Xiao'],'C1':r['motion']['C0']}) for r in rows]
                out[name]['observational_blocks']=dict(method_order=['Xiao','C0'],physical=block_metrics(blocks),
                    conservative_nights=block_metrics(blocks,True),scope='Connected observed groups; fixed fitted reference solution. No spatial interval when fewer than three components.')
        grouped[f'r{low}_{low+1}']=dict(eligible_count=len(selected),strict_common_count=len(common),conditions=out)
    return grouped


def blind_cohorts(records):
    settings=dict(min_publish_count=5,tail_masyr=5,bootstrap_draws=800,bootstrap_absolute_zero=True,seed=9137)
    out={}
    for low in (21,22):
        eligible=[r for r in records if low<=r['magnitude']<low+1]
        selected=[r for r in eligible if 'blind_track_ids' in r]
        if len(selected)!=len(eligible):
            out[f'r{low}_{low+1}']=dict(status='unavailable_in_historical_private_records',known_control_count=len(eligible),missing_blind_membership_count=len(eligible)-len(selected))
            continue
        matched=[r for r in selected if r.get('blind_C0') is not None]
        common=[r for r in matched if set(METHODS)<=r['motion'].keys()]
        values=[dict(NSC=(r['motion']['NSC'],r['error_icrs']['NSC']),Xiao=(r['motion']['Xiao'],r['error_icrs']['Xiao']),
                     C0_known=(r['motion']['C0'],r['error_icrs']['C0']),C0_blind=r['blind_C0']) for r in common]
        weighted=[dict(exposures=sorted(set(r['exposures'])|set(r['blind_visits'])),
            nsc_epoch_days=sorted(set(r['nsc_epoch_days'])|set(r['blind_nights'])),
            motion={key:pair[0] for key,pair in value.items()}) for r,value in zip(common,values)]
        out[f'r{low}_{low+1}']=dict(eligible_known_controls=len(selected),blind_recovered=ratio(len(matched),len(selected)),
            strict_common_comparison=comparison(values,settings,(('C0_blind','Xiao'),('C0_blind','C0_known'))),
            physical_group_sensitivity=estimate(weighted,('NSC','Xiao','C0_known','C0_blind'),9137,candidate='C0_blind'),
            scope='W0 usability including morphology/time/error/association cuts, conditional on eligible known-position spectral controls. No purity claim for unlabelled members.')
    return out


def run(state,protocol,output,release=None):
    sealed=tomllib.loads(protocol.read_text())['final_test_box']
    release_sha=validate_release(state,protocol,release) if release is not None else None
    tasks=[line.split() for line in (state/'tasks.tsv').read_text().splitlines()]
    allowed=[(name,box) for name,box in tasks if release_sha is not None or box!=sealed]
    # Filter BEFORE any fit/reference file access, not after loading outcomes.
    specs, fits, graphs, sets={}, {}, {}, defaultdict(list)
    records=defaultdict(list)
    archives=[];missing_metadata=[]
    for name,box in allowed:
        config=tomllib.loads((state/f'batch-{box}.toml').read_text())
        specs[name]=config['fields'][name]
        fits[name]=state/'outputs'/name
        graphs[box]=state/'shared'/box
        sets[box].append(name)
        chips,complete=chip_receipts((state/'inputs'/name/'run_manifest.json').resolve())
        archives.extend(chips)
        if not complete:missing_metadata.append(name)
        verify_directory(fits[name],('private_qso_blocks.json','continuous_core_summary.json'))
        gauge=json.loads((fits[name]/'continuous_core_summary.json').read_text())['reference_gauge']
        rows=json.loads((fits[name]/'private_qso_blocks.json').read_text())
        if any(r.get('blind_track_ids',{}).get('C0') is not None for r in rows):
            with gzip.open(fits[name]/'private_standard_motion.json.gz','rt') as stream:
                standard={r['track_id']:r for r in json.load(stream)}
        else:standard={}
        for row in rows:
            row=standard_block(row,config['projection_center_deg'])
            row['reference_component']=f'{box}:{gauge}'
            slot=row.get('blind_track_ids',{}).get('C0')
            if slot is not None:
                star=standard[slot]
                row['blind_C0']=(star['pm_icrs_masyr'],star['pm_error_masyr'])
                row['blind_visits']=star['exposures']
                row['blind_nights']=star['nsc_nights_utc18']
            records[box].append(row)
    instruments=instrument_by_exposure(sorted(set(archives)))
    aliases={'c4d':'DECam','decam':'DECam','ksb':'90Prime','90prime':'90Prime','k4m':'Mosaic3','mosaic3':'Mosaic3'}
    for rows in records.values():
        for row in rows:
            used={aliases.get(instruments.get(exp,'').strip().lower(),'unknown') for exp in row['exposures'] if exp.startswith('NSC:')}
            row['instrument_mix']='_and_'.join(sorted(used)) if used and 'unknown' not in used else 'incomplete'
    # Reuse receipt/ownership checks, avoiding duplicated tangent-frame inference
    # and enormous per-core copies of the same cohort comparisons.
    public=pool(specs,fits,graphs,{})
    public.pop('cohorts')
    for core in public['cores'].values():
        core['qso_counts']=core.pop('qso')['totals']
        core.pop('gaia')
    public['standard_icrs_comparison']={box:comparison_cohorts(rows) for box,rows in records.items()}
    public['standard_icrs_comparison']['pooled']=comparison_cohorts([r for rows in records.values() for r in rows])
    public['blind_w0_comparison']={box:blind_cohorts(rows) for box,rows in records.items()}
    public['blind_w0_comparison']['pooled']=blind_cohorts([r for rows in records.values() for r in rows])
    public['instrument_metadata']=dict(complete_cores=len(allowed)-len(missing_metadata),missing_cores=missing_metadata,
        verified_chip_sha256=[digest(path) for path in sorted(set(archives))],
        scope='Actual NSC exposures used by each known-position QSO control, not assumed from sky-window name.')
    if (state/'batch_preparation_summary.json').exists():
        preparation=json.loads((state/'batch_preparation_summary.json').read_text())
        public['registered_cumulative_area_sqdeg']=preparation['area_sqdeg']
        public['runtime_versions']=preparation.get('runtime_versions',dict(status='not_recorded_in_historical_preparation'))
    public.update(final_test_box=sealed,sealed_core_count=sum(box==sealed for _,box in tasks) if release_sha is None else 0,
        registered_test_core_count=sum(box==sealed for _,box in tasks),test_release_sha256=release_sha,
        test_results_read=release_sha is not None,protocol_sha256=digest(protocol),
        vector_conversion='Tangent velocity multiplied by inverse-gnomonic Jacobian; old private published inverse-conversion values ignored.',
        inference_scope='Known-position spectroscopic QSO. Reserved region included only with a recorded frozen release. Physical-group weight sensitivity is not independent spatial confidence.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(public)
    save_json(output,public)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','protocol','output'):p.add_argument(name,type=Path)
    p.add_argument('--release',type=Path,help='Explicit frozen selection/model decision receipt for the reserved final region')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    run(a.state,a.protocol,a.output,a.release)


if __name__=='__main__':main()
