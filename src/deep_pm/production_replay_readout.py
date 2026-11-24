"""Compare old, replayed, and new blind QSO cohorts from signed readout caches."""

import argparse
from collections import Counter
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd

from .__main__ import tomllib
from .astrometry import sdss_tai_to_tt, years

from .blind_catalogue_quality import merge_strata, read
from .continuous_reference import read_offsets, verify_directory
from .exposure_catalogues import inside
from .footprints import sdss_window_mask, sdss_window_selection
from .io import digest, save_json
from .nsc_opportunity_area import opportunity_masks
from .public_summary import check_aggregate


def load_blind(science, blind):
    manifest = science/'private_continuation_manifest.json'
    manifest_sha = digest(manifest)
    proof = read(science/'continuation_science_receipt.json')
    if proof['combined_science_sha256'] != digest(science/'continuation_production_science_summary.json'):
        raise ValueError('Combined science summary digest changed')
    status = read(blind/'v17_blind_catalogue_quality_summary.json')
    if status['manifest_sha256'] != manifest_sha:
        raise ValueError('Blind cohort and final catalogue refer to different scopes')
    parts = read(manifest)['partitions']
    rows = []
    for part in parts:
        owned = Path(part['owned'])
        expected_chunks = (part['cores'] + 63)//64
        for i in range(expected_chunks):
            chunk = read(blind/f"private_{part['label']}_{i:03}.json")
            ticket = dict(manifest_sha256=manifest_sha, receipt_sha256=digest(owned/'run_manifest.json'),
                          implementation_sha256=status['implementation_sha256'])
            if chunk['ticket'] != ticket:
                raise ValueError('A blind checkpoint belongs to different signed catalogue inputs')
            rows.extend(chunk['spectra'])
    identities = Counter(tuple(row['identity']) for row in rows)
    return {tuple(row['identity']):row for row in rows if identities[tuple(row['identity'])]==1}


def shape_class(row):
    value = row['shape']
    if value is None or not np.isfinite(value) or not 0 <= value <= 1:
        return 'unknown_r_shape'
    return 'pointlike_r' if value >= .8 else 'below_r_pointlike_threshold'


def summarize(rows):
    strata = {}
    for low in (21,22):
        view = [row for row in rows if low <= row['magnitude'] < low+1]
        subsamples = {
            'all': view,
            'single_sdss_run': [r for r in view if r['sdss_runs']==1],
            'multiple_sdss_runs': [r for r in view if r['sdss_runs']>1],
            'short_nsc_span': [r for r in view if r['nsc_span']<4],
            'long_nsc_span': [r for r in view if r['nsc_span']>=4],
        }
        for row in view:
            subsamples.setdefault(shape_class(row), []).append(row)
        strata[str(low)] = {key:dict(blind=merge_strata(part,('blind_all',)),
                                    strict_published=merge_strata(part,('blind_all','Xiao','NSC')))
                            for key,part in subsamples.items()}
    return strata


def spectral_opportunity(footprint, spectra, recovered):
    chips_name = 'private_transfer_chips.csv.gz'
    verify_directory(footprint, (chips_name, 'transfer_footprint_summary.json'))
    proof = read(footprint/'transfer_footprint_summary.json')
    plan_file = footprint/'frozen_plan.json'
    plan = read(plan_file)
    if proof['plan_sha256'] != digest(plan_file) or proof['chip_sha256'] != digest(footprint/chips_name):
        raise ValueError('Observation footprint no longer has its original inputs')
    graph = (footprint/'reference_receipt.json').resolve().parent/'private_c0.npz'
    if proof['reference_sha256'] != digest(graph):
        raise ValueError('Observation footprint belongs to a different reference frame')
    solution, gauges = read_offsets(graph, with_gauge=True)
    specs_file=footprint/'config_snapshot.toml'
    if digest(specs_file) != plan['geometry_sha256']:
        raise ValueError('Registered analysis cells changed after estimating native coverage')
    fields = tomllib.loads(specs_file.read_text())['fields']
    center = [float(np.mean([min(fields[core][key][0] for core in plan['cores']),
                            max(fields[core][key][1] for core in plan['cores'])]))
              for key in ('ra','dec')]
    chips = pd.read_csv(footprint/chips_name, dtype={'exposure':str})
    with fits.open(footprint/'window.fits',memmap=True) as handle:
        window = handle[1].data
        choose = sdss_window_mask(window,301)&sdss_window_selection(window,center,8.)
        columns = ('RA','DEC','RUN','CAMCOL','NODE','INCL','MU_START','MU_END','NU_START','NU_END')
        scans = pd.DataFrame({key:np.asarray(window[key][choose],float) for key in columns})
        tai = np.asarray(window['TAI'][choose],float)
    if tai.ndim==2:
        tai=tai[:,2]
    scans['epoch']=years(sdss_tai_to_tt(np.where((tai>=4e9)&(tai<6e9),tai,np.nan)))
    scans=scans.loc[np.isfinite(scans.epoch)]
    owners=np.zeros(len(spectra),bool)
    supported=np.zeros(len(spectra),bool)
    core_gauges=plan.get('core_gauges')
    if core_gauges is None:
        if len(set(gauges.values()))!=1:
            raise ValueError('Cannot assign legacy footprint to a unique reference gauge')
        core_gauges=dict.fromkeys(plan['cores'],next(iter(gauges.values())))
    for gauge in set(core_gauges.values()):
        region=np.zeros(len(spectra),bool)
        for core, value in core_gauges.items():
            if value==gauge:
                region|=inside(spectra.ra,spectra.dec,fields[core])
        if np.any(region&owners):
            raise ValueError('One spectral position maps to multiple reference gauges')
        owners|=region
        if gauge is None or not region.any():
            continue
        retained={key:value for key,value in solution.items() if gauges[key]==gauge}
        _,_,masks=opportunity_masks(chips,scans,spectra.loc[region,['ra','dec']].to_numpy(),center,retained)
        supported[region]=masks['all']
    found=np.array([(row.origin,float(row.ra),float(row.dec)) in recovered
                    for row in spectra.itertuples(index=False)])
    output={}
    for low in (21,22):
        band=spectra.reference_r.between(low,low+1,inclusive='left')
        output[str(low)]=dict(spectral_positions_in_analysis_cores=int(np.count_nonzero(band&owners)),
                             positions_with_calibrated_geometric_opportunity=int(np.count_nonzero(band&supported)),
                             recovered_with_calibrated_geometric_opportunity=int(np.count_nonzero(band&supported&found)),
                             recovered_without_registered_geometric_opportunity=int(np.count_nonzero(band&owners&~supported&found)))
    return output


def run(previous, previous_blind, batch, output, spectra_file=None):
    current = batch/'science'
    plan = read(batch/'reference/private_continuation_reference_plan.json')
    trace = read(batch/'reference/v19_reference_plan_summary.json')
    if trace['plan_sha256'] != digest(batch/'reference/private_continuation_reference_plan.json'):
        raise ValueError('Affected reference graph manifest changed')
    # Final owner keys determine all cohorts, including sources moved in halo overlap.
    affected = {core for part in plan['solve'].values() for core in part['region_by_core']}
    new = {line.split()[0] for line in (batch/'intake/tasks.tsv').read_text().splitlines()}
    replayed = affected & plan['archives'].keys()
    if affected != replayed|new:
        raise ValueError('Some affected cores have no recorded intake or predecessor')
    old = load_blind(previous,previous_blind)
    after = load_blind(current,batch/'blind_quality')
    before_replay = {k:r for k,r in old.items() if r['core'] in replayed}
    after_replay = {k:r for k,r in after.items() if r['core'] in replayed}
    paired = set(before_replay)&set(after_replay)
    before_rows = [before_replay[k] for k in sorted(paired)]
    after_rows = [after_replay[k] for k in sorted(paired)]
    scope_rows = dict(new=[r for r in after.values() if r['core'] in new],
                      before_replay=list(before_replay.values()), after_replay=list(after_replay.values()),
                      same_qso_before=before_rows,same_qso_after=after_rows,cumulative=list(after.values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(scope='Blind optical identities after final ownership. Same QSO before/after requires unique spectral identity in both signed cohorts. R morphology is the saved legacy r shape statistic and may be unavailable.',
                  source_catalogue_sha256=digest(current/'private_continuation_manifest.json'),
                  previous_catalogue_sha256=digest(previous/'private_continuation_manifest.json'),
                  replayed_core_count=len(replayed),new_core_count=len(new),
                  shared_replayed_spectra=len(paired),
                  exited_replayed_spectra=len(before_replay.keys()-after_replay.keys()),
                  entered_replayed_spectra=len(after_replay.keys()-before_replay.keys()),
                  samples={k:summarize(rows) for k,rows in scope_rows.items()},
                  slurm_job=os.environ['SLURM_JOB_ID'])
    if spectra_file is not None:
        verify_directory(spectra_file.parent,(spectra_file.name,))
        registered=read(batch/'registered/wide_selection_summary.json')
        if registered['spectral_metadata_sha256'] != digest(spectra_file):
            raise ValueError('Optical spectrum identities changed since prospective registration')
        spectra=pd.read_csv(spectra_file,usecols=('ra','dec','reference_r','origin'))
        spectra=spectra.loc[spectra.reference_r.between(21,23,inclusive='left')]
        recovered={key for key,row in after.items() if row['core'] in new}
        current_manifest=read(current/'private_continuation_manifest.json')
        windows={line.split()[1] for line in (batch/'intake/tasks.tsv').read_text().splitlines()}
        new_footprints={key:Path(value).parent for key,value in current_manifest['region_footprints'].items()
                        if key in windows}
        if new_footprints.keys()!=windows:
            raise ValueError('Some new geometrical footprints are missing')
        coverage={window:spectral_opportunity(path,spectra,recovered)
                  for window,path in sorted(new_footprints.items())}
        result['spectral_geometric_opportunity_by_window']=coverage
        result['spectral_geometric_opportunity_scope']='Optical spectrum positions in actual registered SDSS scan and NSC CCD footprints, measured availability of calibrated groups, at least two late exposures, at least 0.6 yr late and 5 yr joint spans; individual target depth and independent identity are separate.'
    check_aggregate(result)
    save_json(output,result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous','previous_blind','batch','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--spectra-file',type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use a Slurm compute node')
    run(**vars(args))
