"""Audit completed independent batches without opening the sealed test region."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from .__main__ import tomllib
from .astrometry import project
from .continuous_reference import verify_directory
from .continuous_summary import pool, estimate, METHODS
from .io import digest, save_json
from .nsc_comparison import comparison
from .public_summary import check_aggregate
from .standard_motion import icrs_motion


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
        out={}
        for name,rows in sets.items():
            values=[{k:(r['motion'][k],r['error_icrs'][k]) for k in METHODS} for r in rows]
            stats=comparison(values,settings,(('C0','NSC'),('C0','Xiao')))
            out[name]=dict(common=stats,physical_group_sensitivity=estimate(rows,METHODS,9137),
                sampling_columns=['sdss_span_years','nsc_span_years','joint_span_years','calendar_year_count'],
                sampling_p16_p50_p84=np.quantile([[r[k] for k in ('sdss_span_years','nsc_span_years','joint_span_years','calendar_year_count')] for r in rows],[.16,.5,.84],axis=0) if rows else None)
        grouped[f'r{low}_{low+1}']=dict(eligible_count=len(selected),strict_common_count=len(common),conditions=out)
    return grouped


def run(state,protocol,output):
    sealed=tomllib.loads(protocol.read_text())['final_test_box']
    tasks=[line.split() for line in (state/'tasks.tsv').read_text().splitlines()]
    allowed=[(name,box) for name,box in tasks if box!=sealed]
    # Filter BEFORE any fit/reference file access, not after loading outcomes.
    specs, fits, graphs, sets={}, {}, {}, defaultdict(list)
    records=defaultdict(list)
    for name,box in allowed:
        config=tomllib.loads((state/f'batch-{box}.toml').read_text())
        specs[name]=config['fields'][name]
        fits[name]=state/'outputs'/name
        graphs[box]=state/'shared'/box
        sets[box].append(name)
        verify_directory(fits[name],('private_qso_blocks.json','continuous_core_summary.json'))
        for row in json.loads((fits[name]/'private_qso_blocks.json').read_text()):
            records[box].append(standard_block(row,config['projection_center_deg']))
    # Reuse receipt/ownership checks, avoiding duplicated tangent-frame inference
    # and enormous per-core copies of the same cohort comparisons.
    public=pool(specs,fits,graphs,{})
    public.pop('cohorts')
    for core in public['cores'].values():
        core['qso_counts']=core.pop('qso')['totals']
        core.pop('gaia')
    public['standard_icrs_comparison']={box:comparison_cohorts(rows) for box,rows in records.items()}
    public['standard_icrs_comparison']['pooled']=comparison_cohorts([r for rows in records.values() for r in rows])
    public.update(final_test_box=sealed,sealed_core_count=sum(box==sealed for _,box in tasks),
        test_results_read=False,protocol_sha256=digest(protocol),
        vector_conversion='Tangent velocity multiplied by inverse-gnomonic Jacobian; old private published inverse-conversion values ignored.',
        inference_scope='Known-position spectroscopic QSO. Whole-region holdout unopened. Physical-group weight sensitivity is not independent spatial confidence.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(public)
    save_json(output,public)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','protocol','output'):p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    run(a.state,a.protocol,a.output)


if __name__=='__main__':main()
