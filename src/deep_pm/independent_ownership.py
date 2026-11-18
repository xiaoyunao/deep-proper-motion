"""Check actual source ownership and cross-core duplicates before production."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_ownership import close_pairs
from .continuous_reference import verify_directory
from .exposure_catalogues import inside
from .independent_summary import validate_release
from .io import digest,save_json
from .public_summary import check_aggregate


def audit(state,protocol,output,release=None):
    sealed=tomllib.loads(protocol.read_text())['final_test_box']
    release_sha=validate_release(state,protocol,release) if release is not None else None
    tasks=[line.split() for line in (state/'tasks.tsv').read_text().splitlines()]
    allowed=[(core,box) for core,box in tasks if release_sha is not None or box!=sealed]
    frames=[];centers={};outside=0;receipts={};components={}
    for core,box in allowed:
        work=state/'outputs'/core
        receipt=verify_directory(work,('private_catalogue.csv.gz','continuous_core_summary.json'))
        summary=json.loads((work/'continuous_core_summary.json').read_text())
        graph=state/'shared'/box/'private_c0.npz'
        if summary['shared_solution_sha256']!=digest(graph):raise ValueError('Source reference solution changed')
        config=tomllib.loads((state/f'batch-{box}.toml').read_text())
        centers[core]=config['projection_center_deg']
        rows=pd.read_csv(work/'private_catalogue.csv.gz',usecols=['ra','dec','pm_east','pm_north'])
        if len(rows)!=summary['owned_w0'] or not np.isfinite(rows.to_numpy()).all():raise ValueError('Invalid source count or coordinates')
        outside+=int((~inside(rows.ra,rows.dec,config['fields'][core])).sum())
        frames.append(rows.assign(core=core));receipts[core]=digest(receipt)
        components[core]=str(summary['reference_gauge'])
    data=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame({
        name:pd.Series(dtype=str if name=='core' else float) for name in ('ra','dec','pm_east','pm_north','core')})
    result=close_pairs(data,centers)
    result.update(outside_common_epoch_owner=outside,core_count=len(allowed),source_receipt_sha256=receipts,
        applied_reference_components=components,test_results_read=release_sha is not None,test_release_sha256=release_sha,
        motion_comparison_frame='Per-source ICRS proper motion, transformed only for close-pair candidates.',
        scope='Geometric duplicates and candidate same-source pairs; no claim of catalogue purity or reference-zero certification.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result);save_json(output,result)
    if outside:raise ValueError('Sources outside their common-epoch core')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('state','protocol','output'):p.add_argument(name,type=Path)
    p.add_argument('--release',type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    audit(**vars(a))


if __name__=='__main__':main()
