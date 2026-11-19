"""Replay original C0 on the identical local D0 reference population.

This diagnoses whether a global/local difference can be a solver effect; it
neither adds a DCR parameter nor adopts independent per-core reference frames.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_reference import PRECISION,bind,calibration_components,verify_directory
from .dcr_validation import read_model
from .io import save_json
from .provenance import run_stage
from .public_summary import check_aggregate
from .zero_point import fit_offsets


def run(work,trial):
    config=tomllib.loads(PRECISION.read_text());cal=config['calibration'];rows=pd.read_csv(trial/'private_reference_dcr.csv.gz',dtype={'reference':str,'is_holdout':bool});train=rows.loc[~rows.is_holdout]
    part=max(calibration_components(train,cal['min_references']),key=lambda p:p.reference.nunique()).reset_index(drop=True)
    fit=fit_offsets(part.reference,part[['east','north']].to_numpy(),part[['sigma_e','sigma_n']].to_numpy(),part.group.to_numpy(),cal['min_references'],cal['bootstrap_draws'],config['seed'])
    late=[v for k,v in fit.items() if k.startswith('NSC:')];anchor=np.median([v[0] for v in late],axis=0);datum=np.median([v[1]-v[1].mean(axis=0) for v in late],axis=0);old={}
    for key,(point,samples,n) in fit.items():
        samples=samples-samples.mean(axis=0)-datum
        if np.std(samples,axis=0,ddof=1).max()<=cal['max_offset_se_mas']:old[key]=(point-anchor,samples,n)
    d0=read_model(trial/'private_D0_local_C0.npz')['base'];keys=sorted(set(old)&set(d0));point=np.array([old[k][0]-d0[k][0] for k in keys]);sigma=np.array([np.std(old[k][1],axis=0,ddof=1)-np.std(d0[k][1],axis=0,ddof=1) for k in keys])
    result=dict(local_reference_sources=part.reference.nunique(),local_measurements=len(part),original_solver_supported_groups=len(old),D0_supported_groups=len(d0),common_groups=len(keys),same_supported_groups=set(old)==set(d0),
        maximum_absolute_group_offset_change_mas=float(abs(point).max()),group_offset_change_p16_p50_p84_mas=np.quantile(point,[.16,.5,.84],axis=0),maximum_absolute_bootstrap_se_change_mas=float(abs(sigma).max()),
        slurm_job=os.environ['SLURM_JOB_ID'],scope='Original zero_point.fit_offsets and frozen local no-DCR D0 use exactly the same physical training references, coordinates, native+floor errors, group keys, bootstrap seed and median-NSC gauge. Original tolerance1e-9 versus local1e-10 is retained. Both already center each reference before solving. This is a solver-equivalence diagnostic, not a new calibration/DCR model or permission to adopt local frames independently.')
    check_aggregate(result);save_json(work/'dcr_control_replay_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('work',type=Path);p.add_argument('trial',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'trial_receipt.json',verify_directory(a.trial,('private_reference_dcr.csv.gz','private_D0_local_C0.npz')))
    with run_stage(a.work,PRECISION,'local_C0_equivalence','dcr-control-replay-v15',inputs=('trial_receipt.json',),outputs=('dcr_control_replay_summary.json',)) as go:
        if go:run(a.work,a.trial)


if __name__=='__main__':main()
