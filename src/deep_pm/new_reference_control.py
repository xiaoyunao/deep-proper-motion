"""Stationary known-position controls where the frozen graph had no usable fit."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .allband_qso import physical_candidates
from .astrometry import project
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .exposure_catalogues import inside
from .exposure_pilot import contrast_samples,matched_legacy
from .independent_summary import standard_block
from .io import save_json
from .nsc_comparison import qso_matches,comparison
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .provenance import run_stage
from .public_summary import check_aggregate
from .regional_validation import official_table


def run(work,source,shared,spec,legacy):
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());settings.update(bootstrap_draws=400,bootstrap_absolute_zero=True)
    graph=shared/'private_c0.npz';center=reference_frame(graph);solution,gauges=read_offsets(graph,with_gauge=True)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
    votes=pd.Series(keys[keep&inside(data.ra_deg,data.dec_deg,spec)]).map(gauges).value_counts()
    gauge=min(votes.index,key=lambda k:(-votes[k],k)) if len(votes) else None;solution={k:v for k,v in solution.items() if gauges[k]==gauge}
    qso=pd.read_csv(source/'spectra_union.csv');qso=qso.loc[inside(qso.ra,qso.dec,spec)].reset_index(drop=True);points=project(qso.ra,qso.dec,center)
    labels=pd.read_csv(source/'external_labels.csv',dtype=str).set_index('detection_id').original_object_id
    official=(labels,official_table(source,{}));archive=pd.read_csv(legacy);archive=archive.loc[inside(archive.RA,archive.DEC,spec,.06)];old,_=matched_legacy(archive,points,center)
    magnitude=qso.reference_r.to_numpy();zero=np.zeros((len(qso),2));models={};private={}
    for name,mask in [('r_only',keep),('all_nsc',keep|(nsc&physical_candidates(data)))]:
        lab,truth=qso_matches(data,xy,mask,points);blocks=[]
        contrast_samples(truth,lab,data,xy,time,sigma,mask,nsc,settings,config,(keys,),(solution,),center,magnitude,qso.origin.to_numpy(),zero,zero,official,old,[],{},blocks)
        private[name]=blocks
        restored=[standard_block(r,center) for r in blocks]
        models[name]={}
        for low in (21,22):
            sample=[r for r in restored if low<=r['magnitude']<low+1];common=[r for r in sample if {'C0','NSC','Xiao'}<=r['motion'].keys()]
            models[name]['r'+str(low)]=dict(known_stationary_sources=len(sample),strict_common=comparison([{k:(r['motion'][k],r['error_icrs'][k]) for k in ('C0','NSC','Xiao')} for r in common],settings,(('C0','Xiao'),('C0','NSC'))))
    result=dict(models=models,reference_gauge=gauge,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Known stationary spectral positions; no target proper motions used in associations or fits. All original pre-SE reference draws retained. Separate r/all samples until paired by physical identity. C0 here denotes an experimental high-SE support sensitivity, not the frozen adopted graph or blind-source accuracy.',
        adoption='not_adopted_pending_reference_and_held_source_validation')
    check_aggregate(result);save_json(work/'new_reference_control_summary.json',result);save_json(work/'private_known_qso_blocks.json',private)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','source','shared','geometry','legacy'):p.add_argument(n,type=Path)
    p.add_argument('region');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'source_receipt.json',verify_directory(a.source,('detections.csv','spectra_union.csv','external_labels.csv','official_nsc.csv')))
    bind(a.work,'reference_receipt.json',verify_directory(a.shared,('private_c0.npz',)));bind(a.work,'legacy.csv',a.legacy);bind(a.work,'geometry.toml',a.geometry)
    with run_stage(a.work,PRECISION,a.region,'new-reference-controls-v14',inputs=('source_receipt.json','reference_receipt.json','legacy.csv','geometry.toml'),outputs=('new_reference_control_summary.json','private_known_qso_blocks.json')) as go:
        if go:run(a.work,a.source,a.shared,tomllib.loads(a.geometry.read_text())['fields'][a.region],a.legacy)


if __name__=='__main__':main()
