"""Refit frozen overlap members with one shared physical reference solution."""
import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .allband_qso import evaluated_fit
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .cross_region_audit import source_check
from .io import save_json,digest
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,calibration_support
from .public_summary import check_aggregate
from .provenance import run_stage


def replay(fit,graph):
    verify_directory(fit,('private_standard_motion.json.gz','continuous_core_summary.json'))
    source=(fit/'source_receipt.json').resolve().parent
    verify_directory(source,('detections.csv',))
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    center=reference_frame(graph);data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production'])
    solution,gauges=read_offsets(graph,with_gauge=True);keys=calibration_groups(data)
    with gzip.open(fit/'private_standard_motion.json.gz','rt') as stream:original=json.load(stream)
    lookup=pd.Index(data.detection_id)
    if not lookup.is_unique:raise ValueError('Physical source detections are not unique')
    records=[];missing=0;split=0
    for row in original:
        ix=lookup.get_indexer(row['detection_ids'])
        if (ix<0).any() or not keep[ix].all():raise ValueError('Original frozen r members are missing or changed')
        if not calibration_support(keys[ix],solution).all():missing+=1;continue
        if len({gauges[key] for key in keys[ix]})!=1:split+=1;continue
        result=evaluated_fit(ix,data,xy,time,sigma,keys,solution,center,config['reference_epoch'])
        records.append(dict(row,**{k:result[k] for k in ('position_deg','pm_icrs_masyr','pm_error_masyr','covariance')},projection_center_deg=center))
    return original,records,dict(original_source_count=len(original),replayed_source_count=len(records),missing_reference_group=missing,split_reference_gauge=split)


def run(work,original,graph):
    result={}
    for band in range(28,34):
        originals=[];new=[];support=[]
        for side in ('left','right'):
            old,now,count=replay(original/f'{side}_{band}',graph)
            originals.append(old);new.append(now);support.append(count)
        result[f'dec{band}_{band+1}']=dict(original_independent_graphs=source_check(*originals,True),
            joint_graph_fixed_members=source_check(*new,True),replay_counts=support)
    public=dict(strips=result,graph_sha256=digest(graph),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='All original source members replayed without association or quality reselection. Identical detections in the same frame should agree by construction; this is a consistency check, not independent astrometric precision or absolute spatial-zero certification. Changed member sets reported separately; common reference errors are not divided by source count.')
    check_aggregate(public);save_json(work/'fixed_overlap_reference_summary.json',public)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','original','graph'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'joint_reference_receipt.json',verify_directory(a.graph.parent,(a.graph.name,)))
    names=['joint_reference_receipt.json']
    for band in range(28,34):
        for side in ('left','right'):
            alias=f'{side}_{band}_receipt.json'
            bind(a.work,alias,verify_directory(a.original/f'{side}_{band}',('private_standard_motion.json.gz','continuous_core_summary.json')))
            names.append(alias)
    with run_stage(a.work,PRECISION,'six_north_overlap_strips','fixed-overlap-reference-v14',inputs=names,outputs=('fixed_overlap_reference_summary.json',)) as go:
        if go:run(**vars(a))


if __name__=='__main__':main()
