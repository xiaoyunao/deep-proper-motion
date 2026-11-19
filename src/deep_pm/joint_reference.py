"""Reproject fixed reference identities into one physical N1/N3 graph.

This changes only the declared tangent coordinate basis. Native angular errors
retain the frozen C0 plane approximation; neither galaxies nor split identities
are reselected, and no target proper motions enter this operation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .astrometry import project
from .continuous_reference import bind,common_reference_frame,verify_directory
from .io import digest,save_json
from .provenance import run_stage
from .public_summary import check_aggregate


def identity_split(frame):
    return sha256(frame[['detection_id','reference','is_holdout']].to_csv(index=False).encode()).hexdigest()


def reproject(frame,center):
    out=frame.copy()
    out[['east','north']]=project(out.ra_deg,out.dec_deg,center)
    if identity_split(frame)!=identity_split(out):raise ValueError('Physical reference identities or split changed')
    np.testing.assert_array_equal(frame[['sigma_e','sigma_n']],out[['sigma_e','sigma_n']])
    return out


def source_projection(frame,metadata):
    center=common_reference_frame([metadata])
    if 'projection_center_deg' not in metadata:
        # v1.2 used a fixed center and recorded only the projected coordinates.
        original=project(frame.ra_deg,frame.dec_deg,center)
        if not np.allclose(original,frame[['east','north']],rtol=0,atol=1e-4):
            raise ValueError('Stored legacy coordinates disagree with the v1.2 projection')
    return center


def prepare(work,state,boxes,center,workers):
    parts={};fields={}
    for box in boxes:
        original=json.loads((state/f'references-{box}.json').read_text())
        config=tomllib.loads((state/f'batch-{box}.toml').read_text())
        for name,path in original.items():
            if name in parts:raise ValueError('Duplicate core identifier across regions')
            parts[name]=Path(path);fields[name]=config['fields'][name]
    config=work/'joint_geometry.toml'
    text=['reference_epoch = 2015.5','nside = 128',f'projection_center_deg = {list(center)}']
    for name,spec in fields.items():text += [f'\n[fields.{name}]',f'ra = {spec["ra"]}',f'dec = {spec["dec"]}']
    content='\n'.join(text)+'\n'
    if config.exists() and config.read_text()!=content:raise ValueError('Joint geometry changed')
    if not config.exists():config.write_text(content)
    def shard(item):
        name,source=item;target=work/'refs'/name;target.mkdir(parents=True,exist_ok=True)
        receipt=verify_directory(source,('reference.csv.gz','reference_summary.json'))
        bind(target,'original_reference_receipt.json',receipt)
        with run_stage(target,config,name,'joint-reference-projection-v14',inputs=('original_reference_receipt.json',),outputs=('reference.csv.gz','reference_summary.json')) as go:
            if go:
                frame=pd.read_csv(source/'reference.csv.gz',dtype={'reference':str,'is_holdout':bool})
                converted=reproject(frame,center);converted.to_csv(target/'reference.csv.gz',index=False)
                meta=json.loads((source/'reference_summary.json').read_text())
                original_center=list(source_projection(frame,meta))
                meta.update(projection_center_deg=list(center),original_projection_center_deg=original_center,
                    preserved_identity_split_sha256=identity_split(frame),original_reference_receipt_sha256=digest(receipt),
                    original_native_plane_error_approximation_unchanged=True)
                save_json(target/'reference_summary.json',meta)
        meta=json.loads((target/'reference_summary.json').read_text())
        return name,meta['preserved_identity_split_sha256'],meta['local_reference_rows']
    with ThreadPoolExecutor(max_workers=workers) as pool:checks=list(pool.map(shard,parts.items()))
    plan={name:str((work/'refs'/name).absolute()) for name in parts}
    save_json(work/'private_joint_reference_plan.json',plan)
    result=dict(core_count=len(parts),input_regions=boxes,projection_center_deg=list(center),
        reprojected_measurement_count=sum(v[2] for v in checks),identity_split_sha256={k:h for k,h,_ in checks},
        slurm_job=os.environ['SLURM_JOB_ID'],scope='All original reference detections and original train/holdout labels retained; one coordinate basis for shared physical calibration groups; no QSO tuning or stellar mean-motion constraint.')
    check_aggregate(result);save_json(work/'joint_preparation_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('work',type=Path);p.add_argument('state',type=Path)
    p.add_argument('--boxes',nargs='+',default=['north_one','north_three'])
    p.add_argument('--center',nargs=2,type=float,default=[188.,33.]);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);prepare(a.work,a.state,a.boxes,a.center,a.workers)


if __name__=='__main__':main()
