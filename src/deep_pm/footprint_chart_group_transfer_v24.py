"""Measure how a single planar group offset transports across an actual scan or exposure."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .astrometry import MAS_PER_RADIAN,project,unproject
from .coordinate_metric import sky_basis
from .io import digest,save_json
from .stereographic import Stereographic


def summarize(frame):
    if frame.empty:return dict(detections=0,groups=0)
    r=frame.error_mas
    return dict(detections=len(frame),groups=frame.group.nunique(),
                median_east_north_mas=frame[['east','north']].median().to_numpy(),
                radial_mas_p50_p68_p95=np.quantile(r,[.5,.68,.95]),
                above_5mas=int((r>5).sum()),group_median_error_p50_p95=np.quantile(
                    frame.groupby('group').error_mas.median(),[.5,.95]))


def compare(frame,center):
    # Split physical galaxies consistently for every exposure and repeated visit.
    labels=frame.reference.unique()
    half={k:sha256(('v24-group-transfer:'+k).encode()).digest()[0]%2 for k in labels}
    f=frame.assign(half=frame.reference.map(half).to_numpy()).copy()
    n=f.groupby(['group','half']).reference.nunique().unstack(fill_value=0).reindex(columns=[0,1],fill_value=0)
    f=f[f.group.isin(n.index[n.min(axis=1)>=3])].reset_index(drop=True)
    if not len(f):return f,pd.DataFrame()
    raw=project(f.ra_deg,f.dec_deg,center)
    old=project(f.old_ra,f.old_dec,center)
    f[['dx','dy']]=raw-old
    model=f[f.half==0].groupby('group')[['dx','dy']].median()
    ix=f.half.eq(1).to_numpy()
    result=f[ix].copy()
    corrected=unproject(raw[ix]-model.loc[result.group].to_numpy(),center)
    local=sky_basis(result.old_ra,result.old_dec)
    v=sky_basis(*corrected.T)[:,0]
    dots=np.einsum('nik,nk->ni',local,v)
    result[['east','north']]=np.arctan2(dots[:,1:],dots[:,:1])*MAS_PER_RADIAN
    result['error_mas']=np.arctan2(np.linalg.norm(dots[:,1:],axis=1),dots[:,0])*MAS_PER_RADIAN
    # An actual pair of directions gives a lower bound on the angular diameter.
    xyz=sky_basis(f.ra_deg,f.dec_deg)[:,0,:]
    for j,key in enumerate(['x','y','z']):f[key]=xyz[:,j]
    cols=['x','y','z']
    first=f.groupby('group')[cols].first().loc[f.group].to_numpy()
    f['dot']=(first*xyz).sum(axis=1)
    second=f.loc[f.groupby('group').dot.idxmin()].set_index('group')[cols].loc[f.group].to_numpy()
    f['dot']=(second*xyz).sum(axis=1)
    span=np.degrees(np.arccos(np.clip(f.groupby('group').dot.min(),-1,1)))
    result['observed_span']=result.group.map(span)
    return f,result


def run(work,geom):
    reference=work/'predecessor-held/v24_previous_held_reference_summary.json'
    proof=json.loads(reference.read_text())
    regions=[]
    for chart in json.loads(geom.read_text())['angular_tests']:
        frames=[]
        for c in chart['predecessors']:
            path=reference.parent/f'held-{c:04d}.parquet'
            if digest(path)!=proof['components'][str(c)]['sha256']:raise ValueError('Changed original reference directions')
            frames.append(pd.read_parquet(path))
        frame=pd.concat(frames,ignore_index=True)
        if frame.detection_id.duplicated().any():raise ValueError('Conflicting physical detection identity')
        donors,out=compare(frame[frame.old_supported],Stereographic(chart['candidate_center_deg']))
        stats={}
        if not out.empty:
            for survey in ['SDSS','NSC']:
                subset=out[out.group.str.startswith(survey+':')]
                for name,part in [('all',subset),('under_2deg',subset[subset.observed_span<2]),
                        ('2_to_10deg',subset[subset.observed_span.between(2,10,inclusive='left')]),
                        ('at_least_10deg',subset[subset.observed_span>=10])]:
                    stats[survey+'_'+name]=summarize(part)
        regions.append(dict(old_components=chart['predecessors'],old_supported_detections=int(frame.old_supported.sum()),
                            eligible_detections=len(donors),test_detections=len(out),statistics=stats))
    save_json(geom.parent/'v24_chart_group_transfer_summary.json',dict(slurm_job=os.environ['SLURM_JOB_ID'],
            geometry_sha256=digest(geom),old_held_sha256=digest(reference),components=regions,
            scope='Geometric representation only. For each archived group re-express its old corrected sky in STG, estimate one constant on half of its held galaxies, compare to the exact old corrected sky at disjoint identities. This does not train the final reference, estimate a physical sky truth or modify PM. Require three identities per half. Angular size is the separation of an actual pair, a lower bound.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('work',type=Path);p.add_argument('geometry',type=Path)
    a=p.parse_args()
    if not os.getenv('SLURM_JOB_ID'):p.error('Use assigned Slurm CPUs')
    run(a.work,a.geometry)
