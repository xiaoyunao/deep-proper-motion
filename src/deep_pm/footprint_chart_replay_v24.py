"""Pre-register and replay two physical neighbours under TAN and a shared wide chart."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import stereographic
from .astrometry import basis, project, unproject
from .coordinate_metric import (angular_distance, fit_vector_motion, forward_jacobian, native_covariance,
                                projected_covariance, reference_covariance)
from .full_reference_run import local_support
from .full_reference_solver import fit_component
from .io import digest, save_json
from .standard_motion import icrs_motion


def read(path):
    return json.loads(path.read_text())


def register(work, geometry):
    old=work.parent/'v2.2'
    plan=old/'registration/private_full_coordinate_plan.json'
    potential={}
    wide=read(geometry)['angular_tests'][0]['candidate_center_deg']
    for entry in read(plan)['cores']:
        core=entry['core']
        a,b=entry['spec']['ra']
        low,high=entry['spec']['dec']
        receipt=old/'input'/core/'v22_reference_input_core_summary.json'
        if not 5000<=read(receipt)['measurements']<=150000:
            continue
        potential.setdefault((a%360,low,high),[]).append(entry)
    pairs=[]
    for entries in potential.values():
        for left in entries:
            boundary=left['spec']['ra'][1]%360
            lo,hi=left['spec']['dec']
            for right in potential.get((boundary,lo,hi),[]):
                if left['window']!=right['window']:
                    cosine=basis(boundary,(lo+hi)/2)[0]@basis(*wide)[0]
                    pairs.append((cosine,left['core'],right['core'],boundary,(lo+hi)/2))
    if not pairs:
        raise ValueError('No eligible adjacent archived window pair')
    _,left,right,ra,dec=min(pairs)
    frozen=dict(cores=[left,right],tan_center=[ra,dec],stg_center=wide,
                predecessor_plan_sha256=digest(plan),geometry_sha256=digest(geometry),
                geometry_only_pairs=len(pairs),criteria=dict(fixed=20,duplicates=5,galaxies=20,
                angular_mas=.001,pm_median_masyr=.5,held_median_mas=5.,max_held_p68_ratio=1.1))
    dest=work/'chart-replay/private_registration.json'
    if dest.exists() and read(dest)!=frozen:
        raise ValueError('Changed chart replay registration')
    save_json(dest,frozen)
    return frozen


def prepare(frames):
    raw=pd.concat(frames,ignore_index=True).sort_values('source_core')
    identity=raw.groupby('detection_id').reference.nunique()
    rows=raw[~raw.detection_id.isin(identity[identity>1].index)].drop_duplicates('detection_id')
    clean=rows[~rows.duplicated(['reference','exposure_id'],keep=False)]
    if clean.groupby('reference').is_holdout.nunique().max()!=1:
        raise ValueError('Reference split changed at the real boundary')
    train=clean[~clean.is_holdout]
    while True:
        prev=len(train)
        support=train.groupby('group').reference.nunique()
        train=train[train.group.isin(support[support>=10].index)]
        visits=train.groupby('reference').group.nunique()
        train=train[train.reference.isin(visits[visits>=2].index)]
        if len(train)==prev:
            break
    names=np.array(sorted(set(train.group)),dtype=str)
    if not len(names):
        raise ValueError('No supported boundary groups')
    grp=pd.Index(names).get_indexer(train.group)
    roots=pd.Series(grp,index=train.index).groupby(train.reference).transform('min').to_numpy()
    graph=coo_matrix((np.ones(len(grp)),(roots,grp)),shape=(len(names),len(names)))
    _,components=connected_components(graph,directed=False)
    cross=[i for i in np.unique(components) if {s.split(':')[0] for s in names[components==i]}=={'SDSS','NSC'}]
    if not cross:
        raise ValueError('No cross-survey reference component at registered boundary')
    chosen=max(cross,key=lambda i:np.count_nonzero(components[grp]==i))
    names=names[components==chosen]
    train=train[train.group.isin(names)].sort_values(['reference','group','detection_id']).reset_index(drop=True)
    held=clean[clean.is_holdout & clean.group.isin(names)].reset_index(drop=True)
    return train,held,names


def solve(train,names,center,workers):
    xy=project(train.ra_deg,train.dec_deg,center)
    ptr=np.r_[0,np.cumsum(train.groupby('reference',sort=False).size())]
    relative=xy-np.repeat(xy[ptr[:-1]],np.diff(ptr),axis=0)
    covariance=projected_covariance(train.ra_deg,train.dec_deg,center,native_covariance(train[['sigma_e','sigma_n']].to_numpy()))
    inverse=np.linalg.inv(covariance)
    groups=pd.Index(names).get_indexer(train.group).astype('int32')
    point,draws,_,_=fit_component(ptr,groups,inverse[:,[0,0,1],[0,1,1]],relative,names,draws=48,seed=45116,draw_workers=workers)
    nsc=np.char.startswith(names,'NSC:')
    for table in (point,draws):
        np.testing.assert_allclose(np.median(table[nsc],axis=0),0,atol=1e-8,rtol=0)
    return center,point,draws


def held_readout(held,names,fits):
    common=np.ones(len(held),bool)
    residuals=[]
    tags=pd.Index(names).get_indexer(held.group)
    for center,point,draws in fits:
        keep,_=local_support(held.ra_deg,held.dec_deg,center,held.group,names,draws)
        common &= keep
        raw=project(held.ra_deg,held.dec_deg,center)
        residuals.append(unproject(raw-point[tags],center))
    data=held[common]
    sample=[]
    for _,block in data.groupby('reference'):
        ix=block.index.to_numpy()
        early=block.group.str.startswith('SDSS:').to_numpy()
        if early.all() or not early.any():
            continue
        pivot=(block.ra_deg.iloc[0],block.dec_deg.iloc[0])
        offsets=[project(*sky[ix].T,pivot) for sky in residuals]
        sample.append([np.median(x[early],axis=0)-np.median(x[~early],axis=0) for x in offsets])
    return np.asarray(sample).reshape(-1,2,2),int(common.sum())


def fit_fixed(data,time,sigma,names,solution):
    center,point,draws=solution
    group=pd.Index(names).get_indexer(data.group)
    if group.min()<0:
        raise ValueError('Unsolved fixed member')
    xy=project(data.ra_deg,data.dec_deg,center)
    cov=projected_covariance(data.ra_deg,data.dec_deg,center,native_covariance(sigma))
    model=fit_vector_motion(xy-point[group],time,cov,2015.5)
    common,_=reference_covariance(model.influence,draws[group].transpose(1,0,2))
    answer=icrs_motion(*model.coeff,center,model.cov+common)
    full=answer['covariance']
    np.testing.assert_allclose(full,full.T,rtol=1e-10,atol=1e-8)
    if np.linalg.eigvalsh(full).min()<=0:
        raise ValueError('Nonpositive state covariance')
    return answer


def membership_key(ids):
    return sha256(('v24-before-chart-replay:'+'|'.join(sorted(ids))).encode()).hexdigest()


def fixed_stars(old,entries,names,fits):
    from .__main__ import tomllib
    from .continuous_reference import PRECISION,SPATIAL
    from .pilot_production import quality_detections
    from .precision_v03 import calibration_groups
    from .stream_json import array_items
    inputs=[];tracks=[]
    for entry in entries:
        source=Path(entry['source'])
        if digest(source/'run_manifest.json')!=entry['source_receipt_sha256']:
            raise ValueError('Changed archived source input')
        data,_,time,sigma,_=quality_detections(source,entry['center'],tomllib.loads(PRECISION.read_text()),tomllib.loads(SPATIAL.read_text())['production'])
        data=data.assign(group=calibration_groups(data)).set_index('detection_id',drop=False)
        if not data.index.is_unique:
            raise ValueError('Nonunique archived measurement')
        inputs.append((data,time,sigma))
        dest=old/'sources'/entry['core']
        path=dest/'private_standard_motion.json.gz'
        receipt=read(next(dest.glob('v22*source*summary.json')))
        if digest(path)!=receipt['private_output_sha256'][path.name]:
            raise ValueError('Archived fixed membership changed')
        tracks.append(sorted((membership_key(r['detection_ids']),sorted(r['detection_ids'])) for r in array_items(path)))
    both=set(inputs[0][0].index)&set(inputs[1][0].index)
    repeated=sorted({key:ids for table in tracks for key,ids in table if set(ids)<=both}.items())[:64]
    outcomes=Counter()
    comparison=[]
    dup_errors=[]
    for side,table in enumerate([tracks[0][:64],tracks[1][:64],repeated]):
        for _,ids in table:
            side0=side%2
            datasets=[side0] if side<2 else [0,1]
            fitted=[]
            for k in datasets:
                df,t,s=inputs[k];ix=df.index.get_indexer(ids)
                if (ix<0).any():
                    outcomes['missing_fixed_members']+=1;break
                data=df.iloc[ix]
                if not data.group.isin(names).all():
                    outcomes['missing_local_physical_group']+=1;break
                supported=all(local_support(data.ra_deg,data.dec_deg,c,data.group,names,d)[0].all() for c,_,d in fits)
                if not supported:
                    outcomes['outside_common_native_support']+=1;break
                fitted.append([fit_fixed(data,t[ix],s[ix],names,f) for f in fits])
            if len(fitted)!=len(datasets):
                continue
            tan,wide=fitted[0]
            if side<2:
                comparison.append([wide['pm_icrs_masyr']-tan['pm_icrs_masyr'],wide['pm_error_masyr']/tan['pm_error_masyr']])
            else:
                first,second=wide,fitted[1][1]
                for q in ['position_deg','pm_icrs_masyr','covariance']:
                    np.testing.assert_allclose(first[q],second[q],rtol=1e-12,atol=1e-8)
                dup_errors.append(max(abs(first['pm_icrs_masyr']-second['pm_icrs_masyr'])))
    return np.asarray(comparison).reshape(-1,2,2),dict(original_by_side=list(map(len,tracks)),
             selected_by_side=[min(64,len(t)) for t in tracks],overlap_selected=len(repeated),
             overlap_reproduced=len(dup_errors),overlap_max_pm_difference=max(dup_errors,default=None),exits=dict(outcomes))


def evaluate(work,geometry):
    old=work.parent/'v2.2'
    spec=read(work/'chart-replay/private_registration.json')
    plan=old/'registration/private_full_coordinate_plan.json'
    if digest(plan)!=spec['predecessor_plan_sha256'] or digest(geometry)!=spec['geometry_sha256']:
        raise ValueError('Changed preregistered geometry')
    lookup={r['core']:r for r in read(plan)['cores']}
    entries=[lookup[k] for k in spec['cores']]
    frames=[]
    for e in entries:
        core=old/'input'/e['core'];file=core/'reference.parquet'
        if digest(file)!=read(core/'v22_reference_input_core_summary.json')['parquet_sha256']:
            raise ValueError('Changed calibration input')
        frames.append(pd.read_parquet(file))
    train,held,names=prepare(frames)
    centers=[spec['tan_center'],stereographic.Stereographic(spec['stg_center'])]
    workers=min(4,int(os.environ['SLURM_CPUS_PER_TASK']))
    solutions=[solve(train,names,c,workers) for c in centers]
    # Apply precisely the SAME physical predecessor correction in both coordinate representations.
    xy=project(held.ra_deg,held.dec_deg,centers[0])-solutions[0][1][pd.Index(names).get_indexer(held.group)]
    sky=unproject(xy,centers[0]);wide=project(*sky.T,centers[1])
    back=project(*unproject(wide,centers[1]).T,centers[0])
    maximum=float(np.max(angular_distance(xy,back,centers[0]))) if len(xy) else None
    residual,n=held_readout(held,names,solutions)
    stars,overlap=fixed_stars(old,entries,names,solutions)
    p68=np.quantile(np.linalg.norm(residual,axis=2),.68,axis=0) if len(residual) else None
    delta=np.median(residual[:,1]-residual[:,0],axis=0) if len(residual) else None
    mu=np.median(stars[:,0],axis=0) if len(stars) else None
    crit=spec['criteria']
    checks=dict(fixed_stars=len(stars)>=crit['fixed'],overlap=overlap['overlap_reproduced']>=crit['duplicates'],
                held_objects=len(residual)>=crit['galaxies'],angular=maximum is not None and maximum<crit['angular_mas'],
                median_pm=mu is not None and max(abs(mu))<=crit['pm_median_masyr'],
                median_held=delta is not None and max(abs(delta))<=crit['held_median_mas'],
                held_scatter=p68 is not None and p68[1]<=crit['max_held_p68_ratio']*p68[0])
    result=dict(qualified=all(checks.values()),checks=checks,slurm_job=os.environ['SLURM_JOB_ID'],
        registration_sha256=digest(work/'chart-replay/private_registration.json'),implementation_sha256=digest(Path(__file__)),
        physical_groups=len(names),training_objects=train.reference.nunique(),training_measurements=len(train),
        holdout_objects=held.reference.nunique(),common_supported_held_measurements=n,common_early_late_objects=len(residual),
        held_radial_p68_mas_TAN_STG=p68,held_median_shift_mas=delta,common_fixed_stars=len(stars),
        median_pm_shift_masyr=mu,pm_absolute_delta_p68_masyr=np.quantile(abs(stars[:,0]),.68,axis=0) if len(stars) else None,
        median_pm_se_ratio=np.median(stars[:,1],axis=0) if len(stars) else None,actual_boundary=overlap,
        fixed_correction_roundtrip_max_mas=maximum,reference_draws=48,centers_deg=centers,
        scope='Same two archived neighbouring cores, fixed physical detections, independent held galaxies. Free offsets solved jointly per chart; local seam and numerical test. Final whole-domain admission and independent quality are separate.')
    save_json(work/'registration/v24_stereographic_replay_summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('work',type=Path);p.add_argument('geometry',type=Path)
    p.add_argument('mode',choices=['register','evaluate','both'])
    a=p.parse_args()
    if not os.getenv('SLURM_JOB_ID'):
        p.error('Use an allocated compute node')
    if a.mode in ('register','both'):
        register(a.work,a.geometry)
    if a.mode in ('evaluate','both'):
        evaluate(a.work,a.geometry)
