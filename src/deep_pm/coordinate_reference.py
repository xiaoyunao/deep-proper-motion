"""Bounded two-axis reference-weight experiment, with fixed external C0 boundary.

This is not a replacement global graph. Draws are paired between the two bounded
solutions and conditional on the external offsets, not independent additions to
the archived global reference covariance.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr

from .blind_catalogue_quality import read
from .continuous_reference import read_offsets
from .coordinate_metric import native_covariance,projected_covariance,forward_jacobian
from .io import digest,save_json
from .public_summary import check_aggregate
from .usability_cluster import summarize


def bounded_fit(frame,covariance,free,draws=48):
    """Joint static positions and free group increments; every source tied out."""
    refs,identities=pd.factorize(frame.reference,sort=True)
    groups=pd.Index(free).get_indexer(frame.group);nr=len(identities);nf=len(free)
    if any(not np.any(groups[refs==r]<0) for r in range(nr)):
        raise ValueError('Every bounded reference must tie to the fixed boundary')
    y=frame[['corrected_e','corrected_n']].to_numpy()
    origin=np.asarray([np.median(y[refs==r],axis=0) for r in range(nr)])
    y=y-origin[refs]
    white=np.linalg.inv(np.linalg.cholesky(covariance))
    rr=[];cc=[];vv=[]
    for a in range(2):
        for b in range(2):
            rr.extend((2*np.arange(len(frame))+a).tolist());cc.extend((2*refs+b).tolist());vv.extend(white[:,a,b].tolist())
            ok=groups>=0
            rr.extend((2*np.flatnonzero(ok)+a).tolist());cc.extend((2*nr+2*groups[ok]+b).tolist());vv.extend(white[ok,a,b].tolist())
    design=coo_matrix((vv,(rr,cc)),shape=(2*len(frame),2*(nr+nf))).tocsr()
    rhs=np.einsum('nij,nj->ni',white,y).ravel()
    def solve(weight):
        root=np.repeat(np.sqrt(weight),2);m=design.multiply(root[:,None]).tocsr()
        scale=1/np.maximum(np.sqrt(np.asarray(m.power(2).sum(axis=0)).ravel()),1.e-30)
        answer=lsmr(m.multiply(scale),rhs*root,atol=1.e-10,btol=1.e-10,maxiter=4000)
        if answer[1] not in (0,1,2,4,5):raise RuntimeError('Bounded reference did not converge')
        return answer[0]*scale
    weight=np.ones(len(frame))
    for _ in range(6):
        state=solve(weight);residual=(rhs-design@state).reshape(-1,2)
        weight=np.minimum(1,6/(4+np.sum(residual**2,axis=1)))
    state=solve(weight)
    random=np.random.default_rng(210621).exponential(size=(draws,nr))
    samples=np.asarray([solve(weight*r[refs])[2*nr:].reshape(nf,2) for r in random])
    return state[2*nr:].reshape(nf,2),samples,dict(references=nr,measurements=len(frame),free_groups=nf,
        fixed_boundary_groups=frame.loc[groups<0,'group'].nunique(),minimum_robust_weight=float(weight.min()))


def one(row,dest):
    dest.mkdir(parents=True,exist_ok=True);start=monotonic()
    shard=(Path(row['graph']).parent/(row['core']+'_reference_receipt.json')).resolve().parent/'reference.csv.gz'
    frame=pd.read_csv(shard,dtype={'reference':str})
    offsets=read_offsets(Path(row['graph']));frame=frame.loc[frame.group.isin(offsets)].copy()
    train=frame.loc[~frame.is_holdout].copy()
    counts=train.groupby('group').reference.nunique().sort_values(ascending=False,kind='stable')
    # Metadata-only selection. Other NSC groups are the unchanged external gauge.
    sdss=[g for g,n in counts.items() if g.startswith('SDSS:') and n>=30][:2]
    nsc=[g for g,n in counts.items() if g.startswith('NSC:') and n>=30]
    free=sdss+nsc[:min(4,max(0,len(nsc)-1))]
    eligible=train.groupby('reference').group.agg(lambda s:bool(set(s)&set(free)) and bool(set(s)-set(free)))
    selected=sorted(eligible.index[eligible],key=lambda k:digest_text(k))[:1200]
    train=train.loc[train.reference.isin(selected)].reset_index(drop=True)
    if not free or len(selected)<30:
        return dict(core=row['core'],status='insufficient_boundary_support',references=len(selected),free_groups=len(free))
    for f in (frame,train):
        f[['corrected_e','corrected_n']]=f[['east','north']].to_numpy()-np.asarray([offsets[g][0] for g in f.group])
    sigma=train[['sigma_e','sigma_n']].to_numpy();native=native_covariance(sigma)
    covariances=dict(old_diagonal=native,full2d=projected_covariance(train.ra_deg,train.dec_deg,row['graph_center'],native))
    solutions={};private={};summary={}
    for name,cov in covariances.items():
        p,draw,meta=bounded_fit(train,cov,free);solutions[name]=dict(zip(free,p));private[name]=dict(point=p,draws=draw)
        held=frame.loc[frame.is_holdout].copy();held[['corrected_e','corrected_n']]-=np.asarray([solutions[name].get(g,np.zeros(2)) for g in held.group])
        residual=[]
        for _,g in held.groupby('reference',sort=False):
            moving=g.group.isin(free).to_numpy()
            if not moving.any() or moving.all():continue
            delta=np.median(g.loc[moving,['corrected_e','corrected_n']],axis=0)-np.median(g.loc[~moving,['corrected_e','corrected_n']],axis=0)
            a=forward_jacobian(float(g.ra_deg.median()),float(g.dec_deg.median()),row['graph_center'])
            residual.append(np.linalg.solve(a,delta))
        native_delta=[];native_se=[]
        for j,g in enumerate(free):
            part=train.loc[train.group.eq(g)];a=forward_jacobian(float(part.ra_deg.median()),float(part.dec_deg.median()),row['graph_center'])
            native_delta.append(np.linalg.solve(a,p[j]));native_se.append(np.sqrt(np.diag(np.linalg.inv(a)@np.cov(draw[:,j,:],rowvar=False)@np.linalg.inv(a).T)))
        summary[name]=dict(**meta,holdout_boundary_position_mas=summarize(residual),
            group_delta_native_mas_p16_p50_p84=np.quantile(native_delta,[.16,.5,.84],axis=0),
            conditional_group_error_native_mas_p16_p50_p84=np.quantile(native_se,[.16,.5,.84],axis=0))
    np.savez_compressed(dest/'private_bounded_offsets.npz',names=np.asarray(free),**{f'{k}_{n}':v for k,r in private.items() for n,v in r.items()})
    report=dict(core=row['core'],reference_shard_sha256=digest(shard),graph_sha256=digest(Path(row['graph'])),
        models=summary,elapsed_seconds=monotonic()-start,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Same fixed train/holdout identities, same free groups and fixed old C0 external boundary. Six shared IRLS updates plus final solve; 48 paired reference-object exponential draws. Every reference ties to a fixed group. Errors conditional on boundary; never added as independent global reference errors.')
    check_aggregate(report);save_json(dest/'v21_bounded_reference_core_summary.json',report)
    print('BOUNDED_REFERENCE_DONE',row['core'],report['elapsed_seconds'],flush=True);return report


def digest_text(value):
    from hashlib import sha256
    return sha256(str(value).encode()).hexdigest()


def run(registration,output):
    p=read(registration);results=[one(r,output/r['core']) for r in p['selected'] if r['core'] in p['pilot']]
    save_json(output/'v21_bounded_reference_summary.json',dict(cores=results,slurm_job=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('registration',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
