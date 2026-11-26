"""Training-before-association, spherical held-year tests for changed sources."""
import argparse
from collections import Counter,defaultdict
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import access
from .__main__ import tomllib
from .allband_qso import physical_candidates,evaluated_fit
from .astrometry import basis,project,unproject
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets
from .coordinate_candidate import metadata_chips,opportunities
from .coordinate_metric import (AngularTree,forward_jacobian,native_covariance,projected_covariance,
    fit_vector_motion,reference_covariance,design_matrix)
from .exposure_catalogues import inside
from .full_coordinate_sources import write_rows
from .full_reference_run import local_support
from .independent_year import choose_year,exclusive
from .io import digest,save_json
from .nsc_science_catalogue import select_catalogue
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,fit_with_calibration,calibration_support
from .public_summary import check_aggregate
from .standard_motion import icrs_motion,calibrated_covariance
from .usability_cluster import summarize
from .validation import ratio


def paired_entities(old,new):
    """Exact or bidirectionally unique >=60percent physical-member continuity."""
    members=[defaultdict(list),defaultdict(list)]
    for side,rows in enumerate((old,new)):
        for i,r in enumerate(rows):
            for d in r['members']:members[side][d].append(i)
    choice=[]
    for side,rows in enumerate((old,new)):
        other=(new,old)[side];best={}
        for i,r in enumerate(rows):
            votes=Counter(j for d in r['members'] for j in members[1-side].get(d,[]))
            good={j:n for j,n in votes.items() if n>=.6*len(r['members']) and n>=.6*len(other[j]['members'])}
            if good:
                most=max(good.values());winner=[j for j,n in good.items() if n==most]
                if len(winner)==1:best[i]=winner[0]
        choice.append(best)
    pairs=[(i,j) for i,j in choice[0].items() if choice[1].get(j)==i];a={i for i,j in pairs};b={j for i,j in pairs}
    return [dict(old=i,new=j,kind='common_training_identity') for i,j in pairs]+[dict(old=i,new=None,kind='old_only_training') for i in range(len(old)) if i not in a]+[dict(old=None,new=j,kind='new_only_training') for j in range(len(new)) if j not in b]


def training(data,xy,t,sigma,keep,keys,solution,center,spec,config,settings,full2d):
    cp=projected_covariance(data.ra_deg,data.dec_deg,center,native_covariance(sigma));inverse=np.linalg.inv(cp)
    names=np.array(sorted(solution));draws=np.stack([solution[k][1] for k in names])
    support=local_support(data.ra_deg,data.dec_deg,center,keys,names,draws)[0] if full2d else calibration_support(keys,solution)
    @lru_cache(maxsize=32768)
    def solve(members):
        ix=np.asarray(members,int);offset=[solution[g] for g in keys[ix]]
        if full2d:
            model=fit_vector_motion(xy[ix]-np.array([v[0] for v in offset]),t[ix],cp[ix],config['reference_epoch'])
            common,response=reference_covariance(model.influence,np.stack([v[1] for v in offset],axis=1));independent=model.cov;h=model.influence
            out=icrs_motion(model.coeff[0],model.coeff[1],center,independent+common)
        else:
            model,_=fit_with_calibration(xy[ix],t[ix],sigma[ix],config['reference_epoch'],offset)
            out=evaluated_fit(ix,data,xy,t,sigma,keys,solution,center,config['reference_epoch'],model=model)
            _,independent,_,h=calibrated_covariance(model,t[ix],sigma[ix],offset)
            response=-np.einsum('pni,dni->dp',h,np.stack([v[1] for v in offset],axis=1))
        return model,out,independent,response
    def fit(ix):m,o,_,_=solve(tuple(ix));return m,o['pm_error_masyr']
    def final(ix):return solve(tuple(ix))[1]
    distance=lambda ix,m:np.sqrt(np.einsum('ni,nij,nj->n',m.residual,inverse[ix],m.residual))
    _,candidates,funnel,_=select_catalogue(data,xy,t,sigma,keep,solution,keys,center,spec,config,settings,fit,final,
        sky_center=center if full2d else None,quality_distance=distance if full2d else None,calibration_keep=support if full2d else None)
    results=[]
    for row in candidates:
        ix=row['index'];model,out,cov,response=solve(tuple(ix));results.append(dict(members=data.detection_id.iloc[ix].astype(str).tolist(),
            coeff=model.coeff,epoch=model.epoch,independent=cov,response=response,index=ix))
    return results,funnel


def prediction_covariance(row,held_time,held_covariance,held_reference_draws):
    d=design_matrix(np.array([held_time]),row['epoch'])[0]
    common=np.asarray(held_reference_draws)+row['response']@d.T
    return held_covariance+d@row['independent']@d.T+np.cov(common,rowvar=False)


def one(root,entry,component,output):
    output.mkdir(parents=True,exist_ok=True);source=Path(entry['source']);config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());center=entry['center']
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
    new=read_offsets(root/'reference'/f'component-{component:04d}'/'private_c0_unfiltered.npz');old=read_offsets(entry['old_graph'])
    names=np.array(sorted(new));draws=np.stack([new[g][1] for g in names]);support,se=local_support(data.ra_deg,data.dec_deg,center,keys,names,draws)
    physical=physical_candidates(data);years,held_year=choose_year(time,data.exposure_id.to_numpy(),nsc,nsc&physical&support)
    if held_year is None:
        result=dict(core=entry['core'],status='insufficient_independent_NSC_years',slurm_job=os.environ['SLURM_JOB_ID']);save_json(output/'v22_year_core_summary.json',result);return result
    # Remove EVERY held-year target band physically before either association.
    held=nsc&(years==held_year);train=~held;d=data.loc[train].reset_index(drop=True)
    old_rows,old_funnel=training(d,xy[train],time[train],sigma[train],keep[train],keys[train],old,center,entry['spec'],config,settings,False)
    new_rows,new_funnel=training(d,xy[train],time[train],sigma[train],keep[train],keys[train],new,center,entry['spec'],config,settings,True)
    if any(data.detection_id[held].isin(r['members']).any() for r in old_rows+new_rows):raise ValueError('Held target used during association')
    entities=paired_entities(old_rows,new_rows)
    write_rows(output/'private_training_before_test.json.gz',dict(held_year=held_year,old=old_rows,new=new_rows,entities=entities))
    # Complete universe competes, including lost old hypotheses and new-only
    # hypotheses. Common physical training identities contribute one prediction.
    modes={'old':(old_rows,new_rows,old),'new':(new_rows,old_rows,new)};details={};private=[]
    for method,(primary,fallback,offsets) in modes.items():
        models=[]
        for e in entities:
            j=e[method];other=e['new' if method=='old' else 'old']
            models.append(primary[j] if j is not None else fallback[other])
        found=set();null=[set() for _ in range(4)];residuals=defaultdict(list);errors=defaultdict(list);collisions=0
        for exposure,f in data.loc[held&physical&support].groupby('exposure_id',sort=False):
            ix=f.index.to_numpy()
            if any(g not in offsets for g in keys[ix]):
                ix=ix[pd.Series(keys[ix]).isin(offsets).to_numpy()]
            if not len(ix):continue
            mid=float(np.median(time[ix]));coord=xy[ix]-np.array([offsets[g][0] for g in keys[ix]])
            pred=np.array([r['coeff'][0]+r['coeff'][1]*(mid-r['epoch']) for r in models]).reshape(-1,2)
            tree=AngularTree(coord,center);good,near,n=exclusive(tree,pred,1000,settings['production']['competitor_margin_mas']);collisions+=n
            for shift,store in zip(((30000,10000),(-30000,10000),(10000,-30000),(-10000,-30000)),null):
                # Displacement defined in the per-prediction local physical
                # tangent basis, then forward into the shared chart.
                sky=unproject(pred,center);delta=np.einsum('nij,j->ni',forward_jacobian(*sky.T,center),shift)
                accepted,_,_=exclusive(tree,pred+delta,1000,settings['production']['competitor_margin_mas']);store.update(np.flatnonzero(accepted))
            for j in np.flatnonzero(good):
                found.add(int(j));k=ix[near[j]];r=models[j];prediction=r['coeff'][0]+r['coeff'][1]*(time[k]-r['epoch']);difference=xy[k]-offsets[keys[k]][0]-prediction
                # Fallback hypotheses only compete; their metrics are not mixed
                # into the primary method's valid training output.
                if entities[j][method] is None:continue
                rawcov=native_covariance(sigma[k:k+1]);cp=projected_covariance(data.ra_deg.iloc[k],data.dec_deg.iloc[k],center,rawcov)[0]
                cov=prediction_covariance(r,time[k],cp,offsets[keys[k]][1]);b=np.linalg.inv(forward_jacobian(data.ra_deg.iloc[k],data.dec_deg.iloc[k],center))
                difference=b@difference;cov=b@cov@b.T;kind=entities[j]['kind'];residuals[kind].append(difference);errors[kind].append(np.sqrt(np.diag(cov)))
                private.append(dict(method=method,entity=int(j),held_detection=str(data.detection_id.iloc[k]),native_residual_mas=difference,native_covariance=cov))
        by_class={}
        for kind in ('common_training_identity','old_only_training','new_only_training'):
            chosen={j for j,e in enumerate(entities) if e['kind']==kind and e[method] is not None}
            by_class[kind]=dict(training_sources=len(chosen),unique_reproductions=ratio(len(chosen&found),len(chosen)),
                displaced_sources=[len(chosen&s) for s in null],position=summarize(residuals[kind],errors[kind]) if residuals[kind] else dict(n=0))
        details[method]=dict(by_class=by_class,competing_claims_rejected=collisions,scope='Same full old/new training-identity universe; fallback hypotheses compete but do not enter primary metrics. Unique 1arcsec, margin300mas; residual/coverage are conditional on this gate.')
    write_rows(output/'private_held_test_matches.json.gz',private)
    # Actual all-band CCD polygons, reference presence and native SE are handled
    # separately from target detection existence. Native support is checked at
    # every predicted physical exposure position before opportunity counting.
    chips,chipmeta=metadata_chips(source,access.Cache(output/'chip_cache'));chips.to_csv(output/'private_opportunity_chips.csv.gz',index=False)
    models=[];members=[];lookup=pd.Index(data.detection_id)
    from types import SimpleNamespace
    for r in new_rows:models.append(SimpleNamespace(coeff=r['coeff'],epoch=r['epoch']));members.append(lookup.get_indexer(r['members']))
    def accept(key,prediction,center):
        sky=unproject(prediction,center)
        return local_support(*sky.T,center,np.repeat(key,len(sky)),np.array([key]),np.asarray(new[key][1])[None])[0]
    opportunity,opmeta=opportunities(data,models,members,chips,center,new,held_year,calibration_accept=accept)
    # Archived helper only checks group presence; refine the denominator by
    # native per-prediction support, never silently call it certified imaging.
    result=dict(core=entry['core'],status='completed',held_year=held_year,old_training=len(old_rows),new_training=len(new_rows),
        entities=dict(Counter(e['kind'] for e in entities)),old_funnel=old_funnel,new_funnel=new_funnel,methods=details,
        opportunity=dict(new_sources=len(new_rows),geometry=int(opportunity['geometric'].sum()),catalogue_10sigma=int(opportunity['catalogue_10sigma'].sum()),metadata=dict(**chipmeta,**opmeta),
            by_class={kind:{tag:dict(candidates=sum(bool(opportunity[tag][e['new']]) for e in entities if e['new'] is not None and e['kind']==kind),
                uniquely_reproduced=sum(bool(opportunity[tag][e['new']]) and j in found for j,e in enumerate(entities) if e['new'] is not None and e['kind']==kind))
                for tag in ('geometric','catalogue_10sigma','background_limited_5sigma_proxy','depth_or_band_unknown')} for kind in ('common_training_identity','new_only_training')},
            scope='Actual CCD footprint, exposure epoch and train-band magnitude/depth. Native12mas group support checked at each predicted published sky position. Internal bad pixels and true target extraction completeness not certified.'),
        training_before_all_held_target_association=True,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Full new-reference conditional prediction; reference galaxies can calibrate test exposures, held target bands never enter training. Source continuity is physical training-member evidence, not stellar truth; footprint/depth is not image extraction completeness. Old and new reference draw streams are kept separate.')
    check_aggregate(result);save_json(output/'v22_year_core_summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--core',required=True);p.add_argument('--component',type=int,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    entry=next(e for e in read(a.root/'registration/private_full_coordinate_plan.json')['cores'] if e['core']==a.core)
    one(a.root,entry,a.component,a.root/'year'/a.core)
