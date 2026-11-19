"""Independent fixed-target validation of the one-scalar local DCR experiment."""
import argparse
from collections import Counter,defaultdict
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .allband_qso import evaluated_fit,physical_candidates
from .astrometry import project,unproject
from .condition_holdout import held_prediction_variance
from .continuous_reference import PRECISION,SPATIAL,bind,reference_frame,verify_directory
from .dark_precision import query_unique
from .dcr_geometry import tangent_geometry,combine_color
from .fixed_year_multiband import summarize
from .io import save_json,digest
from .nsc_comparison import comparison,qso_matches
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate
from .validation import ratio
from .year_opportunities import utc_year

MODELS=('D0_local_C0','D1_geometric_DCR')


def read_model(path):
    with np.load(path,allow_pickle=False) as a:
        return dict(base={str(k):(p,s,int(n)) for k,p,s,n in zip(a['names'],a['position'],a['draws'],a['counts'])},
            geometry={str(k):(bool(role),z) for k,role,z in zip(a['names'],a['dcr_role'],a['zenith_vectors'])},
            beta=float(a['dcr_coefficient']),beta_draw=a['dcr_coefficient_draws'],center=tuple(a['projection_center_deg']))


def offsets(model,keys,sky,color,color_error,identity):
    known=np.isfinite(color+color_error) and color_error>0
    value=color if known else 1.2
    seed=int.from_bytes(sha256(identity.encode()).digest()[:4],'little')
    noise=np.random.default_rng(seed).normal(size=len(model['beta_draw']))*(color_error if known else 0.)
    result={}
    for key in sorted(set(keys)):
        if key not in model['base']:continue
        point,samples,n=model['base'][key];role,z=model['geometry'][key];h=np.zeros(2)
        if role:h=tangent_geometry([sky[0]],[sky[1]],z[None,:],model['center'])[0][0]
        shift=model['beta']*(value-1.2)*h
        # One color perturbation for the entire source, shared between all
        # training/held observations. Group and coefficient draws stay coupled.
        draw=samples+model['beta_draw'][:,None]*(value-1.2)*h+(model['beta']+model['beta_draw'])[:,None]*noise[:,None]*h
        result[key]=(point+shift,draw-draw.mean(axis=0),n)
    return result,known


def context(trial,model_root,kind):
    source=(trial/'source_receipt.json').resolve().parent
    models={name:read_model(model_root/('private_'+name+'.npz')) for name in MODELS};center=models[MODELS[0]]['center']
    if models[MODELS[1]]['center']!=center:raise ValueError('D0/D1 projection mismatch')
    graph=(trial/('new_reference_receipt.json' if kind=='qso' else 'reference_receipt.json')).resolve().parent/'private_c0.npz'
    if tuple(reference_frame(graph))!=center:raise ValueError('Frozen target coordinate basis changed')
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());settings.update(bootstrap_draws=800,bootstrap_absolute_zero=True,min_publish_count=5)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data);lookup=pd.Index(data.detection_id)
    phot=pd.read_csv(model_root/'private_color_photometry.csv').set_index('detection_id')
    def source_solutions(index,all_keys):
        early=index[data.survey.iloc[index].eq('SDSS').to_numpy()]
        colors=phot.reindex(data.detection_id.iloc[early]);color,error=combine_color(colors)
        sky=unproject(np.median(xy[index],axis=0),center)[0];identity='|'.join(sorted(data.detection_id.iloc[index].astype(str)))
        return {name:offsets(model,all_keys,sky,color,error,identity)[0] for name,model in models.items()},bool(np.isfinite(color+error)),(color,error)
    return source,center,config,settings,data,xy,time,sigma,keep,keys,lookup,source_solutions


def run_qso(work,trial,model_root):
    source,center,config,settings,data,xy,time,sigma,keep,keys,lookup,source_solutions=context(trial,model_root,'qso')
    with gzip.open(trial/'private_fixed_qso.json.gz','rt') as f:original=json.load(f)
    records=[];exits=Counter()
    for row in original:
        if not {'quality_r','all_nsc_w0'}<=row['members'].keys():continue
        r=lookup.get_indexer(row['members']['quality_r']);all_index=lookup.get_indexer(row['members']['all_nsc_w0'])
        if (np.r_[r,all_index]<0).any():raise ValueError('Frozen target detections missing')
        solutions,known,color=source_solutions(all_index,keys[all_index])
        if any(any(k not in sol for k in keys[all_index]) for sol in solutions.values()):exits['local_reference_support_loss']+=1;continue
        values=dict(C0_r=row['values']['quality_r'],C0_all=row['values']['all_nsc_w0'])
        for name,sol in solutions.items():
            for suffix,index in [('r',r),('all',all_index)]:
                fitted=evaluated_fit(index,data,xy,time,sigma,keys,sol,center,config['reference_epoch']);values[name+'_'+suffix]=(fitted['pm_icrs_masyr'],fitted['pm_error_masyr'])
        for name in ('NSC','Xiao'):
            if name in row['values']:values[name]=row['values'][name]
        records.append(dict(magnitude=row['magnitude'],known_color_error=known,color=color,values=values))
    contrasts=(('D1_geometric_DCR_all','D0_local_C0_all'),('D1_geometric_DCR_all','C0_all'),('D1_geometric_DCR_all','C0_r'))
    cohorts={}
    for low in (20,21,22):
        part=[r for r in records if low<=r['magnitude']<low+1]
        methods=('C0_r','C0_all','D0_local_C0_r','D0_local_C0_all','D1_geometric_DCR_r','D1_geometric_DCR_all')
        cohorts['r'+str(low)]=dict(fixed=comparison([{k:r['values'][k] for k in methods} for r in part],settings,contrasts),unknown_color_error_count=sum(not r['known_color_error'] for r in part),
            strict_published_common=comparison([r['values'] for r in part if {'NSC','Xiao'}<=r['values'].keys()],settings,contrasts+(('D1_geometric_DCR_all','Xiao'),('D1_geometric_DCR_all','NSC'))))
    return dict(cohorts=cohorts,exits=dict(exits),input_qso_count=len(original)),records


def run_year(work,trial,model_root,blind_zero=False):
    info=json.loads((trial/'allband_blind_year_summary.json').read_text())
    if info['status']!='completed':return dict(status=info['status']),[]
    source,center,config,settings,data,xy,time,sigma,keep,keys,lookup,source_solutions=context(trial,model_root,'year')
    training=json.loads((trial/'private_training.json').read_text());held=json.loads((trial/'private_held.json').read_text());by_source=defaultdict(list)
    for row in held:by_source[row['slot']].append(row)
    nsc=data.survey.eq('NSC').to_numpy();year=utc_year(data.observation_time.to_numpy());records=[];exits=Counter()
    if blind_zero:
        qso=pd.read_csv(source/'spectra_union.csv');points=project(qso.ra,qso.dec,center)
        found,nearest=query_unique(cKDTree(points),np.array([r['position'] for r in training]),500,300) if len(qso) and training else (np.zeros(len(training),bool),np.zeros(len(training),int))
        claims=Counter(nearest[found]);slots=[j for j in np.flatnonzero(found) if claims[nearest[j]]==1]
        labels,_=qso_matches(data,xy,~(nsc&(year==info['year']))&(keep|(nsc&physical_candidates(data))),points)
    else:slots=sorted(by_source)
    for slot in slots:
        row=training[slot];index=lookup.get_indexer(row['detection_ids']);tests=by_source.get(slot,[])
        if (index<0).any() or (nsc[index]&(year[index]==info['year'])).any():raise ValueError('Missing frozen training or held-year leakage')
        test_index=lookup.get_indexer([r['held_detection_id'] for r in tests]) if not blind_zero else np.empty(0,int)
        if (test_index<0).any():raise ValueError('Frozen held observation missing')
        all_keys=keys[np.r_[index,test_index]];solutions,known,color=source_solutions(index,all_keys)
        if any(any(k not in sol for k in all_keys) for sol in solutions.values()):exits['local_reference_support_loss']+=1;continue
        if blind_zero:
            values=dict(C0_all=(row['pm_icrs_masyr'],row['pm_error_masyr']))
            for name,sol in solutions.items():
                fit=evaluated_fit(index,data,xy,time,sigma,keys,sol,center,config['reference_epoch']);values[name]=(fit['pm_icrs_masyr'],fit['pm_error_masyr'])
            wrong=int(((labels[index]>=0)&(labels[index]!=nearest[slot])).sum());unknown=int((labels[index]<0).sum())
            records.append(dict(kind=row['kind'],magnitude=row['magnitude'],known_color_error=known,values=values,members=len(index),wrong=wrong,unknown=unknown))
        else:
            fits={name:fit_with_calibration(xy[index],time[index],sigma[index],config['reference_epoch'],[sol[k] for k in keys[index]])[0] for name,sol in solutions.items()}
            for k,saved in zip(test_index,tests):
                residual=[saved['residual_mas']];error=[saved['error_mas']]
                for name,sol in solutions.items():
                    fit=fits[name];residual.append(xy[k]-sol[keys[k]][0]-fit.predict(time[k]));error.append(np.sqrt(held_prediction_variance(fit,time[index],sigma[index],[sol[g] for g in keys[index]],sol[keys[k]],sigma[k],time[k])))
                records.append(dict(slot=slot,kind=row['kind'],magnitude=row['magnitude'],known_color_error=known,residual=np.array(residual),error=np.array(error)))
    cohorts={}
    for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
        for low in (21,22):
            selected=[r for r in records if r['kind']==kind and low<=r['magnitude']<low+1]
            if blind_zero:
                cohorts[kind+'__r'+str(low)]=dict(all_spectral=comparison([r['values'] for r in selected],settings,((MODELS[1],MODELS[0]),(MODELS[1],'C0_all'))),
                    consistent_spectral=comparison([r['values'] for r in selected if r['wrong']==0 and r['unknown']==0],settings,((MODELS[1],MODELS[0]),)),
                    unknown_color_error_sources=sum(not r['known_color_error'] for r in selected),known_competing_members=ratio(sum(r['wrong'] for r in selected),sum(r['members'] for r in selected)),unknown_members=ratio(sum(r['unknown'] for r in selected),sum(r['members'] for r in selected)))
            else:
                names=('C0_all',)+MODELS;cohorts[kind+'__r'+str(low)]={}
                for a,b in ((1,0),(2,1),(2,0)):
                    pair=[dict(r,residual=r['residual'][[b,a]],error=r['error'][[b,a]]) for r in selected]
                    cohorts[kind+'__r'+str(low)][names[a]+'_vs_'+names[b]]=summarize(pair,names=(names[b],names[a]))
    return dict(status='completed',year=info['year'],frozen_source_count=len(training),cohorts=cohorts,exits=dict(exits)),records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('qso','year','blind-zero'))
    for n in ('work','trial','model'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);files=('private_fixed_qso.json.gz',) if a.mode=='qso' else ('private_training.json','private_held.json','allband_blind_year_summary.json')
    bind(a.work,'target_receipt.json',verify_directory(a.trial,files));bind(a.work,'model_receipt.json',verify_directory(a.model,('private_D0_local_C0.npz','private_D1_geometric_DCR.npz','private_color_photometry.csv')))
    with run_stage(a.work,PRECISION,a.mode,'dcr-validation-v15',inputs=('target_receipt.json','model_receipt.json'),outputs=('dcr_validation_summary.json','private_dcr_validation.json')) as go:
        if go:
            summary,records=run_qso(a.work,a.trial,a.model) if a.mode=='qso' else run_year(a.work,a.trial,a.model,blind_zero=a.mode=='blind-zero')
            summary.update(slurm_job=os.environ['SLURM_JOB_ID'],scope='Frozen source/member/held identities, original global C0 and identical-local-graph D0/D1 controls. No QSO coefficient training or held-year target fitting. Synchronous joint group/beta draws plus one measured color perturbation per source shared across training and held. Unknown color/SED/systematic errors remain unqualified; no adopted model.',model_sha256={name:digest(a.model/('private_'+name+'.npz')) for name in MODELS})
            check_aggregate(summary);save_json(a.work/'dcr_validation_summary.json',summary);save_json(a.work/'private_dcr_validation.json',records)


if __name__=='__main__':main()
