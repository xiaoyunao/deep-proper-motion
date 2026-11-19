"""Fixed-target independent validation of the joint one-color reference trial."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from scipy.spatial import cKDTree

from .allband_qso import evaluated_fit
from .astrometry import project
from .dark_precision import query_unique
from .condition_holdout import held_prediction_variance
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .fixed_year_multiband import summarize
from .io import save_json,digest
from .nsc_comparison import comparison,qso_matches
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate
from .year_opportunities import utc_year
from .validation import ratio


def read_model(path):
    with np.load(path,allow_pickle=False) as a:
        labels=a['feature_component_labels'];coef=dict(zip(labels,a['feature_coefficients']));draws=dict(zip(labels,a['feature_draws']))
        base={k:(point,samples,int(n)) for k,point,samples,n in zip(a['names'],a['position'],a['draws'],a['counts'])}
        return dict(base=base,gauges=dict(zip(a['names'],a['gauge'])),roles=dict(zip(a['names'],a['feature_groups'])),
            coefficient=coef,feature_draws=draws,color_reference=float(a['reference_color_g_i']),center=a['projection_center_deg'].tolist())


def source_offsets(model,keys,color):
    color=float(color) if np.isfinite(color) else model['color_reference']
    delta=color-model['color_reference'];result={}
    for key in sorted(set(keys)):
        if key not in model['base']:continue
        point,samples,n=model['base'][key];gauge=model['gauges'][key]
        if model['roles'][key]:
            point=point+model['coefficient'][gauge]*delta
            samples=samples+model['feature_draws'][gauge]*delta
        result[key]=(point,samples,n)
    return result


def target_color(data,index):
    early=index[data.survey.iloc[index].eq('SDSS').to_numpy()]
    value=data.color.iloc[early].to_numpy(float) if 'color' in data else np.array([])
    value=value[np.isfinite(value)&(value>-1)&(value<4)]
    return float(np.median(value)) if len(value) else np.nan


def qso_trial(work,trial,model_path):
    model=read_model(model_path);config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());settings.update(bootstrap_draws=800,bootstrap_absolute_zero=True)
    source=(trial/'source_receipt.json').resolve().parent
    data,xy,time,sigma,keep=quality_detections(source,model['center'],config,settings['production']);keys=calibration_groups(data);lookup=pd.Index(data.detection_id)
    with gzip.open(trial/'private_fixed_qso.json.gz','rt') as stream:original=json.load(stream)
    output=[];exits=Counter()
    for row in original:
        if 'all_nsc_w0' not in row['members'] or 'quality_r' not in row['members']:exits['without_paired_physical_r_control']+=1;continue
        r_index=lookup.get_indexer(row['members']['quality_r']);all_index=lookup.get_indexer(row['members']['all_nsc_w0'])
        if (r_index<0).any() or (all_index<0).any():raise ValueError('Frozen target detections missing')
        color=target_color(data,r_index);solution=source_offsets(model,keys[all_index],color)
        if any(k not in solution for k in keys[all_index]):exits['new_reference_support_loss']+=1;continue
        values=dict(C0_r=row['values']['quality_r'],C0_all=row['values']['all_nsc_w0'])
        fits={}
        for name,index in [('color_r',r_index),('color_all',all_index)]:
            fitted=evaluated_fit(index,data,xy,time,sigma,keys,solution,model['center'],config['reference_epoch']);fits[name]=fitted
            values[name]=(fitted['pm_icrs_masyr'],fitted['pm_error_masyr'])
        for name in ('NSC','Xiao'):
            if name in row['values']:values[name]=row['values'][name]
        output.append(dict(magnitude=row['magnitude'],known_color=bool(np.isfinite(color)),color=color if np.isfinite(color) else None,
            values=values,fits=fits,qso_local_index=row['qso_local_index'],r_detection_ids=row['members']['quality_r'],all_detection_ids=row['members']['all_nsc_w0']))
    methods=('C0_r','C0_all','color_r','color_all');cohorts={};contrasts=(('color_all','C0_all'),('color_r','C0_r'),('color_all','color_r'),('color_all','Xiao'),('color_all','NSC'))
    for low in (20,21,22):
        sample=[r for r in output if low<=r['magnitude']<low+1]
        for label,part in [('all',sample),('known_color',[r for r in sample if r['known_color']])]:
            cohorts[f'r{low}__{label}']=comparison([{k:r['values'][k] for k in methods} for r in part],settings,contrasts)
            common=[r for r in part if {'NSC','Xiao'}<=r['values'].keys()]
            cohorts[f'r{low}__{label}__published_common']=comparison([r['values'] for r in common],settings,contrasts)
    result=dict(input_known_position_qso_count=len(original),evaluated_qso_count=len(output),unknown_color_count=sum(not r['known_color'] for r in output),exits=dict(exits),cohorts=cohorts,
        model_sha256=digest(model_path),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Same frozen objects and exact r/all detection identities, original native W0 weights and four parameters. Reference groups and a single DECam-g color coefficient were trained only on original galaxy training references. QSO never train coefficients. Scores are conditional on fixed source colors; color measurement uncertainty and its effect on coefficient attenuation remain unqualified. No production adoption from this trial.')
    check_aggregate(result);save_json(work/'chromatic_qso_summary.json',result);save_json(work/'private_chromatic_qso.json',output)


def year_trial(work,trial,model_path):
    info=json.loads((trial/'allband_blind_year_summary.json').read_text())
    if info['status']!='completed':
        save_json(work/'chromatic_year_summary.json',dict(status=info['status']));save_json(work/'private_chromatic_held.json',[]);return
    model=read_model(model_path);config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    source=(trial/'source_receipt.json').resolve().parent;graph=(trial/'reference_receipt.json').resolve().parent/'private_c0.npz'
    if tuple(reference_frame(graph))!=tuple(model['center']):raise ValueError('Held-year trial requires the same reference coordinate basis')
    data,xy,time,sigma,keep=quality_detections(source,model['center'],config,settings['production']);keys=calibration_groups(data);lookup=pd.Index(data.detection_id)
    year=utc_year(data.observation_time.to_numpy());nsc=data.survey.eq('NSC').to_numpy()
    training=json.loads((trial/'private_training.json').read_text());held=json.loads((trial/'private_held.json').read_text());held_by_source={}
    for row in held:held_by_source.setdefault(row['slot'],[]).append(row)
    records=[];exits=Counter()
    for slot,tests in held_by_source.items():
        row=training[slot];index=lookup.get_indexer(row['detection_ids'])
        if (index<0).any() or (nsc[index]&(year[index]==info['year'])).any():raise ValueError('Frozen training members missing or held-year leakage')
        test_index=lookup.get_indexer([r['held_detection_id'] for r in tests])
        if (test_index<0).any():raise ValueError('Held observations missing')
        color=target_color(data,index);solution=source_offsets(model,keys[np.r_[index,test_index]],color)
        if any(k not in solution for k in keys[index]):exits['lost_training_support']+=1;continue
        fitted,_=fit_with_calibration(xy[index],time[index],sigma[index],config['reference_epoch'],[solution[k] for k in keys[index]])
        for k,saved in zip(test_index,tests):
            if keys[k] not in solution:exits['lost_held_support']+=1;continue
            residual=xy[k]-solution[keys[k]][0]-fitted.predict(time[k])
            variance=held_prediction_variance(fitted,time[index],sigma[index],[solution[g] for g in keys[index]],solution[keys[k]],sigma[k],time[k])
            records.append(dict(slot=slot,kind=row['kind'],magnitude=row['magnitude'],known_color=bool(np.isfinite(color)),band=saved['band'],
                residual=np.array([saved['residual_mas'],residual]),error=np.array([saved['error_mas'],np.sqrt(variance)])))
    cohorts={}
    for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
        for low in (20,21,22):
            selected=[r for r in records if r['kind']==kind and low<=r['magnitude']<low+1]
            cohorts[kind+'__r'+str(low)]=summarize(selected,names=('C0_all','color_all'))
    result=dict(status='completed',year=info['year'],cohorts=cohorts,exits=dict(exits),model_sha256=digest(model_path),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Exactly frozen training target identities and same C0-gated held physical detections. New reference corrections applied coherently to training and held measurements with synchronous group/color-bootstrap covariance. No held target fitted or used to tune coefficients. Predictive comparison only, not rerun end-to-end blind association. Conditional fixed-color uncertainty; no production adoption.')
    check_aggregate(result);save_json(work/'chromatic_year_summary.json',result);save_json(work/'private_chromatic_held.json',records)


def blind_zero_trial(work,trial,model_path):
    """Evaluate frozen blind members; stationary labels enter only after fitting."""
    info=json.loads((trial/'allband_blind_year_summary.json').read_text())
    if info['status']!='completed':
        save_json(work/'chromatic_blind_zero_summary.json',dict(status=info['status']));save_json(work/'private_chromatic_blind_zero.json',[]);return
    model=read_model(model_path);config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());settings.update(bootstrap_draws=800,bootstrap_absolute_zero=True)
    source=(trial/'source_receipt.json').resolve().parent;graph=(trial/'reference_receipt.json').resolve().parent/'private_c0.npz'
    if tuple(reference_frame(graph))!=tuple(model['center']):raise ValueError('Blind trial requires the same reference coordinate basis')
    data,xy,time,sigma,keep=quality_detections(source,model['center'],config,settings['production']);keys=calibration_groups(data);lookup=pd.Index(data.detection_id)
    training=json.loads((trial/'private_training.json').read_text());year=utc_year(data.observation_time.to_numpy());nsc=data.survey.eq('NSC').to_numpy()
    # The identity universe is the original blind catalogue, including objects
    # without held matches. New calibration cannot choose more favorable labels.
    qso=pd.read_csv(source/'spectra_union.csv');qso=qso.loc[np.isfinite(qso[['ra','dec']]).all(axis=1)].reset_index(drop=True)
    positions=np.asarray([r['position'] for r in training]);found,nearest=query_unique(cKDTree(project(qso.ra,qso.dec,model['center'])),positions,500,300) if len(qso) and len(training) else (np.zeros(len(training),bool),np.zeros(len(training),int))
    claims=Counter(nearest[found]);output=[];exits=Counter()
    mask=~(nsc&(year==info['year']))
    from .allband_qso import physical_candidates
    labels,_=qso_matches(data,xy,mask&(keep|(nsc&physical_candidates(data))),project(qso.ra,qso.dec,model['center']))
    for slot in np.flatnonzero(found):
        if claims[nearest[slot]]!=1:exits['nonunique_stationary_identity']+=1;continue
        row=training[slot];index=lookup.get_indexer(row['detection_ids'])
        if (index<0).any() or (nsc[index]&(year[index]==info['year'])).any():raise ValueError('Frozen blind members missing or held-year leakage')
        color=target_color(data,index);solution=source_offsets(model,keys[index],color)
        if any(k not in solution for k in keys[index]):exits['lost_reference_support']+=1;continue
        fit=evaluated_fit(index,data,xy,time,sigma,keys,solution,model['center'],config['reference_epoch'])
        known=labels[index];wrong=int(((known>=0)&(known!=nearest[slot])).sum());unknown=int((known<0).sum())
        output.append(dict(kind=row['kind'],magnitude=row['magnitude'],known_color=bool(np.isfinite(color)),members=len(index),known_competing_members=wrong,unknown_members=unknown,
            values=dict(C0_all=(row['pm_icrs_masyr'],row['pm_error_masyr']),color_all=(fit['pm_icrs_masyr'],fit['pm_error_masyr']))))
    cohorts={}
    for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
        for low in (21,22):
            part=[r for r in output if r['kind']==kind and low<=r['magnitude']<low+1];members=sum(r['members'] for r in part)
            consistent=[r for r in part if r['known_competing_members']==0 and r['unknown_members']==0]
            cohorts[kind+'__r'+str(low)]=dict(spectral_sources=len(part),all_members_consistent=ratio(len(consistent),len(part)),
                known_competing_member_fraction=ratio(sum(r['known_competing_members'] for r in part),members),unknown_member_fraction=ratio(sum(r['unknown_members'] for r in part),members),
                all_spectral=comparison([r['values'] for r in part],settings,(('color_all','C0_all'),)),
                consistent_spectral=comparison([r['values'] for r in consistent],settings,(('color_all','C0_all'),)))
    result=dict(status='completed',year=info['year'],frozen_blind_source_count=len(training),cohorts=cohorts,exits=dict(exits),model_sha256=digest(model_path),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Exact frozen blind source universe and member identities, including sources without held matches. Spectral positions label identities and individual members only; no PM similarity or QSO coefficient fitting. Same W0 native errors. C0 versus joint color model conditional on fixed source colors; object intervals do not certify common spatial zero or blend truth. Candidate classes are not adopted counts.')
    check_aggregate(result);save_json(work/'chromatic_blind_zero_summary.json',result);save_json(work/'private_chromatic_blind_zero.json',output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('qso','year','blind-zero'))
    for n in ('work','trial','model'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);prefix={'qso':'chromatic_qso','year':'chromatic_year','blind-zero':'chromatic_blind_zero'}[a.mode];name=prefix+'_summary.json'
    inputs=('private_fixed_qso.json.gz',) if a.mode=='qso' else ('private_training.json','private_held.json','allband_blind_year_summary.json')
    bind(a.work,'target_trial_receipt.json',verify_directory(a.trial,inputs));bind(a.work,'model_receipt.json',verify_directory(a.model,('private_chromatic_reference.npz','chromatic_reference_summary.json')))
    with run_stage(a.work,PRECISION,a.mode,'chromatic-validation-v14',inputs=('target_trial_receipt.json','model_receipt.json'),outputs=(name,{'qso':'private_chromatic_qso.json','year':'private_chromatic_held.json','blind-zero':'private_chromatic_blind_zero.json'}[a.mode])) as go:
        if go:{'qso':qso_trial,'year':year_trial,'blind-zero':blind_zero_trial}[a.mode](a.work,a.trial,a.model/'private_chromatic_reference.npz')


if __name__=='__main__':main()
