"""Fixed-member NSC new-night versus same-night information ablations.

SDSS positions remain r only, with every selected physical run unchanged.
B includes all new-night measurements; B1 separately limits each new night to
one deterministic exposure so B's own repeated centroids are not called time.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from .__main__ import tomllib
from .allband_qso import evaluated_fit,physical_candidates
from .astrometry import seasons
from .condition_holdout import held_prediction_variance
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .fixed_year_multiband import summarize
from .io import save_json
from .nsc_comparison import comparison
from .pilot_production import quality_detections,valid_epochs
from .precision_v03 import calibration_groups,fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate
from .year_opportunities import utc_year

NAMES=('A_r','B_new_nights','C_all','B1_one_per_new_night')
PAIRS=(('B_new_nights','A_r'),('C_all','B_new_nights'),('C_all','A_r'),('B1_one_per_new_night','A_r'),('B_new_nights','B1_one_per_new_night'))


def night_labels(data):
    stamps,inverse=np.unique(data.observation_time.to_numpy(),return_inverse=True)
    return np.floor(Time(stamps,format='mjd',scale='tt').utc.mjd[inverse]-.75).astype(int)


def split_members(r_index,all_index,data,night):
    r_index=np.asarray(r_index,int);all_index=np.asarray(all_index,int)
    if len(set(r_index))!=len(r_index) or len(set(all_index))!=len(all_index) or not set(r_index)<=set(all_index):raise ValueError('r members must be a unique identity-preserving subset')
    nsc=data.survey.eq('NSC').to_numpy();band=data.band.to_numpy();visit=data.exposure_id.to_numpy()
    if np.any(band[r_index]!='r') or np.any(~nsc[all_index]&(band[all_index]!='r')):raise ValueError('SDSS positions and the frozen baseline must be r only')
    if set(all_index[~nsc[all_index]])!=set(r_index[~nsc[r_index]]):raise ValueError('SDSS physical detections changed')
    extra=np.array([i for i in all_index if i not in set(r_index)],int)
    if np.any(~nsc[extra]) or np.any(band[extra]=='r'):raise ValueError('Only non-r NSC detections may be appended')
    original_nights=set(night[r_index[nsc[r_index]]]);new=extra[~np.isin(night[extra],list(original_nights))]
    first=[]
    for value in sorted(set(night[new])):
        choices=new[night[new]==value]
        first.append(min(choices,key=lambda i:(data.observation_time.iat[i],str(visit[i]),str(data.detection_id.iat[i]))))
    # Preserve C order exactly for a numerical replay of v1.4.
    return dict(A_r=r_index,B_new_nights=np.r_[r_index,new],C_all=all_index,B1_one_per_new_night=np.r_[r_index,np.asarray(first,int)])


def information(index,union,data,time,sigma,night,keys,fit):
    nsc=data.survey.eq('NSC').to_numpy();late=index[nsc[index]];late_union=union[nsc[union]]
    seasonal=dict(zip(late_union,seasons(time[late_union],90)))
    w=np.asarray(fit['robust_weight'])[:,None]/sigma[index]**2
    mean=(w*time[index,None]).sum(axis=0)/w.sum(axis=0)
    info=w*(time[index,None]-mean)**2;total=info.sum(axis=0)
    groups={};effective={}
    for label,tags in [('physical_group',keys[index]),('observing_night',np.char.add(data.survey.iloc[index].to_numpy(str),night[index].astype(str)))]:
        part=np.array([info[np.asarray(tags)==g].sum(axis=0) for g in sorted(set(tags))])
        fraction=part/total
        groups[label]=fraction.max(axis=0);effective[label]=1/np.sum(fraction**2,axis=0)
    return dict(detections=len(index),nsc_exposures=data.exposure_id.iloc[late].nunique(),nsc_nights=len(set(night[late])),
        nsc_seasons=len({seasonal[i] for i in late}),nsc_span_years=float(np.ptp(time[late])) if len(late) else 0.,joint_span_years=float(np.ptp(time[index])),
        sdss_physical_runs=data.exposure_id.iloc[index[~nsc[index]]].nunique(),diagonal_weight_time_information=total,
        largest_information_fraction=groups,effective_information_groups=effective,
        independent_pm_variance=np.diag(fit['independent_tangent_covariance'])[[1,3]],reference_pm_variance=np.diag(fit['reference_tangent_covariance'])[[1,3]])


def summarize_information(rows):
    if not rows:return dict(n=0)
    return dict(n=len(rows),nsc_nights_p16_p50_p84=np.quantile([r['nsc_nights'] for r in rows],[.16,.5,.84]),
        nsc_seasons_p16_p50_p84=np.quantile([r['nsc_seasons'] for r in rows],[.16,.5,.84]),
        nsc_span_p16_p50_p84_years=np.quantile([r['nsc_span_years'] for r in rows],[.16,.5,.84]),
        joint_span_p16_p50_p84_years=np.quantile([r['joint_span_years'] for r in rows],[.16,.5,.84]),
        diagonal_weight_time_information_p16_p50_p84=np.quantile([r['diagonal_weight_time_information'] for r in rows],[.16,.5,.84],axis=0),
        max_group_information_fraction_p16_p50_p84=np.quantile([r['largest_information_fraction']['physical_group'] for r in rows],[.16,.5,.84],axis=0),
        max_night_information_fraction_p16_p50_p84=np.quantile([r['largest_information_fraction']['observing_night'] for r in rows],[.16,.5,.84],axis=0))


def qso(work,trial):
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());settings.update(bootstrap_draws=800,bootstrap_absolute_zero=True,min_publish_count=5)
    source=(trial/'source_receipt.json').resolve().parent;graph=(trial/'new_reference_receipt.json').resolve().parent/'private_c0.npz';center=reference_frame(graph)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data);solution=read_offsets(graph);lookup=pd.Index(data.detection_id);night=night_labels(data)
    with gzip.open(trial/'private_fixed_qso.json.gz','rt') as f:original=json.load(f)
    output=[];maximum=0.;exits=Counter()
    for row in original:
        if not {'quality_r','all_nsc_w0'}<=row['members'].keys():exits['no_fixed_pair']+=1;continue
        ix=lookup.get_indexer(row['members']['quality_r']);full=lookup.get_indexer(row['members']['all_nsc_w0'])
        if (np.r_[ix,full]<0).any():raise ValueError('Frozen detection missing')
        indices=split_members(ix,full,data,night);values={};details={};fits={};quality={}
        for name,index in indices.items():
            fit=evaluated_fit(index,data,xy,time,sigma,keys,solution,center,config['reference_epoch']);fits[name]=fit
            values[name]=(fit['pm_icrs_masyr'],fit['pm_error_masyr']);details[name]=information(index,full,data,time,sigma,night,keys,fit)
            quality[name]=bool(np.max(fit['pm_error_masyr'])<=settings['production']['max_axis_pm_error_masyr'])
        for new,old in [('A_r','quality_r'),('C_all','all_nsc_w0')]:
            np.testing.assert_allclose(values[new],row['values'][old],rtol=1e-9,atol=1e-8)
            maximum=max(maximum,float(np.max(abs(np.asarray(values[new])-np.asarray(row['values'][old])))))
        for name in ('NSC','Xiao'):
            if name in row['values']:values[name]=row['values'][name]
        output.append(dict(magnitude=row['magnitude'],sdss_runs=row['sdss_runs'],values=values,information=details,formal_error_pass=quality,
            members={k:data.detection_id.iloc[v].tolist() for k,v in indices.items()},qso_local_index=row['qso_local_index']))
    cohorts={}
    for low in (20,21,22):
        for tag,select in [('all',lambda r:True),('one_sdss_run',lambda r:r['sdss_runs']==1),('multiple_sdss_runs',lambda r:r['sdss_runs']>1)]:
            part=[r for r in output if low<=r['magnitude']<low+1 and select(r)];common=[r for r in part if {'NSC','Xiao'}<=r['values'].keys()]
            cohorts[f'r{low}__{tag}']=dict(fixed_qso=comparison([{k:r['values'][k] for k in NAMES} for r in part],settings,PAIRS),
                strict_published_common=comparison([r['values'] for r in common],settings,PAIRS+(('C_all','Xiao'),('C_all','NSC'))),
                information={name:summarize_information([r['information'][name] for r in part]) for name in NAMES},
                formal_error_pass_counts={name:sum(r['formal_error_pass'][name] for r in part) for name in NAMES})
    result=dict(cohorts=cohorts,exits=dict(exits),maximum_frozen_pm_or_error_replay_difference=maximum,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Identical frozen source/r/SDSS detections and unchanged C0/W0. A r; B adds all other-band measurements on nights without NSC r; C adds other bands on original r nights. B1 uses the first physical exposure on each new night, not a result-selected exposure. Seasons use the C union and 90day gaps. Time information/leverage is conditional diagonal W0 information, not proof of independent nightly errors. Full reference 4x4 covariance retained. Formal-error pass counts describe this fixed QSO sample, not catalogue yield.')
    check_aggregate(result);save_json(work/'nsc_information_summary.json',result);save_json(work/'private_information_qso.json',output)


def year(work,trial):
    summary=json.loads((trial/'allband_blind_year_summary.json').read_text());config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    if summary['status']!='completed':save_json(work/'nsc_information_summary.json',dict(status=summary['status']));save_json(work/'private_information_held.json',[]);return
    source=(trial/'source_receipt.json').resolve().parent;graph=(trial/'reference_receipt.json').resolve().parent/'private_c0.npz';center=reference_frame(graph)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data);solution,gauges=read_offsets(graph,with_gauge=True);solution={k:v for k,v in solution.items() if gauges[k]==summary['reference_gauge']};lookup=pd.Index(data.detection_id);night=night_labels(data)
    training=json.loads((trial/'private_training.json').read_text());held=json.loads((trial/'private_held.json').read_text());models={};maximum=0.;nsc=data.survey.eq('NSC').to_numpy();years=utc_year(data.observation_time.to_numpy())
    for slot,row in enumerate(training):
        if row['kind']!='r_existing':continue
        full=lookup.get_indexer(row['detection_ids']);ix=full[keep[full]]
        if (full<0).any() or np.any(nsc[full]&(years[full]==summary['year'])):raise ValueError('Missing training members or held-year leakage')
        models[slot]={}
        for name,index in split_members(ix,full,data,night).items():
            fit,_=fit_with_calibration(xy[index],time[index],sigma[index],config['reference_epoch'],[solution[k] for k in keys[index]]);models[slot][name]=(fit,index)
        np.testing.assert_allclose(models[slot]['C_all'][0].coeff[1],row['motion'],rtol=1e-9,atol=1e-8)
    records=[]
    for row in held:
        slot=row['slot']
        if slot not in models:continue
        index=lookup.get_indexer([row['held_detection_id']])[0]
        if index<0:raise ValueError('Held detection missing')
        residual=[];errors=[]
        for name in NAMES:
            fit,use=models[slot][name];residual.append(xy[index]-solution[keys[index]][0]-fit.predict(time[index]));errors.append(np.sqrt(held_prediction_variance(fit,time[use],sigma[use],[solution[k] for k in keys[use]],solution[keys[index]],sigma[index],time[index])))
        np.testing.assert_allclose(residual[2],row['residual_mas'],rtol=1e-9,atol=1e-7)
        records.append(dict(slot=slot,magnitude=training[slot]['magnitude'],residual=np.asarray(residual),error=np.asarray(errors)))
    cohorts={}
    for low in (20,21,22):
        selected=[r for r in records if low<=r['magnitude']<low+1];cohorts['r'+str(low)]={}
        for a,b in PAIRS:
            rows=[dict(r,residual=r['residual'][[NAMES.index(b),NAMES.index(a)]],error=r['error'][[NAMES.index(b),NAMES.index(a)]]) for r in selected]
            cohorts['r'+str(low)][a+'_vs_'+b]=summarize(rows,names=(b,a))
    result=dict(status='completed',year=summary['year'],fixed_r_training_sources=len(models),cohorts=cohorts,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Whole held year removed before original blind association. Same frozen r-existing objects and same C0-gated physical held measurements for every model; no held target tunes coefficients. Conditional fixed-match prediction, not rerun matching or absolute PM truth. B1 separates new-night first positions from additional new-night centroids. Shared train-held reference covariance retained.')
    check_aggregate(result);save_json(work/'nsc_information_summary.json',result);save_json(work/'private_information_held.json',records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('qso','year'));p.add_argument('work',type=Path);p.add_argument('trial',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);files=('private_fixed_qso.json.gz',) if a.mode=='qso' else ('private_training.json','private_held.json','allband_blind_year_summary.json')
    bind(a.work,'trial_receipt.json',verify_directory(a.trial,files));output='private_information_qso.json' if a.mode=='qso' else 'private_information_held.json'
    with run_stage(a.work,PRECISION,a.mode,'nsc-time-information-v15',inputs=('trial_receipt.json',),outputs=('nsc_information_summary.json',output)) as go:
        if go:{'qso':qso,'year':year}[a.mode](a.work,a.trial)


if __name__=='__main__':main()
