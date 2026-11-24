"""Fixed-detection QSO reference-frame replay and provisional all-filter tests.

Spectra identify stationary validation objects only. No QSO estimates a reference
zero or a position-error parameter. Object-index acquisition remains a selection
of the retrieved catalogue, not image or blind-association completeness.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.time import Time

from .__main__ import tomllib
from .astrometry import fit_motion, project
from .continuous_reference import PRECISION, SPATIAL, bind, read_offsets, reference_frame, verify_directory
from .continuous_instruments import chip_receipts, instrument_by_exposure
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .independent_summary import standard_block
from .io import save_json, json_default
from .nsc_comparison import comparison, qso_matches
from .pilot_production import quality_detections, valid_epochs
from .precision_v03 import calibration_groups, calibration_support
from .provenance import run_stage
from .public_summary import check_aggregate
from .standard_motion import calibrated_covariance, icrs_motion


def physical_candidates(data):
    """Every band is eligible; native errors/flags/SNR are necessary, not sufficient."""
    native=data[['sigma_east_mas','sigma_north_mas']].to_numpy(float)
    valid=np.isfinite(native).all(axis=1)&(native>0).all(axis=1)&(native<=200).all(axis=1)
    valid &= np.isfinite(data[['ra_deg','dec_deg','observation_time']].to_numpy(float)).all(axis=1)
    valid &= data.magnitude.notna().to_numpy() & data.magnitude_error.between(0,2.5/np.log(10)/5,inclusive='right').to_numpy()
    flags=pd.to_numeric(data.quality_flags,errors='coerce').to_numpy(float)
    defined=np.isfinite(flags)&(flags>=0)&(flags<=255)&(flags==np.floor(flags))
    fatal=np.where(defined,flags,0).astype(np.int64)&252
    # No new interpretation of the scalar legacy SDSS flags: SDSS remains r only.
    nsc=data.survey.eq('NSC').to_numpy()
    return valid&np.where(nsc,defined&(fatal==0),data.band.eq('r').to_numpy())


def evaluated_fit(index,data,xy,time,sigma,keys,solution,center,epoch, *, model=None):
    offsets = [solution[k] for k in keys[index]]
    if model is None:
        corrections = np.array([row[0] for row in offsets])
        model = fit_motion(xy[index] - corrections, time[index], sigma[index], epoch)
    total,independent,common,_=calibrated_covariance(model,time[index],sigma[index],offsets)
    result=icrs_motion(model.coeff[0],model.coeff[1],center,total)
    result.update(tangent_state=model.coeff.T.ravel(),reference_tangent_covariance=common,
                  independent_tangent_covariance=independent,robust_weight=model.weight,
                  span_years=float(np.ptp(time[index])),used_count=len(index))
    w=model.weight[:,None]/sigma[index]**2
    mean=np.average(np.tile(time[index,None],(1,2)),axis=0,weights=w)
    result['time_leverage_year2_mas_minus2']=(w*(time[index,None]-mean)**2).sum(axis=0)
    return result


def summaries(records,exits):
    settings=dict(min_publish_count=5,tail_masyr=5,bootstrap_draws=400,bootstrap_absolute_zero=True,seed=9137)
    result={}
    methods=sorted({k for r in records for k in r['values']})
    contrasts=tuple((a,b) for a,b in [('joint_r','frozen_r'),('quality_r','joint_r'),('all_nsc_w0','quality_r'),('all_nsc_w0','joint_r'),('all_nsc_w0','Xiao'),('all_nsc_w0','NSC')] if a in methods and b in methods)
    for low in (20,21,22):
        sample=[r for r in records if low<=r['magnitude']<low+1]
        strata={'all':sample,'one_sdss_run':[r for r in sample if r['sdss_runs']==1],
                'multiple_sdss_runs':[r for r in sample if r['sdss_runs']>1]}
        for label,selected in strata.items():
            families={}
            for required in [('frozen_r','joint_r'),('joint_r','quality_r','all_nsc_w0'),('NSC','Xiao','joint_r','quality_r','all_nsc_w0')]:
                if not set(required)<=set(methods):continue
                part=[r for r in selected if set(required)<=r['values'].keys()]
                values=[{k:r['values'][k] for k in required} for r in part]
                families['_'.join(required)]=comparison(values,settings,contrasts)
            result[f'r{low}_{low+1}__{label}']=families
    delta=[np.asarray(r['values']['joint_r'][0])-r['values']['frozen_r'][0] for r in records if 'joint_r' in r['values']]
    return dict(known_qso_count=len(records),exits=dict(exits),cohorts=result,
                same_detection_reference_pm_change=dict(n=len(delta),median_masyr=np.median(delta,axis=0),
                nmad_masyr=1.4826*np.median(abs(np.asarray(delta)-np.median(delta,axis=0)),axis=0)) if delta else dict(n=0))


def run(work,source,oldfit,oldshared,newshared,spec,multiband):
    start=monotonic();config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    data,oldxy,time,sigma,keep=quality_detections(source,reference_frame(oldshared),config,settings['production'])
    if data.detection_id.duplicated().any():raise ValueError('Duplicate physical detection IDs in fixed source')
    keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
    old_solution,old_gauges=read_offsets(oldshared,with_gauge=True)
    gauge=json.loads((oldfit/'continuous_core_summary.json').read_text())['reference_gauge']
    old_solution={k:v for k,v in old_solution.items() if old_gauges[k]==gauge}
    solution,gauges=read_offsets(newshared,with_gauge=True)
    center=reference_frame(newshared);xy=project(data.ra_deg,data.dec_deg,center)
    oldcenter=reference_frame(oldshared)
    qso=pd.read_csv(source/'spectra_union.csv');qso=qso.loc[inside(qso.ra,qso.dec,spec)].reset_index(drop=True)
    _,truth=qso_matches(data,oldxy,keep,project(qso.ra,qso.dec,oldcenter))
    original=json.loads((oldfit/'private_qso_blocks.json').read_text())
    lookup={};exits=Counter();maximum=0.
    if len(original) and len(qso):
        found,nearest=query_unique(cKDTree(project(qso.ra,qso.dec,oldcenter)),project(*np.asarray([r['position_deg'] for r in original]).T,oldcenter),500,300)
        count=Counter(nearest[found])
        for j in np.flatnonzero(found):
            k=int(nearest[j])
            if count[k]!=1:continue
            lookup[k]=standard_block(original[j],oldcenter)
    exits['frozen_blocks_without_unique_stationary_position_identity']=len(original)-len(lookup)
    candidates=physical_candidates(data)
    chips,complete=chip_receipts((source/'run_manifest.json').resolve());instruments=instrument_by_exposure(chips)
    extra_mask=candidates&nsc&~data.band.eq('r').to_numpy()
    _,alltruth=qso_matches(data,xy,extra_mask,project(qso.ra,qso.dec,center)) if multiband else (None,{})
    records=[]
    for k,index in truth.items():
        magnitude=float(qso.reference_r.iloc[k])
        if not 20<=magnitude<23:continue
        ix=index[calibration_support(keys[index],old_solution)]
        if not valid_epochs(ix,nsc,time,settings['production']):continue
        old=evaluated_fit(ix,data,oldxy,time,sigma,keys,old_solution,oldcenter,config['reference_epoch'])
        saved=lookup.get(k)
        if magnitude>=21 and saved is None:
            exits['eligible_faint_without_frozen_identity']+=1;continue
        if saved is not None:
            if sorted(set(data.exposure_id.iloc[ix]))!=saved['exposures']:raise ValueError('Frozen exposures changed')
            difference=float(np.max(abs(np.asarray(saved['motion_icrs']['C0'])-old['pm_icrs_masyr'])))
            maximum=max(maximum,difference)
            np.testing.assert_allclose(saved['motion_icrs']['C0'],old['pm_icrs_masyr'],rtol=1e-9,atol=1e-8)
        values={'frozen_r':(old['pm_icrs_masyr'],old['pm_error_masyr'])}
        if saved:
            for name in ('NSC','Xiao'):
                if name in saved['motion']:values[name]=(saved['motion'][name],saved['error_icrs'][name])
        fits={};members={}
        if not calibration_support(keys[ix],solution).all():exits['joint_missing_fixed_r_support']+=1
        elif len({gauges[key] for key in keys[ix]})!=1:exits['joint_split_gauge_fixed_r']+=1
        else:
            tag=gauges[keys[ix[0]]];local={key:v for key,v in solution.items() if gauges[key]==tag}
            choices={'joint_r':ix}
            if multiband:
                qualified=ix[candidates[ix]]
                extra=alltruth.get(k,np.empty(0,int));extra=extra[calibration_support(keys[extra],local)]
                if valid_epochs(qualified,nsc,time,settings['production']):
                    choices['quality_r']=qualified;choices['all_nsc_w0']=np.concatenate([qualified,extra])
                else:exits['quality_r_insufficient_time']+=1
            for name,use in choices.items():
                if data.exposure_id.iloc[use].duplicated().any():raise ValueError('One source claimed two detections in one exposure')
                fits[name]=evaluated_fit(use,data,xy,time,sigma,keys,local,center,config['reference_epoch'])
                values[name]=(fits[name]['pm_icrs_masyr'],fits[name]['pm_error_masyr'])
                members[name]=data.detection_id.iloc[use].tolist()
            if multiband and 'all_nsc_w0' in fits:
                joined=np.concatenate([choices['quality_r'],extra]);byband={}
                for band in sorted(set(data.band.iloc[extra])):
                    left=joined[~(nsc[joined]&data.band.iloc[joined].eq(band).to_numpy())]
                    if valid_epochs(left,nsc,time,settings['production']):
                        trial=evaluated_fit(left,data,xy,time,sigma,keys,local,center,config['reference_epoch'])
                        byband[band]=(trial['pm_icrs_masyr'],trial['pm_error_masyr'])
                fits['leave_one_band_out']=byband
        visits=sorted(set(data.exposure_id.iloc[ix]));utc=Time(data.observation_time.iloc[ix].to_numpy(),format='mjd',scale='tt').utc
        records.append(dict(magnitude=magnitude,values=values,fits=fits,members=members,
            frozen_detection_ids=data.detection_id.iloc[ix].tolist(),qso_local_index=int(k),
            exposures=visits,sdss_runs=len({e for e in visits if e.startswith('SDSS:')}),
            instrument_mix='_and_'.join(sorted({instruments.get(e,'unknown') for e in visits if e.startswith('NSC:')})),
            nsc_epoch_days=sorted(set(np.floor(utc.mjd[nsc[ix]]-.75).astype(int))),
            nsc_span_years=float(np.ptp(time[ix[nsc[ix]]])),joint_span_years=float(np.ptp(time[ix]))))
    result=summaries(records,exits)
    result.update(frozen_block_count=len(original),maximum_frozen_motion_difference_masyr=maximum,
        metadata_receipts_complete=complete,elapsed_seconds=monotonic()-start,
        provisional_multiband=multiband,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Position-identified stationary spectra; identical frozen r detections replayed before reference comparison. All-filter trial is provisional native-error/SNR/fatal-flag qualified, not adoption or blind completeness. Faint Gaia proper motions never read. Native plane error approximation unchanged.')
    check_aggregate(result);save_json(work/'allband_qso_summary.json',result)
    with gzip.open(work/'private_fixed_qso.json.gz','wt') as stream:json.dump(records,stream,default=json_default,allow_nan=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','source','oldfit','oldshared','newshared','config'):p.add_argument(name,type=Path)
    p.add_argument('region');p.add_argument('--multiband',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    for name,path,files in [('source',a.source,('detections.csv','spectra_union.csv')),('oldfit',a.oldfit,('private_qso_blocks.json','continuous_core_summary.json'))]:
        bind(a.work,name+'_receipt.json',verify_directory(path,files))
    for name,path in [('old',a.oldshared),('new',a.newshared)]:bind(a.work,name+'_reference_receipt.json',verify_directory(path.parent,(path.name,)))
    option=a.work/'options.json'
    expected=dict(multiband=a.multiband)
    if option.exists() and json.loads(option.read_text())!=expected:raise ValueError('Changed trial options')
    if not option.exists():save_json(option,expected)
    with run_stage(a.work,a.config,a.region,'fixed-qso-allband-v14',inputs=('source_receipt.json','oldfit_receipt.json','old_reference_receipt.json','new_reference_receipt.json','options.json'),outputs=('allband_qso_summary.json','private_fixed_qso.json.gz')) as go:
        if go:run(a.work,a.source,a.oldfit,a.oldshared,a.newshared,tomllib.loads(a.config.read_text())['fields'][a.region],a.multiband)


if __name__=='__main__':main()
