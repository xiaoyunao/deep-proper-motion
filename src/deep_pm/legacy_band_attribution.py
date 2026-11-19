"""Replay legacy extra-source W2 fits before attributing their observed zero points.

This is a conditional linear decomposition of saved robust fits, not a causal
model, a new selection, or permission to tune corrections on acceptance QSOs.
"""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .astrometry import project
from .condition_error_trial import families
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .dark_precision import query_unique
from .exposure_catalogues import coordinates,inside
from .io import save_json
from .nsc_comparison import comparison
from .pilot_production import quality_detections,valid_epochs
from .precision_v03 import calibration_groups,fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate
from .standard_motion import calibrated_covariance,icrs_motion


def conditional_contributions(model,time,sigma,offsets,raw_positions,stationary_position,center,labels):
    """Sum conditional influence by physical family; mean parts close exactly."""
    covariance,_,_,response=calibrated_covariance(model,time,sigma,offsets)
    jac=icrs_motion(model.coeff[0],model.coeff[1],center,covariance)['state_jacobian'][np.ix_([1,3],[1,3])]
    h=response[[1,3]]
    raw=np.asarray(raw_positions)-stationary_position
    correction=np.array([v[0] for v in offsets])
    parts={}
    for tag in sorted(set(labels)):
        use=labels==tag
        first=jac@np.einsum('pna,na->p',h[:,use],raw[use])
        second=-jac@np.einsum('pna,na->p',h[:,use],correction[use])
        parts[str(tag)]=dict(raw_position=first,C0_zero_point=second,total=first+second)
    total=sum(row['total'] for row in parts.values())
    expected=jac@model.coeff[1]
    np.testing.assert_allclose(total,expected,rtol=1e-8,atol=1e-7)
    return parts,float(np.max(abs(total-expected)))


def run(work,trial):
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());settings.update(bootstrap_draws=800,bootstrap_absolute_zero=True)
    source=(trial/'source_receipt.json').resolve().parent;shared=(trial/'shared_receipt.json').resolve().parent;geometry=(trial/'geometry_config.toml').resolve()
    spec=tomllib.loads(geometry.read_text())['fields'][json.loads((trial/'run_manifest.json').read_text())['region']]
    center=reference_frame(shared/'private_c0.npz');solution,gauges=read_offsets(shared/'private_c0.npz',with_gauge=True)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
    votes=pd.Series(keys[keep&inside(data.ra_deg,data.dec_deg,spec)]).map(gauges).value_counts();gauge=min(votes.index,key=lambda k:(-votes[k],k)) if len(votes) else None
    solution={k:v for k,v in solution.items() if gauges[k]==gauge}
    saved=json.loads((trial/'private_extra_training_sources.json').read_text());summary=json.loads((trial/'error_trial_summary.json').read_text())
    raw_error=data[['sigma_east_mas','sigma_north_mas']].to_numpy(float);error=sigma.copy();family=families(data)
    for label,model in summary['error_models'].items():
        if model['enabled_for_trial_only']:
            p=np.asarray(model['parameters_scale_floor']);use=family==label
            error[use]=np.sqrt((raw_error[use]*p[:,0])**2+p[:,1]**2)
    qso=pd.read_csv(source/'spectra_union.csv');qso=qso.loc[inside(qso.ra,qso.dec,spec)].reset_index(drop=True)
    good=np.zeros(len(saved),bool);nearest=np.full(len(saved),-1,int)
    if len(saved) and len(qso):
        position=np.asarray([r['position_deg'] for r in saved]);radius=lambda mas:2*np.sin(np.deg2rad(mas/3600000)/2)
        good,nearest=query_unique(cKDTree(coordinates(qso.ra.to_numpy(),qso.dec.to_numpy())),coordinates(*position.T),radius(500),radius(300))
    claims=Counter(nearest[good]);lookup=pd.Index(data.detection_id);records=[];maximum=0.;closure=0.;missing=Counter()
    stamps,inverse=np.unique(data.observation_time,return_inverse=True);year=np.array([int(v[:4]) for v in Time(stamps,format='mjd',scale='tt').utc.isot])[inverse]
    label=(data.survey+':'+data.band.astype(str)).to_numpy()
    for j,row in enumerate(saved):
        if not good[j] or claims[nearest[j]]!=1:continue
        index=lookup.get_indexer(row['detection_ids'])
        if (index<0).any():raise ValueError('Saved legacy detection IDs missing')
        held_year=summary['annual_same_errors_multiband']['year']
        if (nsc[index]&(year[index]==held_year)).any():raise ValueError('Legacy saved target contains held year')
        offsets=[solution[k] for k in keys[index]]
        model,err=fit_with_calibration(xy[index],time[index],error[index],config['reference_epoch'],offsets)
        total=calibrated_covariance(model,time[index],error[index],offsets)[0];std=icrs_motion(model.coeff[0],model.coeff[1],center,total)
        difference=float(np.max(abs(std['pm_icrs_masyr']-row['pm_icrs_masyr'])));maximum=max(maximum,difference)
        np.testing.assert_allclose(std['pm_icrs_masyr'],row['pm_icrs_masyr'],rtol=1e-9,atol=1e-8)
        np.testing.assert_allclose(std['pm_error_masyr'],row['pm_error_masyr'],rtol=1e-9,atol=1e-8)
        reference=project([qso.ra.iloc[nearest[j]]],[qso.dec.iloc[nearest[j]]],center)[0]
        by_band,delta=conditional_contributions(model,time[index],error[index],offsets,xy[index],reference,center,label[index]);closure=max(closure,delta)
        by_year,delta=conditional_contributions(model,time[index],error[index],offsets,xy[index],reference,center,np.char.add(label[index].astype(str),np.char.add(':',year[index].astype(str))));closure=max(closure,delta)
        alternatives={}
        for band in sorted(set(data.band.iloc[index[nsc[index]]])):
            left=index[~(nsc[index]&data.band.iloc[index].eq(band).to_numpy())]
            if not valid_epochs(left,nsc,time,settings['production']):missing[row['r_bin']+'__without_'+band]+=1;continue
            fit,_=fit_with_calibration(xy[left],time[left],error[left],config['reference_epoch'],[solution[k] for k in keys[left]])
            cov=calibrated_covariance(fit,time[left],error[left],[solution[k] for k in keys[left]])[0];converted=icrs_motion(fit.coeff[0],fit.coeff[1],center,cov)
            alternatives[band]=(converted['pm_icrs_masyr'],converted['pm_error_masyr'])
        records.append(dict(r_bin=row['r_bin'],values=(std['pm_icrs_masyr'],std['pm_error_masyr']),
            conditional_band_contributions=by_band,conditional_band_year_contributions=by_year,without_band=alternatives,
            detection_ids=row['detection_ids'],legacy_position_deg=row['position_deg']))
    cohorts={}
    for tag in ('20_to_21','21_to_22','22_to_23'):
        selected=[r for r in records if r['r_bin']==tag];parts={}
        for dimension in ('conditional_band_contributions','conditional_band_year_contributions'):
            labels=sorted({key for row in selected for key in row[dimension]})
            parts[dimension]={key:dict(total_sample_count=len(selected),contributors=sum(key in row[dimension] for row in selected),
                mean_components_masyr={term:np.mean([row[dimension].get(key,{}).get(term,np.zeros(2)) for row in selected],axis=0) for term in ('raw_position','C0_zero_point','total')}) for key in labels}
        deletions={}
        for band in sorted({b for row in selected for b in row['without_band']}):
            rows=[dict(full=row['values'],without=row['without_band'][band]) for row in selected if band in row['without_band']]
            deletions[band]=comparison(rows,settings,(('full','without'),))
        cohorts[tag]=dict(n=len(selected),replayed_zero=comparison([dict(multi=row['values']) for row in selected],settings),mean_attribution=parts,leave_band_out=deletions)
    result=dict(unique_spectral_extra_sources=len(records),maximum_replayed_pm_difference_masyr=maximum,maximum_linear_closure_difference_masyr=closure,
        cohorts=cohorts,insufficient_time_deletions=dict(missing),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Exact saved extra-source detection order, original W2 scale/floors and original reference graph replayed before analysis. Conditional robust-influence decomposition into native positions and C0 shifts by band/year. Mean contributions, including zeros for absent measurements, add to mean PM; medians do not add. Not causal, no QSO-tuned correction, no new adopted selection.')
    check_aggregate(result);save_json(work/'legacy_band_attribution_summary.json',result);save_json(work/'private_attributions.json',records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('work',type=Path);p.add_argument('trial',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'trial_receipt.json',verify_directory(a.trial,('error_trial_summary.json','private_extra_training_sources.json')))
    with run_stage(a.work,PRECISION,'legacy_extra_sources','legacy-band-attribution-v14',inputs=('trial_receipt.json',),outputs=('legacy_band_attribution_summary.json','private_attributions.json')) as go:
        if go:run(**vars(a))


if __name__=='__main__':main()
