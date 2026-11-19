"""Train empirical independent/night variance components without QSO tuning.

This tests identifiability rather than presupposing a correct error model. Native
errors may already contain calibration variance. The learned components are a
conditional empirical decomposition; reference-bootstrap variance is accounted
for separately. Cross-band/night correlation is bracketed, not asserted known.
"""
import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.optimize import minimize

from .__main__ import tomllib
from .allband_qso import physical_candidates
from .condition_error_trial import split_reference
from .continuous_instruments import chip_receipts,instrument_by_exposure
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .correlated_motion import fit_correlated_motion
from .io import save_json
from .nsc_comparison import comparison
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,calibration_support
from .provenance import run_stage
from .public_summary import check_aggregate
from .standard_motion import icrs_motion


def physical_pairs(indices,time,night,visits):
    """Up to one within-night and one between-night pair, no reused detections."""
    order=sorted(indices,key=lambda i:time[i]);used=set();result=[]
    for same in (True,False):
        found=False
        for p,a in enumerate(order):
            if a in used:continue
            for b in order[p+1:]:
                if b not in used and visits[a]!=visits[b] and (night[a]==night[b])==same:
                    result.append((a,b));used.update((a,b));found=True;break
            if found:break
    return result


def variance_components(delta,native_square_sum,different_night,known_variance):
    delta=np.asarray(delta);native_square_sum=np.asarray(native_square_sum);different_night=np.asarray(different_night,float)
    known_variance=np.asarray(known_variance)
    design=np.column_stack([native_square_sum,np.ones(len(delta)),different_night])
    norms=np.linalg.norm(design,axis=0)
    if (norms==0).any() or np.linalg.matrix_rank(design/norms)<3:raise ValueError('Variance components are not identifiable')
    def objective(p):
        variance=p[0]**2*native_square_sum+2*(50*p[1])**2+2*different_night*(50*p[2])**2+known_variance
        if (variance<=0).any():return np.inf
        # Fixed 95% core plus 5% broad Gaussian contamination, not a calibrated source probability.
        core=np.log(.95)-.5*(np.log(variance)+delta**2/variance)
        broad=np.log(.05)-.5*(np.log(100*variance)+delta**2/(100*variance))
        return -np.logaddexp(core,broad).sum()
    results=[minimize(objective,start,method='L-BFGS-B',bounds=((.25,4),(0,4),(0,4))) for start in ([1,.7,.7],[1,1,0],[.5,.5,.5])]
    result=min(results,key=lambda r:r.fun)
    if not result.success or not np.isfinite(result.fun):raise ValueError('Variance fit did not converge')
    return np.asarray([result.x[0],50*result.x[1],50*result.x[2]])


def run(work,pilot,reference):
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    source=(pilot/'source_receipt.json').resolve().parent;graph=(pilot/'new_reference_receipt.json').resolve().parent/'private_c0.npz'
    verify_directory(source,('detections.csv',));verify_directory(graph.parent,('private_c0.npz',))
    center=reference_frame(graph);solution=read_offsets(graph)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production'])
    keys=calibration_groups(data);valid=physical_candidates(data)&calibration_support(keys,solution)
    chips,complete=chip_receipts((source/'run_manifest.json').resolve());instruments=instrument_by_exposure(chips)
    instrument=data.exposure_id.map(instruments).fillna('unknown');instrument.loc[data.survey.eq('SDSS')]='SDSS_imager'
    family=(data.survey+':'+instrument+':'+data.band.astype(str)).to_numpy()
    stamps,inverse=np.unique(data.observation_time,return_inverse=True);utc=Time(stamps,format='mjd',scale='tt').utc
    night=np.floor(utc.mjd[inverse]-.75).astype(int);visits=data.exposure_id.to_numpy()
    raw=data[['sigma_east_mas','sigma_north_mas']].to_numpy(float)
    coord=xy-np.array([solution[k][0] if k in solution else np.zeros(2) for k in keys])
    refs=pd.read_csv(reference/'reference.csv.gz',dtype={'reference':str,'is_holdout':bool})
    lookup=pd.Index(data.detection_id);indices=lookup.get_indexer(refs.detection_id)
    allowed=(indices>=0)&refs.is_holdout.to_numpy();allowed[indices>=0]&=valid[indices[indices>=0]]
    refs=refs.loc[allowed].copy();refs['input_index']=indices[allowed];refs['family']=family[indices[allowed]]
    refs['training']=refs.reference.map(lambda name:split_reference(name,9137))
    pairs={True:defaultdict(list),False:defaultdict(list)}
    for (reference_name,label),part in refs.groupby(['reference','family'],sort=False):
        arm=bool(part.training.iloc[0])
        for a,b in physical_pairs(part.input_index.to_numpy(),time,night,visits):
            common=np.var(solution[keys[a]][1]-solution[keys[b]][1],axis=0,ddof=1)
            pairs[arm][label].append(dict(delta=coord[a]-coord[b],raw=raw[a]**2+raw[b]**2,
                different=night[a]!=night[b],known=common,reference=reference_name,night_a=int(night[a]),night_b=int(night[b])))
    models={};independent=sigma.copy();night_error=np.zeros_like(sigma)
    for label,part in pairs[True].items():
        nsame=sum(not row['different'] for row in part);ndiff=len(part)-nsame
        nref=len({row['reference'] for row in part});nights={row[k] for row in part for k in ('night_a','night_b')}
        identifiable=nref>=50 and nsame>=20 and ndiff>=20 and len(nights)>=4
        parameters=[];failure=None
        if identifiable:
            try:
                for axis in range(2):parameters.append(variance_components([r['delta'][axis] for r in part],[r['raw'][axis] for r in part],[r['different'] for r in part],[r['known'][axis] for r in part]))
            except ValueError as error:failure=str(error);identifiable=False
        if identifiable:
            parameter=np.asarray(parameters);use=family==label
            independent[use]=np.sqrt((raw[use]*parameter[:,0])**2+parameter[:,1]**2);night_error[use]=parameter[:,2]
        models[label]=dict(training_references=nref,within_night_pairs=nsame,between_night_pairs=ndiff,distinct_nights=len(nights),
            identifiable=identifiable,parameter_order=['native_scale','independent_floor_mas','additional_night_mas'],
            parameters_by_axis=parameters if identifiable else None,failure=failure,
            status='conditional_trial_only' if identifiable else 'W0_fallback_no_identifiable_decomposition')
    validation={}
    for label,part in pairs[False].items():
        parameters=models.get(label,{}).get('parameters_by_axis');bykind={}
        for kind in (False,True):
            sub=[r for r in part if r['different']==kind]
            if not sub:continue
            delta=np.array([r['delta'] for r in sub]);native=np.array([r['raw'] for r in sub]);known=np.array([r['known'] for r in sub])
            w0=np.sqrt(native+2*50**2+known)
            calibrated=np.sqrt(native*np.asarray(parameters)[:,0]**2+2*np.asarray(parameters)[:,1]**2+2*kind*np.asarray(parameters)[:,2]**2+known) if parameters else w0
            med=np.median(delta,axis=0)
            bykind['different_nights' if kind else 'same_night']=dict(pair_count=len(sub),physical_references=len({r['reference'] for r in sub}),
                median_mas=med,nmad_mas=1.4826*np.median(abs(delta-med),axis=0),
                W0_within_1sigma=(abs(delta)<=w0).mean(axis=0),trial_within_1sigma=(abs(delta)<=calibrated).mean(axis=0))
        validation[label]=bykind
    with gzip.open(pilot/'private_fixed_qso.json.gz','rt') as stream:objects=json.load(stream)
    records=[]
    labels=[('C_within_band_night',np.char.add(family.astype(str),np.char.add(':',night.astype(str)))),
            ('C_cross_band_night_bound',np.char.add((data.survey+':'+instrument).to_numpy(str),np.char.add(':',night.astype(str))))]
    calibrated_family=pd.Series(family).isin([label for label,model in models.items() if model['identifiable']]).to_numpy()
    for row in objects:
        if 'all_nsc_w0' not in row['members']:continue
        ix=lookup.get_indexer(row['members']['all_nsc_w0'])
        if (ix<0).any():raise ValueError('Fixed multiband members changed')
        values=dict(B_all_nsc_W0=row['values']['all_nsc_w0'])
        for name,group in labels:
            fit=fit_correlated_motion(coord[ix],time[ix],independent[ix],group[ix],night_error[ix],config['reference_epoch'],[solution[k] for k in keys[ix]])
            std=icrs_motion(fit.model.coeff[0],fit.model.coeff[1],center,fit.covariance)
            values[name]=(std['pm_icrs_masyr'],std['pm_error_masyr'])
        records.append(dict(magnitude=row['magnitude'],values=values,calibrated_detection_count=int(calibrated_family[ix].sum()),detection_count=len(ix)))
    opts=dict(min_publish_count=5,tail_masyr=5,bootstrap_draws=400,bootstrap_absolute_zero=True,seed=9137)
    result=dict(error_models=models,independent_validation_pairs=validation,
        fixed_qso={f'r{low}_{low+1}':comparison([r['values'] for r in records if low<=r['magnitude']<low+1],opts,(('C_within_band_night','B_all_nsc_W0'),('C_cross_band_night_bound','B_all_nsc_W0'))) for low in (20,21,22)},
        known_qso_count=len(records),metadata_receipts_complete=complete,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Only half the original C0-held physical galaxies train; disjoint half validates. At least 50 training references, 20 within-night and 20 between-night pairs and four nights required. QSO never train parameters. Same fixed B detections in every fit.',
        limitations='Conditional mixture-likelihood decomposition of published marginal errors; parameter-learning uncertainty and cross-axis night covariance not propagated. Cross-band nightly correlation is unknown: independent-band and fully shared-instrument night terms bracket assumptions, not validated final models. Source/physical-night validation and independent-year testing required.',
        adoption='not_adopted_pending_independent_error_and_year_validation')
    check_aggregate(result);save_json(work/'multiband_error_trial_summary.json',result);save_json(work/'private_qso_trials.json',records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','pilot','reference'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'pilot_receipt.json',verify_directory(a.pilot,('private_fixed_qso.json.gz',)))
    bind(a.work,'reference_receipt.json',verify_directory(a.reference,('reference.csv.gz',)))
    with run_stage(a.work,PRECISION,'ordinary_fixed_qso','multiband-error-v14',inputs=('pilot_receipt.json','reference_receipt.json'),outputs=('multiband_error_trial_summary.json','private_qso_trials.json')) as go:
        if go:run(**vars(a))


if __name__=='__main__':main()
