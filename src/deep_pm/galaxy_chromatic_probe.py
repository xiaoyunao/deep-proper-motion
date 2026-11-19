"""Train/held-galaxy chromatic residual diagnostics, never a QSO zero fit.

Nested empirical constant/color/magnitude regressions test residual predictive
structure. Their intercept is not an adopted extra band zero: exposure zeros
must be refit jointly before any coherent calibrated model can be adopted.
"""
import argparse
import json
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .__main__ import tomllib
from .allband_qso import physical_candidates
from .continuous_instruments import chip_receipts,instrument_by_exposure
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .exposure_catalogues import inside
from .io import save_json
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,calibration_support
from .provenance import run_stage
from .public_summary import check_aggregate


def regression(design,target,error):
    if np.linalg.matrix_rank(design)<design.shape[1]:raise ValueError('Diagnostic predictors not identifiable')
    weight=np.ones(len(target))
    for iteration in range(8):
        coeff=np.column_stack([np.linalg.lstsq(design*np.sqrt(weight)[:,None]/error[:,a,None],target[:,a]*np.sqrt(weight)/error[:,a],rcond=None)[0] for a in range(2)])
        residual=target-design@coeff
        if iteration<7:weight=np.minimum(1,6/(4+np.sum((residual/error)**2,axis=1)))
    return coeff


def correlation(x,y):
    if len(x)<20 or np.ptp(x)==0 or np.ptp(y)==0:return None
    value=float(spearmanr(x,y).correlation)
    return value if np.isfinite(value) else None


def assess(train,held):
    if len(train)<100 or len(held)<30:return dict(training_count=len(train),validation_count=len(held),status='insufficient_independent_galaxies')
    def arrays(part):
        x=np.asarray([[1.,r['color']-1.2,r['r']-20.] for r in part]);y=np.asarray([r['residual'] for r in part]);err=np.asarray([r['error'] for r in part]);return x,y,err
    x,y,error=arrays(train);xv,yv,ev=arrays(held)
    models={};radial=[]
    for name,n in [('constant',1),('color',2),('color_magnitude',3)]:
        try:coef=regression(x[:,:n],y,error)
        except ValueError:return dict(training_count=len(train),validation_count=len(held),status='predictor_rank_deficient')
        residual=yv-xv[:,:n]@coef;median=np.median(residual,axis=0)
        models[name]=dict(coefficients=coef,validation_median_mas=median,validation_nmad_mas=1.4826*np.median(abs(residual-median),axis=0),validation_radial_p68_mas=np.quantile(np.linalg.norm(residual,axis=1),.68))
        radial.append(np.linalg.norm(residual,axis=1))
    samples=np.random.default_rng(9137).integers(0,len(held),size=(400,len(held)));draw=np.quantile(np.asarray(radial)[:,samples],.68,axis=2)
    gains={name:dict(relative_gain=1-models[name]['validation_radial_p68_mas']/models['constant']['validation_radial_p68_mas'],
        object_bootstrap95=np.quantile(1-draw[i]/draw[0],[.025,.975])) for i,name in enumerate(('constant','color','color_magnitude')) if i>0}
    conditional={}
    for feature in ('color','r','seeing','airmass','epoch'):
        feature_value=np.array([r[feature] for r in held]);finite=np.isfinite(feature_value)
        conditional[feature]=dict(n=int(finite.sum()),p16_p50_p84=np.quantile(feature_value[finite],[.16,.5,.84]) if finite.any() else None,
            spearman_raw_residual=[correlation(feature_value[finite],yv[finite,a]) for a in range(2)])
    return dict(status='diagnostic_only',training_count=len(train),validation_count=len(held),models=models,paired_gain_vs_constant=gains,
        held_covariates=conditional,coefficient_order=['constant_mas','per_g_minus_i','per_r_magnitude'],
        scope='Same physical held galaxies and fixed corrected positions. Object bootstrap does not certify shared-exposure/field systematics. A predicted intercept is diagnostic, not an additional free C0 band correction.')


def run(work,pilot,reference):
    source=(pilot/'source_receipt.json').resolve().parent;graph=(pilot/'new_reference_receipt.json').resolve().parent/'private_c0.npz'
    spec=tomllib.loads((pilot/'config_snapshot.toml').read_text())['fields'][json.loads((pilot/'run_manifest.json').read_text())['region']]
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());center=reference_frame(graph)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data);solution,gauges=read_offsets(graph,with_gauge=True)
    counts=pd.Series(keys[keep&inside(data.ra_deg,data.dec_deg,spec)]).map(gauges).value_counts();gauge=min(counts.index,key=lambda k:(-counts[k],k)) if len(counts) else None
    solution={k:v for k,v in solution.items() if gauges[k]==gauge};valid=physical_candidates(data)&calibration_support(keys,solution)
    coord=xy-np.array([solution[k][0] if k in solution else np.zeros(2) for k in keys])
    refs=pd.read_csv(reference/'reference.csv.gz',dtype={'reference':str,'is_holdout':bool});lookup=pd.Index(data.detection_id);index=lookup.get_indexer(refs.detection_id)
    mask=index>=0;mask[index>=0]&=valid[index[index>=0]];refs=refs.loc[mask].copy();refs['input_index']=index[mask]
    chips,complete=chip_receipts((source/'run_manifest.json').resolve());instrument=instrument_by_exposure(chips)
    family=np.where(data.survey.eq('SDSS'),'SDSS:r',data.exposure_id.map(instrument).fillna('unknown')+':'+data.band.astype(str))
    samples={True:defaultdict(list),False:defaultdict(list)}
    missing_color=0
    for _,part in refs.groupby('reference',sort=False):
        ix=part.input_index.to_numpy();early=ix[data.survey.iloc[ix].eq('SDSS').to_numpy()]
        if not len(early):continue
        color=data.color.iloc[early].to_numpy(float) if 'color' in data else np.array([]);color=color[np.isfinite(color)&(color>-1)&(color<4)]
        if not len(color):missing_color+=1;continue
        w=1/sigma[early]**2;w/=w.sum(axis=0);origin=np.sum(coord[early]*w,axis=0);origin_var=np.sum(w**2*sigma[early]**2,axis=0)
        r=float(np.median(data.magnitude.iloc[early]));held=bool(part.is_holdout.iloc[0])
        late=ix[data.survey.iloc[ix].eq('NSC').to_numpy()]
        for label in sorted(set(family[late])):
            use=late[family[late]==label];weights=1/sigma[use]**2;weights/=weights.sum(axis=0)
            point=np.sum(coord[use]*weights,axis=0);variance=origin_var+np.sum(weights**2*sigma[use]**2,axis=0)
            # Known common reference covariance of the difference; do not divide it by number of detections.
            ns=np.einsum('na,dna->da',weights,np.stack([solution[k][1] for k in keys[use]],axis=1))
            ss=np.einsum('na,dna->da',w,np.stack([solution[k][1] for k in keys[early]],axis=1))
            variance+=np.var(ns-ss,axis=0,ddof=1)
            def median(column):
                v=data[column].iloc[use].to_numpy(float) if column in data else np.array([]);v=v[np.isfinite(v)];return float(np.median(v)) if len(v) else np.nan
            samples[held][label].append(dict(color=float(np.median(color)),r=r,residual=point-origin,error=np.sqrt(variance),seeing=median('fwhm'),airmass=median('airmass'),epoch=float(np.median(time[use]))))
    result=dict(by_instrument_band={label:assess(samples[False][label],samples[True][label]) for label in sorted(set(samples[False])|set(samples[True]))},
        missing_reliable_sdss_color_references=missing_color,chip_metadata_receipts_complete=complete,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Original C0 training/held physical reference identities retained. Stationary-galaxy cross-survey residual, SDSS g-i color and r magnitude; QSO and stellar target data never train these regressions. Fits are residual diagnostics only. Global physical-group zero/color covariance must be solved before applying any model across regions.',adoption='no_new_correction_applied')
    check_aggregate(result);save_json(work/'galaxy_chromatic_probe_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','pilot','reference'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'pilot_receipt.json',verify_directory(a.pilot,('private_fixed_qso.json.gz',)));bind(a.work,'reference_receipt.json',verify_directory(a.reference,('reference.csv.gz',)))
    with run_stage(a.work,PRECISION,'ordinary_galaxy_residuals','galaxy-chromatic-probe-v14',inputs=('pilot_receipt.json','reference_receipt.json'),outputs=('galaxy_chromatic_probe_summary.json',)) as go:
        if go:run(**vars(a))


if __name__=='__main__':main()
