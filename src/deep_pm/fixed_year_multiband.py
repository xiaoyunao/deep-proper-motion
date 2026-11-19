"""Same-object, same-held-detection r versus all-band stellar prediction control."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .allband_qso import physical_candidates
from .condition_holdout import held_prediction_variance
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .independent_year import exclusive
from .io import save_json
from .pilot_production import quality_detections,valid_epochs
from .precision_v03 import calibration_groups,calibration_support,fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate
from .validation import ratio
from .year_opportunities import utc_year


def summarize(rows,names=('r_only','all_nsc')):
    if not rows:return dict(source_count=0,held_positions=0)
    vectors=np.array([r['residual'] for r in rows]);errors=np.array([r['error'] for r in rows]);groups=defaultdict(list)
    for j,row in enumerate(rows):groups[row['slot']].append(j)
    source_medians=np.array([np.median(np.linalg.norm(vectors[ix],axis=2),axis=0) for ix in groups.values()])
    p68=np.quantile(source_medians,.68,axis=0)
    sample=np.random.default_rng(9137).integers(0,len(groups),size=(800,len(groups)))
    draws=np.quantile(source_medians[sample],.68,axis=1)
    valid=draws[:,0]>0
    models={}
    for j,name in enumerate(names):
        residual=vectors[:,j];error=errors[:,j]
        models[name]=dict(source_median_radial_p68_mas=p68[j],position_radial_p68_mas=np.quantile(np.linalg.norm(residual,axis=1),.68),
            position_median_mas=np.median(residual,axis=0),position_nmad_mas=1.4826*np.median(abs(residual-np.median(residual,axis=0)),axis=0),
            position_within_1sigma=[ratio(int(count),len(rows)) for count in (abs(residual)<=error).sum(axis=0)])
    return dict(source_count=len(groups),held_positions=len(rows),models=models,
        relative_source_p68_gain=1-p68[1]/p68[0] if p68[0]>0 else None,
        object_bootstrap_gain95=np.quantile(1-draws[valid,1]/draws[valid,0],[.025,.975]) if valid.any() else None)


def run(work,trial):
    summary=json.loads((trial/'allband_blind_year_summary.json').read_text())
    if summary['status']!='completed':save_json(work/'fixed_year_multiband_summary.json',dict(status=summary['status']));save_json(work/'private_common_held.json',[]);return
    source=(trial/'source_receipt.json').resolve().parent;graph=(trial/'reference_receipt.json').resolve().parent/'private_c0.npz'
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());center=reference_frame(graph)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
    solution,gauges=read_offsets(graph,with_gauge=True);solution={k:v for k,v in solution.items() if gauges[k]==summary['reference_gauge']}
    training=json.loads((trial/'private_training.json').read_text());saved=json.loads((trial/'private_held.json').read_text());lookup=pd.Index(data.detection_id)
    models={};maximum=0.
    year=utc_year(data.observation_time.to_numpy())
    for slot,row in enumerate(training):
        if row['kind']!='r_existing':continue
        all_index=lookup.get_indexer(row['detection_ids'])
        if (all_index<0).any() or (nsc[all_index]&(year[all_index]==summary['year'])).any():raise ValueError('Frozen training changed or held-year leakage')
        r_index=all_index[keep[all_index]]
        if not valid_epochs(r_index,nsc,time,settings['production']):raise ValueError('Recorded r-existing source has no valid original r subset')
        models[slot]=[]
        for use in (r_index,all_index):
            model,_=fit_with_calibration(xy[use],time[use],sigma[use],config['reference_epoch'],[solution[k] for k in keys[use]])
            models[slot].append((model,use))
        difference=float(np.max(abs(models[slot][1][0].coeff[1]-row['motion'])));maximum=max(maximum,difference)
        np.testing.assert_allclose(models[slot][1][0].coeff[1],row['motion'],rtol=1e-9,atol=1e-8)
    slots=sorted(models);local={slot:j for j,slot in enumerate(slots)}
    targets=defaultdict(dict)
    for row in saved:
        if row['slot'] in models:
            index=lookup.get_indexer([row['held_detection_id']])[0]
            if index<0:raise ValueError('Frozen held detection missing')
            targets[data.exposure_id.iloc[index]][row['slot']]=(index,row)
    coord=xy-np.array([solution[k][0] if k in solution else np.zeros(2) for k in keys]);qualified=nsc&physical_candidates(data)&calibration_support(keys,solution)&(year==summary['year'])
    common=[];exits=defaultdict(int)
    for exposure,frame in data.loc[qualified].groupby('exposure_id',sort=False):
        if exposure not in targets:continue
        ix=frame.index.to_numpy();midpoint=np.median(time[ix]);tree=cKDTree(coord[ix])
        predictions=np.array([models[slot][0][0].predict(midpoint) for slot in slots]);good,nearest,_=exclusive(tree,predictions,1000,settings['production']['competitor_margin_mas'])
        for slot,(index,saved_row) in targets[exposure].items():
            if not good[local[slot]] or ix[nearest[local[slot]]]!=index:exits['r_counterfactual_no_same_unique_held_detection']+=1;continue
            residuals=[];errors=[]
            for model,use in models[slot]:
                residuals.append(coord[index]-model.predict(time[index]))
                errors.append(np.sqrt(held_prediction_variance(model,time[use],sigma[use],[solution[k] for k in keys[use]],solution[keys[index]],sigma[index],time[index])))
            np.testing.assert_allclose(residuals[1],saved_row['residual_mas'],rtol=1e-9,atol=1e-7)
            common.append(dict(slot=slot,magnitude=training[slot]['magnitude'],band=str(data.band.iloc[index]),residual=np.asarray(residuals),error=np.asarray(errors)))
    cohorts={}
    for low in (20,21,22):
        rows=[r for r in common if low<=r['magnitude']<low+1]
        cohorts['r'+str(low)+'__all_held_bands']=summarize(rows)
        for band in sorted({r['band'] for r in rows}):cohorts['r'+str(low)+'__held_'+band]=summarize([r for r in rows if r['band']==band])
    result=dict(status='completed',year=summary['year'],fixed_r_existing_training_count=len(models),common_held_position_count=len(common),
        exits=dict(exits),maximum_replayed_allband_pm_difference_masyr=maximum,cohorts=cohorts,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Exactly frozen r-existing objects. The original r subset is an identity-preserving subset of appended all-band members. Refit r and all on training-only detections; compare the same held observation only if both predictions give its same unique match. Common-source competition universe and 1arcsec gate explicitly condition these metrics. Bootstrap resamples physical sources using each source median radial held error, not individual repeated detections. Shared reference terms propagate between train and held groups; this is conditional position precision, not absolute PM truth.')
    check_aggregate(result);save_json(work/'fixed_year_multiband_summary.json',result);save_json(work/'private_common_held.json',common)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('work',type=Path);p.add_argument('trial',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'source_trial_receipt.json',verify_directory(a.trial,('private_training.json','private_held.json','allband_blind_year_summary.json')))
    with run_stage(a.work,PRECISION,'fixed_r_existing_year','fixed-year-allband-v14',inputs=('source_trial_receipt.json',),outputs=('fixed_year_multiband_summary.json','private_common_held.json')) as go:
        if go:run(**vars(a))


if __name__=='__main__':main()
