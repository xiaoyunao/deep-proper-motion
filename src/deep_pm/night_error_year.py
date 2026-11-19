"""Independent held-year test of frozen galaxy-learned night-error parameters."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .allband_qso import physical_candidates
from .condition_holdout import held_prediction_variance
from .continuous_instruments import chip_receipts,instrument_by_exposure
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .correlated_motion import fit_correlated_motion
from .fixed_year_multiband import summarize
from .io import save_json,digest
from .nsc_information import night_labels
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate
from .year_opportunities import utc_year


def prediction_variance(fit,times,offsets,held_offset,held_time,held_sigma,train_night,held_night,train_common,held_common):
    dt=held_time-fit.model.epoch;h=np.array([[1,dt,0,0],[0,0,1,dt]])
    independent=np.einsum('ap,pq,aq->a',h,fit.independent_covariance+fit.night_covariance,h)
    samples=np.stack([o[1] for o in offsets],axis=1);reference=np.einsum('ap,pna,dna->da',h,fit.response,samples)
    cross=np.zeros(2)
    same=np.asarray(train_night)==held_night
    for axis in range(2):
        influence=np.einsum('p,pna->na',h[axis],fit.response)[:,axis]
        cross[axis]=np.sum(influence[same]*train_common[same,axis])*held_common[axis]
    variance=held_sigma**2+held_common**2+independent-2*cross+np.var(reference-held_offset[1],axis=0,ddof=1)
    if (variance<-1e-5).any():raise ValueError('Negative predictive variance')
    return np.maximum(variance,0.)


def run(work,trial,error_trial):
    info=json.loads((trial/'allband_blind_year_summary.json').read_text());learned=json.loads((error_trial/'multiband_error_trial_summary.json').read_text())
    if info['status']!='completed':save_json(work/'night_error_year_summary.json',dict(status=info['status']));save_json(work/'private_night_error_fits.json',[]);return
    source=(trial/'source_receipt.json').resolve().parent;graph=(trial/'reference_receipt.json').resolve().parent/'private_c0.npz';center=reference_frame(graph);config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);keys=calibration_groups(data);lookup=pd.Index(data.detection_id)
    solution,gauges=read_offsets(graph,with_gauge=True);solution={k:v for k,v in solution.items() if gauges[k]==info['reference_gauge']}
    chips,complete=chip_receipts((source/'run_manifest.json').resolve());instrument=data.exposure_id.map(instrument_by_exposure(chips)).fillna('unknown');instrument.loc[data.survey.eq('SDSS')]='SDSS_imager'
    family=(data.survey+':'+instrument+':'+data.band.astype(str)).to_numpy();night=night_labels(data)
    raw=data[['sigma_east_mas','sigma_north_mas']].to_numpy(float);independent=sigma.copy();extra=np.zeros_like(sigma)
    for label,model in learned['error_models'].items():
        if not model['identifiable']:continue
        p=np.asarray(model['parameters_by_axis']);mask=family==label
        independent[mask]=np.sqrt((raw[mask]*p[:,0])**2+p[:,1]**2);extra[mask]=p[:,2]
    labels=dict(E_within_band=np.char.add(family.astype(str),night.astype(str)),E_shared_instrument=np.char.add((data.survey+':'+instrument).to_numpy(str),night.astype(str)))
    train=json.loads((trial/'private_training.json').read_text());held=json.loads((trial/'private_held.json').read_text());tests={}
    for row in held:tests.setdefault(row['slot'],[]).append(row)
    ns=data.survey.eq('NSC').to_numpy();year=utc_year(data.observation_time.to_numpy());coord=xy-np.array([solution[k][0] if k in solution else np.zeros(2) for k in keys]);records=[];private=[];boundary=0
    for slot,rows in tests.items():
        row=train[slot];index=lookup.get_indexer(row['detection_ids'])
        if (index<0).any() or (ns[index]&(year[index]==info['year'])).any():raise ValueError('Frozen members missing or held-year leakage')
        offsets=[solution[k] for k in keys[index]];models={name:fit_correlated_motion(coord[index],time[index],independent[index],label[index],extra[index],config['reference_epoch'],offsets) for name,label in labels.items()}
        private.append(dict(slot=slot,models={name:dict(state=fit.model.coeff.T.ravel(),covariance=fit.covariance,independent_covariance=fit.independent_covariance,night_covariance=fit.night_covariance,reference_covariance=fit.reference_covariance) for name,fit in models.items()}))
        for saved in rows:
            k=lookup.get_indexer([saved['held_detection_id']])[0]
            if k<0:raise ValueError('Missing frozen held detection')
            boundary+=bool(np.any(night[index]==night[k]));residual=[saved['residual_mas']];errors=[saved['error_mas']]
            for name,fit in models.items():
                residual.append(coord[k]-fit.model.predict(time[k]));v=prediction_variance(fit,time[index],offsets,solution[keys[k]],time[k],independent[k],labels[name][index],labels[name][k],extra[index],extra[k]);errors.append(np.sqrt(v))
            records.append(dict(slot=slot,kind=row['kind'],magnitude=row['magnitude'],residual=np.asarray(residual),error=np.asarray(errors)))
    names=('W0',)+tuple(labels);cohorts={}
    for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
        for low in (21,22):
            part=[r for r in records if r['kind']==kind and low<=r['magnitude']<low+1];cohorts[kind+'__r'+str(low)]={}
            for j,name in enumerate(names[1:],1):
                pair=[dict(r,residual=r['residual'][[0,j]],error=r['error'][[0,j]]) for r in part]
                cohorts[kind+'__r'+str(low)][name+'_vs_W0']=summarize(pair,names=('W0',name))
    result=dict(status='completed',year=info['year'],cohorts=cohorts,held_positions_sharing_utc18_night_with_training=boundary,parameters_sha256=digest(error_trial/'multiband_error_trial_summary.json'),error_models=learned['error_models'],frozen_QSO_point_and_error_comparisons=learned['fixed_qso'],chip_receipts_complete=complete,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Exactly frozen training members and same C0-gated held positions. Error parameters frozen from v1.4 disjoint galaxy training before these stellar tests; QSO point metrics retained separately and never train parameters. Explicit independent/night/reference full4x4 components; held-night covariance subtracts shared train-held night terms at calendar-year edges. Within-band versus fully shared-instrument nights bracket unknown cross-band correlation. Parameter-learning and unmodeled spatial systematics remain unqualified; no automatic adoption.')
    check_aggregate(result);save_json(work/'night_error_year_summary.json',result);save_json(work/'private_night_error_fits.json',private)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','trial','error_trial'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'target_receipt.json',verify_directory(a.trial,('private_training.json','private_held.json','allband_blind_year_summary.json')));bind(a.work,'error_receipt.json',verify_directory(a.error_trial,('multiband_error_trial_summary.json',)))
    with run_stage(a.work,PRECISION,'frozen_error_year','night-error-year-v15',inputs=('target_receipt.json','error_receipt.json'),outputs=('night_error_year_summary.json','private_night_error_fits.json')) as go:
        if go:run(a.work,a.trial,a.error_trial)


if __name__=='__main__':main()
