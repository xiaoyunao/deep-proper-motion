"""Allocate frozen formal covariance; this is not empirical error attribution."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .continuous_reference import FILES,PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .io import save_json
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate
from .standard_motion import calibrated_covariance,icrs_motion

TERMS=('SDSS_published_error','NSC_published_error','per_detection_floor','reference_bootstrap')


def allocate(model,time,sigma,native_sigma,nsc,offsets,center):
    total,independent,reference,response=calibrated_covariance(model,time,sigma,offsets)
    native=np.asarray(native_sigma,float)**2
    floor=np.asarray(sigma,float)**2-native
    if np.any(floor < -1e-7):raise ValueError('Native errors exceed fitted error model')
    floor=np.maximum(floor,0)
    # Conditional IRLS covariance assumes effective variance sigma^2/weight.
    # H diag(sigma^2/weight) H.T reproduces the frozen inverse normal matrix.
    def term(variance):
        return np.einsum('pna,na,qna->pq',response,variance/model.weight[:,None],response)
    matrices={TERMS[0]:term(native*(~nsc[:,None])),TERMS[1]:term(native*nsc[:,None]),
              TERMS[2]:term(floor),TERMS[3]:reference}
    np.testing.assert_allclose(sum(matrices[k] for k in TERMS[:3]),independent,rtol=1e-9,atol=1e-7)
    np.testing.assert_allclose(sum(matrices.values()),total,rtol=1e-9,atol=1e-7)
    jac=icrs_motion(model.coeff[0],model.coeff[1],center)['state_jacobian']
    covariances={key:jac@value@jac.T for key,value in matrices.items()}
    variance=np.array([np.diag(covariances[k])[[1,3]] for k in TERMS])
    total_variance=variance.sum(axis=0)
    if np.any(total_variance<=0):raise ValueError('Nonpositive proper-motion covariance')
    return dict(variance_masyr2=variance,variance_fraction=variance/total_variance,
                total_sigma_masyr=np.sqrt(total_variance))


def summarize(records):
    result={}
    for low in (20,21,22):
        rows=[r for r in records if low<=r['magnitude']<low+1]
        subsets={'all':rows,'one_sdss_run':[r for r in rows if r['sdss_runs']==1],
                 'multiple_sdss_runs':[r for r in rows if r['sdss_runs']>=2]}
        for subset,selected in subsets.items():
            out=dict(n=len(selected))
            if len(selected)>=5:
                fraction=np.array([r['variance_fraction'] for r in selected])
                out.update(fraction_p16_p50_p84=np.quantile(fraction,[.16,.5,.84],axis=0),
                    total_sigma_p16_p50_p84_masyr=np.quantile([r['total_sigma_masyr'] for r in selected],[.16,.5,.84],axis=0),
                    sdss_published_term_above_half_count=(fraction[:,0]>.5).sum(axis=0))
            result[f'r{low}_{low+1}__{subset}']=out
    return result


def run(work,source,conditions,shared):
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    graph=shared/'private_c0.npz';center=reference_frame(graph);offsets=read_offsets(graph)
    data,xy,time,sigma,_=quality_detections(source,center,config,settings['production'])
    ids=pd.Index(data.detection_id)
    if not ids.is_unique:raise ValueError('Detection identity must be unique')
    native=data[['sigma_east_mas','sigma_north_mas']].to_numpy(float)
    keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
    records=[];maximum=0.
    for row in json.loads((conditions/'private_conditions.json').read_text()):
        ix=ids.get_indexer(row['detection_ids'])
        if np.any(ix<0):raise ValueError('A frozen detection is missing')
        selected=[offsets[keys[i]] for i in ix]
        model,error=fit_with_calibration(xy[ix],time[ix],sigma[ix],config['reference_epoch'],selected)
        expected=np.asarray(row['motion']['all_SDSS'])
        maximum=max(maximum,float(np.max(abs(model.coeff[1]-expected))))
        np.testing.assert_allclose(model.coeff[1],expected,rtol=0,atol=1e-8)
        np.testing.assert_allclose(error,row['error']['all_SDSS'],rtol=1e-10,atol=1e-8)
        budget=allocate(model,time[ix],sigma[ix],native[ix],nsc[ix],selected,center)
        records.append(dict(qso_index=row['qso_index'],magnitude=row['magnitude'],sdss_runs=row['sdss_runs'],**budget))
    save_json(work/'private_budget.json',records)
    public=dict(control_count=len(records),maximum_replayed_motion_difference_masyr=maximum,
        covariance_closure_verified=True,terms=TERMS,component_order=['ICRS_east','ICRS_north'],cohorts=summarize(records),
        interpretation='Allocation of the frozen conditional IRLS formal covariance, including native catalog calibration terms treated as independent. Not measured error causality, not a forecast of achievable PSF gains. Reference terms remain separate; original correlations in native errors remain unknown.',
        selection='Exactly saved detection identities and order of the frozen control experiment; no new associations, cuts or fit parameters.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(public);save_json(work/'motion_budget_summary.json',public)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','source','conditions','shared'):p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    for alias,path,names in [('source',a.source,FILES),('conditions',a.conditions,('private_conditions.json','conditions_summary.json')),
                             ('shared',a.shared,('private_c0.npz',))]:
        bind(a.work,alias+'_receipt.json',verify_directory(path,names))
    bind(a.work,'spatial_config.toml',SPATIAL)
    with run_stage(a.work,PRECISION,'frozen_controls','motion-error-budget-v13',
        inputs=('source_receipt.json','conditions_receipt.json','shared_receipt.json','spatial_config.toml'),
        outputs=('private_budget.json','motion_budget_summary.json')) as go:
        if go:run(a.work,a.source,a.conditions,a.shared)


if __name__=='__main__':main()
