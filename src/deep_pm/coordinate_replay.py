"""Fixed-source B1/B2 geometry replay; old C0 and all physical members frozen."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .astrometry import basis
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets,verify_directory
from .coordinate_inventory import angle
from .coordinate_metric import (native_covariance,projected_covariance,old_influence,
    covariance_of_estimator,fit_vector_motion,reference_covariance)
from .exposure_catalogues import inside
from .io import digest,save_json,json_default
from .nsc_comparison import comparison
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .public_summary import check_aggregate
from .standard_motion import icrs_motion


@lru_cache(maxsize=3)
def offsets(path):return read_offsets(Path(path))


def quantile(values):return np.quantile(values,[.16,.5,.84],axis=0) if len(values) else None


def one(task):
    row,destination=task;dest=Path(destination)/row['core'];dest.mkdir(parents=True,exist_ok=True)
    source=Path(row['source']);folder=Path(row['fit']);graph=Path(row['graph']);center=row['graph_center']
    ticket=dict(source_sha256=digest(source/'run_manifest.json'),fit_sha256=digest(folder/'run_manifest.json'),
        graph_sha256=digest(graph),implementation_sha256=digest(Path(__file__)),metric_sha256=digest(Path(__file__).with_name('coordinate_metric.py')))
    result=dest/'v21_coordinate_core_summary.json'
    if result.exists():
        old=read(result)
        if old['ticket']!=ticket:raise ValueError('Coordinate replay cache has changed inputs')
        return old
    start=monotonic();verify_directory(source,('detections.csv','spectra_union.csv'))
    verify_directory(folder,('private_standard_motion.json.gz','private_blind_qso.json'))
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production'])
    groups=calibration_groups(data);lookup=pd.Index(data.detection_id)
    solution=offsets(str(graph));cp=projected_covariance(data.ra_deg,data.dec_deg,center,native_covariance(sigma))
    frozen=read(folder/'private_standard_motion.json.gz')
    if data.detection_id.duplicated().any():raise ValueError('Duplicate physical input detection')
    qso={r['source_slot']:r for r in read(folder/'private_blind_qso.json')}
    spectra=pd.read_csv(source/'spectra_union.csv');spectra=spectra.loc[inside(spectra.ra,spectra.dec,row['spec'])].reset_index(drop=True)
    private=[];angles=[];ratios=[];changes=[];pass_new=[];qrows=[];maximum_common=0.;maximum_old_ind=0.
    for slot,old in enumerate(frozen):
        ix=lookup.get_indexer(old['detection_ids'])
        if (ix<0).any():raise ValueError('A frozen member is missing')
        t=time[ix];s=sigma[ix];h=old_influence(old,t,s)
        j=np.asarray(old['state_jacobian']);weight=np.asarray(old['robust_weight'])
        independent_old=covariance_of_estimator(h,native_covariance(s),weight)
        independent_inflated=covariance_of_estimator(h,cp[ix],weight)
        independent_sandwich=covariance_of_estimator(h,cp[ix])
        delta=np.max(abs(independent_old-np.asarray(old['independent_tangent_covariance'])))
        maximum_old_ind=max(maximum_old_ind,float(delta))
        np.testing.assert_allclose(independent_old,old['independent_tangent_covariance'],rtol=2.e-8,atol=2.e-6)
        draws=np.stack([solution[g][1] for g in groups[ix]],axis=1)
        common,refdraws=reference_covariance(h,draws)
        np.testing.assert_allclose(common,old['reference_tangent_covariance'],rtol=2.e-8,atol=2.e-6)
        maximum_common=max(maximum_common,float(np.max(abs(common-np.asarray(old['reference_tangent_covariance'])))))
        # Keep the original saved common term numerically identical in B1.
        common=np.asarray(old['reference_tangent_covariance'])
        b1=j@(independent_inflated+common)@j.T
        sandwich=j@(independent_sandwich+common)@j.T
        b1error=np.sqrt(np.diag(b1)[[1,3]])
        corrected=xy[ix]-np.asarray([solution[g][0] for g in groups[ix]])
        model=fit_vector_motion(corrected,t,cp[ix],old['reference_epoch'])
        ref2,draw2=reference_covariance(model.influence,draws)
        b2=icrs_motion(model.coeff[0],model.coeff[1],center,model.cov+ref2)
        distance=np.sqrt(np.einsum('ni,nij,nj->n',model.residual,np.linalg.inv(cp[ix]),model.residual))
        passed=bool(np.max(b2['pm_error_masyr'])<=settings['production']['max_axis_pm_error_masyr'] and np.quantile(distance,.68)<=3)
        theta=float(angle(*old['position_deg'],center));angles.append(theta)
        ratios.append(b1error/old['pm_error_masyr']);changes.append(b2['pm_icrs_masyr']-old['pm_icrs_masyr']);pass_new.append(passed)
        entry=dict(source_slot=slot,detection_ids=old['detection_ids'],kind=old['kind'],magnitude=old['magnitude'],
            position_deg=old['position_deg'],theta_deg=theta,old_motion=old['pm_icrs_masyr'],old_error=old['pm_error_masyr'],
            b1_covariance=b1,b1_error=b1error,b1_sandwich_covariance=sandwich,
            b1_independent_graph_covariance=independent_inflated,b1_sandwich_graph_covariance=independent_sandwich,
            frozen_reference_graph_covariance=common,b1_reference_state_draws=refdraws,
            b2=b2,b2_tangent_state=model.coeff.T.ravel(),b2_independent_graph_covariance=model.cov,
            b2_sandwich_graph_covariance=covariance_of_estimator(model.influence,cp[ix]),
            b2_reference_graph_covariance=ref2,b2_reference_state_draws=draw2,b2_weight=model.weight,
            b2_mahalanobis_p68=np.quantile(distance,.68),b2_icrs_quality_pass=passed,
            b2_weight_change_p16_p50_p84=quantile(model.weight-weight))
        private.append(entry)
        if slot in qso:
            target=qso[slot];k=target['qso_local_index'];r=spectra.iloc[k]
            magnitude=float(r.reference_r)
            if not 20<=magnitude<23:continue
            values=dict(old=(old['pm_icrs_masyr'],old['pm_error_masyr']),
                B1=(old['pm_icrs_masyr'],b1error),B1_sandwich=(old['pm_icrs_masyr'],np.sqrt(np.diag(sandwich)[[1,3]])),
                B2=(b2['pm_icrs_masyr'],b2['pm_error_masyr']))
            values.update({m:v for m,v in target['values'].items() if m in ('NSC','Xiao')})
            qrows.append(dict(identity=(str(r.origin),float(r.ra),float(r.dec)),magnitude=magnitude,core=row['core'],window=row['window'],
                graph_sha256=ticket['graph_sha256'],theta_deg=theta,kind=old['kind'],values=values,
                exposures=old['exposures'],nsc_span=old['information'].get('nsc_span_years'),
                fixed_source_slot=slot,quality_pass_B2=passed))
    with gzip.open(dest/'private_coordinate_states.json.gz','wt') as stream:json.dump(private,stream,default=json_default,allow_nan=False)
    save_json(dest/'private_coordinate_qso.json',qrows)
    statistics={}
    options=dict(min_publish_count=3,tail_masyr=5,bootstrap_draws=200,bootstrap_draw_chunk=8,seed=210526,bootstrap_absolute_zero=True)
    for low in (21,22):
        sample=[{m:r['values'][m] for m in ('old','B1','B1_sandwich','B2')} for r in qrows if low<=r['magnitude']<low+1]
        statistics[str(low)]=comparison(sample,options,(('B2','old'),))
    report=dict(ticket=ticket,core=row['core'],window=row['window'],graph_center_deg=center,
        fixed_sources=len(frozen),fixed_qso=len(qrows),theta_p16_p50_p84_deg=quantile(angles),
        B1_total_error_ratio_p16_p50_p84=quantile(ratios),B2_pm_change_p16_p50_p84_masyr=quantile(changes),
        B2_passed_frozen_icrs_quality=sum(pass_new),maximum_original_independent_covariance_replay_difference=maximum_old_ind,
        maximum_reference_covariance_replay_difference=maximum_common,
        B1_fixed_points_members_and_reference=True,statistics=statistics,
        elapsed_seconds=monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        slurm_job=os.environ['SLURM_JOB_ID'],scope='Same frozen source slots and detection identities. B1 independent errors only; frozen old C0 common term unchanged. B2 complete2D source weighting conditional on same old C0, no reference re-solve. Quality results annotate rather than remove fixed objects. Raw source-origin core cohort, not a new globally owned production catalogue.')
    check_aggregate(report);save_json(result,report)
    print('COORDINATE_CORE_DONE',row['core'],len(frozen),quantile(ratios),flush=True)
    return report


def run(registration,output,workers,pilot):
    plan=read(registration);selected=[r for r in plan['selected'] if not pilot or r['core'] in plan['pilot']]
    output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(workers,len(os.sched_getaffinity(0)))) as pool:
        results=list(pool.map(one,[(r,str(output)) for r in selected],chunksize=1))
    report=dict(plan_sha256=digest(registration),pilot=pilot,cores=results,slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(report);save_json(output/('v21_coordinate_pilot_summary.json' if pilot else 'v21_coordinate_replay_summary.json'),report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('registration',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute resources')
    run(**vars(a))
