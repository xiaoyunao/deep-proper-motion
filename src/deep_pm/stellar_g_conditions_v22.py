"""Actual-g matched colour contrasts and repeat-geometry identifiability.

Residual colour contrasts use Gaia as validation only. Repeat opportunity and
short-time raw position differences never read Gaia PM/position predictions.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .astrometry import project
from .blind_catalogue_quality import read
from .coordinate_metric import forward_jacobian
from .io import digest,save_json
from .public_summary import check_aggregate
from .stellar_g_mechanism_v22 import COEFFICIENTS,finite_json
from .usability_cluster import summarize


def describe(values):
    result=summarize(values)
    # Rows are repeated measurements, not independent physical objects.
    result.pop('median_object_interval95',None)
    return result


def pair_summary(rows):
    if not rows:return dict(n=0)
    if len(rows)<5:return dict(n=len(rows),scope='Counts only below five pairs')
    night=defaultdict(list)
    for r in rows:night[r['night']].append(r)
    med=np.array([np.median([r['observed'] for r in g],axis=0) for g in night.values()])
    rng=np.random.default_rng(220226)
    idx=rng.integers(0,len(med),size=(400,len(med)))
    return dict(n=len(rows),physical_stars=len({tuple(k) for r in rows for k in r['source_keys']}),nights=len(night),
        exposure_CCD_groups=len({r['physical_group'] for r in rows}),
        delta_color_p16_p50_p84=np.quantile([r['delta_color'] for r in rows],[.16,.5,.84]),
        observed_common_basis_red_minus_blue_mas=describe([r['observed'] for r in rows]),
        parallel_p16_p50_p84_mas=np.quantile([r['parallel'] for r in rows],[.16,.5,.84]),
        perpendicular_p16_p50_p84_mas=np.quantile([r['perpendicular'] for r in rows],[.16,.5,.84]),
        frozen_models={str(k):describe([np.asarray(r['observed'])-k*np.asarray(r['predictor']) for r in rows]) for k in COEFFICIENTS},
        equal_night_median_sensitivity95=np.quantile(np.median(med[idx],axis=1),[.025,.975],axis=0).T if len(med)>=3 else None,
        scope='Metadata-only match in actual g mag0.3/logSNR0.15/sky0.05deg, unique deep self and neighbour>=3arcsec. Gaia residuals validate contrasts; no candidate coefficient fit. Night interval is conditional sampling sensitivity.')


def run(root,output):
    output.mkdir(parents=True,exist_ok=True)
    input=root/'v2.2/stellar-g/private_stellar_g_mechanism.json';p=read(input);rows=p['measurements']
    plan=read(root/'v2.1/registration/private_coordinate_plan.json');centers={r['core']:r['graph_center'] for r in plan['selected']}
    matched=[];counts={};maximum_cancellation=0.
    for name,group in pd.DataFrame(rows).groupby('physical_group',sort=True):
        part=group.to_dict('records');pivot=np.array(part[0]['group_to_graph']);inv=np.linalg.inv(pivot)
        selected=[m for m in part if m['deep_neighbour'].get('status')=='unique_deep_self' and m['deep_neighbour'].get('neighbour_distance_mas',0)>=3000 and m['band_snr']>0]
        b=[m for m in selected if m['color']<.5];r=[m for m in selected if m['color']>=2]
        counts[name]=dict(blue=len(b),red=len(r),matched=0)
        if not b or not r:continue
        center=(float(np.median([m['ra_deg'] for m in selected])),float(np.median([m['dec_deg'] for m in selected])))
        def features(v):
            sky=project([m['ra_deg'] for m in v],[m['dec_deg'] for m in v],center)
            return np.column_stack([sky/180000,np.array([m['band_mag'] for m in v])/.3,np.log10([m['band_snr'] for m in v])/.15])
        distance,near=cKDTree(features(b)).query(features(r));used=set()
        for j in np.argsort(distance,kind='stable'):
            k=int(near[j]);red,blue=r[j],b[k]
            if k in used:continue
            position=project([red['ra_deg']],[red['dec_deg']],(blue['ra_deg'],blue['dec_deg']))[0]
            if np.linalg.norm(position)>180000 or abs(red['band_mag']-blue['band_mag'])>.3 or abs(np.log10(red['band_snr']/blue['band_snr']))>.15:continue
            used.add(k);counts[name]['matched']+=1
            raw=inv@(np.array(red['raw_graph'])-blue['raw_graph']);observed=inv@(np.array(red['c0_graph'])-blue['c0_graph'])
            maximum_cancellation=max(maximum_cancellation,float(np.max(abs(observed-raw))))
            predictors=[]
            for m in (red,blue):
                a=forward_jacobian(m['ra_deg'],m['dec_deg'],centers[m['source_key'][0]])
                predictors.append(a@np.asarray(m['native_geometry'])*(m['color']-1.2))
            predictor=inv@(predictors[0]-predictors[1]);direction=predictor/np.linalg.norm(predictor)
            matched.append(dict(physical_group=name,night=red['utc_night'],source_keys=[red['source_key'],blue['source_key']],
                delta_color=red['color']-blue['color'],observed=observed,predictor=predictor,
                parallel=float(observed@direction),perpendicular=float(observed@np.array([-direction[1],direction[0]])),
                band_mag=[red['band_mag'],blue['band_mag']],band_snr=[red['band_snr'],blue['band_snr']],
                train_identities=red['split']<3 and blue['split']<3))
    repeats=[]
    exposures=defaultdict(dict)
    for m in rows:exposures[(m['utc_night'],m['calibration_group'])][tuple(m['source_key'])]=m
    for night in sorted({k[0] for k in exposures}):
        names=sorted([k for k in exposures if k[0]==night])
        for a,left in enumerate(names):
            for right in names[a+1:]:
                common=sorted(exposures[left].keys()&exposures[right].keys())
                if not common:
                    repeats.append(dict(night=night,physical_exposures=[left[1],right[1]],common_stars=0));continue
                values=[];predictors=[];elapsed=[];train=0;colors=[];control=0;noise=[];detector_pairs=[]
                for key in common:
                    l,r=exposures[left][key],exposures[right][key];center=centers[key[0]]
                    if center!=centers[r['source_key'][0]]:raise ValueError('Repeat coordinate chart changed')
                    actual=project([r['ra_deg'],l['ra_deg']],[r['dec_deg'],l['dec_deg']],center)
                    # Direct observed positions: no raw_graph Gaia subtraction.
                    values.append(actual[0]-actual[1]);colors.append(l['color']);elapsed.append(abs(r['time']-l['time']))
                    ah=[]
                    for m in (r,l):ah.append(forward_jacobian(m['ra_deg'],m['dec_deg'],center)@np.asarray(m['native_geometry'])*(m['color']-1.2))
                    predictors.append(ah[0]-ah[1]);train+=l['split']<3
                    noise.append(np.asarray(l['sigma_graph'])+np.asarray(r['sigma_graph']));detector_pairs.append((l['detector_id'],r['detector_id']))
                    control+=all(m['band_snr']>=20 and m['deep_neighbour'].get('neighbour_distance_mas',0)>=3000 for m in (l,r))
                predictor=np.asarray(predictors);centered=predictor-predictor.mean(axis=0)
                fisher=0.
                for detector in sorted(set(detector_pairs)):
                    ix=np.array([d==detector for d in detector_pairs]);x=predictor[ix];w=np.linalg.inv(np.asarray(noise)[ix])
                    mean=np.linalg.solve(w.sum(axis=0),np.einsum('nij,nj->i',w,x))
                    delta=x-mean;fisher+=float(np.einsum('ni,nij,nj->',delta,w,delta))
                repeats.append(dict(night=night,physical_exposures=[left[1],right[1]],common_stars=len(common),
                    training_identity_stars=int(train),gSNR20_deep3arcsec_stars=int(control),
                    color_p16_p50_p84=np.quantile(colors,[.16,.5,.84]),
                    predictor_after_exposure_intercept_rms=float(np.sqrt(np.mean(centered**2))),
                    conditional_coefficient_Fisher=float(fisher),conditional_coefficient_sigma=1/np.sqrt(fisher) if fisher>1e-30 else None,
                    expected_signal_to_noise_at_frozen28=28.32*np.sqrt(fisher),detector_pair_intercepts=len(set(detector_pairs)),
                    maximum_interval_hours=float(max(elapsed)*365.25*24),
                    maximum_motion_sensitivity_mas_at10000masyr=float(max(elapsed)*10000)))
    supported=[r for r in repeats if r['common_stars']>=20 and r['gSNR20_deep3arcsec_stars']>=20 and r['predictor_after_exposure_intercept_rms']>=.03]
    nights={r['night'] for r in supported}
    # This is a support test, not a quality selection by measured coefficient.
    decision='candidate_not_fit_insufficient_independent_repeat_nights' if len(nights)<3 else 'repeat_support_requires_frozen_night_and_identity_split_before_one_candidate'
    save_json(output/'private_actual_g_pairs.json',finite_json(dict(validation_pairs=matched,repeated_geometry=repeats)))
    report=dict(input_sha256=digest(input),matched_actual_g_colour=pair_summary(matched),
        training_identity_validation_only=pair_summary([r for r in matched if r['train_identities']]),
        maximum_same_group_C0_cancellation_mas=maximum_cancellation,
        potential_colour_pair_groups=len(counts),groups_with_matched_pairs=sum(r['matched']>0 for r in counts.values()),
        repeated_exposure_support=[r if r['common_stars']>=5 else {k:r[k] for k in ('night','physical_exposures','common_stars')} for r in repeats],qualified_repeat_nights=len(nights),decision=decision,
        slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Matched actual-g/deep-neighbour colour contrasts are Gaia validation. Raw same-star within-night geometry support uses observed coordinates only, with exposure intercept nuisance and short-motion sensitivity. No new physical coefficient trained or applied. Additional metadata-selected cores limited to2 if needed; old historical coefficients remain frozen.')
    check_aggregate(report);save_json(output/'v22_stellar_g_conditions_summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(**vars(a))
