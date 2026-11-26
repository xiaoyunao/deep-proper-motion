"""One relative-colour coefficient with free exposure/CCD-pair intercepts.

Observed repeat positions only. Independent identities AND nights validate the
coefficient. This cannot identify an absolute stellar reference-frame intercept.
"""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .blind_catalogue_quality import read
from .io import save_json,digest
from .public_summary import check_aggregate
from .coordinate_metric import forward_jacobian
from .continuous_reference import verify_directory
from .stellar_g_mechanism_v22 import finite_json
from .usability_cluster import summarize


def profile_fit(y,x,cov,groups,weights=None):
    """Full2D GLS, profile two nuisance intercepts per physical CCD pair."""
    unique,g=np.unique(groups,return_inverse=True);w=np.linalg.inv(cov)
    if weights is not None:w=w*np.asarray(weights)[:,None,None]
    a=np.zeros((len(unique),2,2));u=np.zeros((len(unique),2));v=np.zeros((len(unique),2))
    np.add.at(a,g,w);np.add.at(u,g,np.einsum('nij,nj->ni',w,x));np.add.at(v,g,np.einsum('nij,nj->ni',w,y))
    # Empty groups have zero Bayesian weight only if the caller corrupts it;
    # true weights are strictly positive exponential draws.
    inv=np.linalg.inv(a);mx=np.einsum('gij,gj->gi',inv,u);my=np.einsum('gij,gj->gi',inv,v)
    xr=x-mx[g];yr=y-my[g]
    information=float(np.einsum('ni,nij,nj->',xr,w,xr));rhs=float(np.einsum('ni,nij,nj->',xr,w,yr))
    if not np.isfinite(information) or information<=1e-12:raise ValueError('Colour coefficient not identifiable after physical intercepts')
    beta=rhs/information;intercept=my-beta*mx;residual=y-beta*x-intercept[g]
    return beta,information,residual,intercept,unique


def robust_fit(y,x,cov,groups):
    weight=np.ones(len(y));inverse=np.linalg.inv(cov)
    for step in range(7):
        result=profile_fit(y,x,cov,groups,weight)
        if step<6:
            d2=np.einsum('ni,nij,nj->n',result[2],inverse,result[2]);weight=np.minimum(1,6/(4+d2))
    return result,weight


def describe(y):
    r=summarize(y);r.pop('median_object_interval95',None);return r


def run(root,output):
    output.mkdir(parents=True,exist_ok=True);input=root/'v2.2/stellar-g-extra-repeat/private_observed_repeat_pairs.json';raw=read(input);pairs=raw['pairs']
    selection=read(root/'v2.2/stellar-g-metadata-final-single/private_additional_core_plan.json')
    native={};state_receipts={}
    requested={r['source_key'] for r in pairs}
    for entry in selection['selected']:
        folder=Path(entry['fit']);verify_directory(folder,('private_standard_motion.json.gz',))
        state_receipts[entry['core']]=digest(folder/'private_standard_motion.json.gz')
        for row in read(folder/'private_standard_motion.json.gz'):
            key=entry['core']+':'+sha256('\0'.join(sorted(row['detection_ids'])).encode()).hexdigest()
            if key in requested:native[key]=np.linalg.inv(forward_jacobian(*row['position_deg'],entry['center']))
    if set(native)!=requested:raise ValueError('Repeat physical identities lost their frozen source basis')
    training=[r for r in pairs if not r['held_star'] and not r['held_night']]
    independent=[r for r in pairs if r['held_star'] and r['held_night']]
    def supported(rows):
        count=Counter(r['group'] for r in rows);return [r for r in rows if count[r['group']]>=5]
    training=supported(training);independent=supported(independent)
    register=dict(input_sha256=digest(input),train_pairs=len(training),train_stars=len({r['source_key'] for r in training}),train_nights=len({r['night'] for r in training}),
        validation_pairs=len(independent),validation_stars=len({r['source_key'] for r in independent}),validation_nights=len({r['night'] for r in independent}),
        split='Independent star hashmod3 and observing-night hashmod4; train uses neither held, final validation uses BOTH held. No validation position/PM read in training fit. Groups need >=5 stars.',
        predeclared_gate='At least3 train and3 validation nights; >=50 train and50 independent validation stars; conditional beta SE<=10mas/(mag tanZ). One two-axis common coefficient, pivotg-i1.2, no additional colour/CCD parameters.')
    save_json(output/'v22_repeat_coefficient_registration.json',register)
    if min(register['train_nights'],register['validation_nights'])<3 or min(register['train_stars'],register['validation_stars'])<50:
        save_json(output/'v22_stellar_g_coefficient_summary.json',dict(registration=register,decision='not_fit_insufficient_independent_support'));return
    y=np.array([r['delta_observed_common_mas'] for r in training]);h=np.array([r['delta_geometry'] for r in training]);c=np.array([r['color'] for r in training]);error=np.array([r['color_error'] for r in training]);x=(c-1.2)[:,None]*h
    cov=np.array([r['independent_common_covariance'] for r in training]);group=np.array([r['group'] for r in training]);result,weight=robust_fit(y,x,cov,group);beta,information,residual,intercept,unique=result
    if 1/np.sqrt(information)>10:
        save_json(output/'v22_stellar_g_coefficient_summary.json',dict(registration=register,decision='not_adopted_weak_geometric_information',conditional_SE=float(1/np.sqrt(information))));return
    # One colour deviate per physical star, shared over ALL its repeat nights.
    # Coefficient and group intercept are recomputed in the same draw. Physical
    # reference perturbations stay synchronous within each old reference graph;
    # here they are exactly absorbed by the profiled common-basis intercept.
    stars,s=np.unique([r['source_key'] for r in training],return_inverse=True);nights,n=np.unique([r['night'] for r in training],return_inverse=True)
    sigma=np.array([error[np.flatnonzero(s==i)[0]] for i in range(len(stars))]);rng=np.random.default_rng(220226)
    samples=[];maximum_reference_beta_change=0.
    for draw in range(192):
        color_delta=rng.normal(size=len(stars))*sigma
        xx=(c+color_delta[s]-1.2)[:,None]*h
        multiplier=rng.exponential(size=len(stars))[s]*rng.exponential(size=len(nights))[n]*weight
        fitted=profile_fit(y,xx,cov,group,multiplier)
        if draw<48:
            perturb=np.array([r['group_reference_difference_draws'][draw] for r in training])
            coupled=profile_fit(y-perturb,xx,cov,group,multiplier)
            maximum_reference_beta_change=max(maximum_reference_beta_change,abs(fitted[0]-coupled[0]))
        samples.append(dict(coefficient=fitted[0],group_intercept=fitted[3],color_delta=color_delta))
    if maximum_reference_beta_change>1e-6:raise ValueError('C0 common-mode did not cancel against the registered physical group intercept')
    # Validate fixed coefficient on wholly new stars/nights. Raw/C0 zero and
    # intercept-profiled relative scatter both reported; neither is hidden.
    validations={}
    for name,rows in [('independent_identity_and_night',independent),('training',training)]:
        yy=np.array([r['delta_observed_common_mas'] for r in rows]);hh=np.array([r['delta_geometry'] for r in rows]);cc=np.array([r['color'] for r in rows]);xx=(cc-1.2)[:,None]*hh
        vv=np.array([r['independent_common_covariance'] for r in rows]);gg=np.array([r['group'] for r in rows]);labels,inverse=np.unique(gg,return_inverse=True)
        transformed={}
        for tag,k in [('uncorrected',0.),('frozen_minus28p32',-28.32),('frozen_minus23p39',-23.39),('one_stellar_candidate',beta)]:
            rr=yy-k*xx;w=np.linalg.inv(vv);a=np.zeros((len(labels),2,2));b=np.zeros((len(labels),2));np.add.at(a,inverse,w);np.add.at(b,inverse,np.einsum('nij,nj->ni',w,rr));zero=np.einsum('gij,gj->gi',np.linalg.inv(a),b)
            b=np.array([native[r['source_key']] for r in rows])
            transformed[tag]=dict(raw_C0_difference_native_mas=describe(np.einsum('nij,nj->ni',b,rr)),relative_after_nuisance_intercept_native_mas=describe(np.einsum('nij,nj->ni',b,rr-zero[inverse])))
        validations[name]=dict(pairs=len(rows),nights=len(set(r['night'] for r in rows)),stars=len(set(r['source_key'] for r in rows)),models=transformed,
            color_p16_p50_p84=np.quantile(cc,[.16,.5,.84]),gSNR_p16_p50_p84=np.quantile(np.array([r['snr'] for r in rows]),[.16,.5,.84]),
            maximum_motion_bound_mas_at10000=float(max(r['motion_sensitivity_mas_at10000'] for r in rows)))
    values=np.array([r['coefficient'] for r in samples]);summary=dict(registration=register,frozen_source_state_sha256=state_receipts,coefficient_mas_per_mag_tanz=float(beta),conditional_centroid_SE=float(1/np.sqrt(information)),
        crossed_star_night_and_shared_colour_sensitivity95=np.quantile(values,[.025,.975]),crossed_sensitivity_SE=float(np.std(values,ddof=1)),
        maximum_common_reference_coefficient_change=float(maximum_reference_beta_change),train_physical_pair_intercepts=len(unique),validation=validations,
        uncertainty_scope='192 crossed star/night Bayesian reweights with one colour error deviate per physical star shared over its entire trajectory; coefficients and intercepts recomputed jointly. Sampling sensitivity, not a full posterior or absolute offset certification. Repeat-group C0 draws cancel numerically after intercept profiling; no per-source sqrt reduction.',
        decision='Relative-colour candidate only; no main-production correction. Absolute stellar/QSO intercept and cross-source-type reference transfer remain independent acceptance requirements.',
        slurm_job=os.environ['SLURM_JOB_ID'],scope='One physical two-axis zenith-vector colour derivative trained on observed repeats without Gaia, independently tested by star AND night. Free pair intercepts prevent reference zeros from masquerading as colour. Their freedom also means absolute stellar zero is not determined. No new candidate beyond this coefficient.')
    check_aggregate(summary);save_json(output/'v22_stellar_g_coefficient_summary.json',summary)
    save_json(output/'private_joint_coefficient_draws.json',finite_json(dict(point=beta,group_labels=unique,group_intercepts=intercept,samples=samples,source_labels=stars,robust_weights=weight)))
    print('G_RELATIVE_COEFFICIENT_CANDIDATE',beta,summary['crossed_star_night_and_shared_colour_sensitivity95'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.root,a.output)
