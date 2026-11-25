"""Independent QSO quality products for frozen source motions and covariance."""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import os
from pathlib import Path

import healpy as hp
import numpy as np

from .blind_catalogue_quality import merge_strata,read
from .io import digest,save_json
from .production_replay_readout import load_blind
from .public_summary import check_aggregate


SEED=2205466
NORMAL_ABS_MEDIAN=.6744897501960817


def split_identity(identity):
    # Released source identity is fixed once at input. Never split by fitted PM.
    value='|'.join(map(str,identity))
    return int(sha256(('v20:'+value).encode()).hexdigest()[:16],16)%5


def condition(row):
    return '__'.join((str(int(row['magnitude'])),'one_run' if row['sdss_runs']==1 else 'multi_run',
                     'nsc_lt4' if row['nsc_span']<4 else 'nsc_ge4',
                     'leverage_ge08' if row['leverage']>=.8 else 'leverage_lt08'))


def vectors(rows):
    values=np.asarray([r['values']['blind_all'] for r in rows],float)
    return values[:,0,:],values[:,1,:]


def metrics(rows,zero=None,scale=None):
    if not rows:return dict(n=0)
    motion,error=vectors(rows)
    center=np.median(motion,axis=0)
    normalized=motion/error
    centered=(motion-center)/error
    value=dict(n=len(rows),median=center,nmad=1.4826*np.median(abs(motion-center),axis=0),
        radial_p68=np.quantile(np.linalg.norm(motion,axis=1),.68),
        fraction_abs_gt5=np.mean(abs(motion)>5,axis=0),
        normalized_nmad=1.4826*np.median(abs(normalized-np.median(normalized,axis=0)),axis=0),
        raw_1sigma=np.mean(abs(normalized)<=1,axis=0),raw_2sigma=np.mean(abs(normalized)<=2,axis=0),
        centered_1sigma=np.mean(abs(centered)<=1,axis=0),centered_2sigma=np.mean(abs(centered)<=2,axis=0),
        sdss_runs=len({g for r in rows for g in r['exposures'] if g.startswith('SDSS:')}),
        nsc_exposures=len({g for r in rows for g in r['exposures'] if g.startswith('NSC:')}),
        nsc_nights=len({g for r in rows for g in r['nsc_epoch_days']}))
    if zero is not None:
        z=(motion-np.asarray(zero))/(error*np.asarray(scale))
        value.update(training_center_scale_1sigma=np.mean(abs(z)<=1,axis=0),
                     training_center_scale_2sigma=np.mean(abs(z)<=2,axis=0),
                     validation_center_residual=np.median(motion-np.asarray(zero),axis=0))
    return value


def learn(rows):
    if len(rows)<60:return dict(n=len(rows),supported=False)
    motion,error=vectors(rows)
    center=np.median(motion,axis=0)
    scale=np.median(abs((motion-center)/error),axis=0)/NORMAL_ABS_MEDIAN
    rng=np.random.default_rng(SEED);centers=[];scales=[]
    for start in range(0,128,8):
        ids=rng.integers(0,len(rows),(8,len(rows)))
        c=np.median(motion[ids],axis=1)
        centers.extend(c)
        scales.extend(np.median(abs((motion[ids]-c[:,None,:])/error[ids]),axis=1)/NORMAL_ABS_MEDIAN)
    return dict(n=len(rows),supported=bool(np.isfinite(scale).all() and (scale>0).all()),
        median_zero=center,marginal_total_error_multiplier=scale,
        zero_object_interval95=np.quantile(centers,[.025,.975],axis=0).T,
        scale_object_interval95=np.quantile(scales,[.025,.975],axis=0).T,
        uncertainty_scope='Conditional objects; common group and spatial errors are not removed by increasing source count.')


def register_models(rows,new_windows):
    training=[r for r in rows if r['window'] not in new_windows and split_identity(r['identity'])<3]
    by_condition=defaultdict(list)
    for r in training:by_condition[condition(r)].append(r)
    return {key:learn(part) for key,part in sorted(by_condition.items())}


def run(registration,output):
    plan=read(registration);manifest=Path(plan['frozen_manifest'])
    if digest(manifest)!=plan['frozen_manifest_sha256']:raise ValueError('Frozen scope changed')
    frozen=manifest.parent;blind=frozen.parent/'blind_quality'
    rows=list(load_blind(frozen,blind).values())
    new_windows={r['window'] for r in plan['selected']}
    output.mkdir(parents=True,exist_ok=True)
    models=register_models(rows,new_windows)
    evaluated={}
    for key,model in models.items():
        subset=[r for r in rows if condition(r)==key]
        parts={'old_identity_validation':[r for r in subset if r['window'] not in new_windows and split_identity(r['identity'])==3],
               'old_identity_confirmation':[r for r in subset if r['window'] not in new_windows and split_identity(r['identity'])==4],
               'new_window_transfer':[r for r in subset if r['window'] in new_windows]}
        evaluated[key]={}
        for tag,part in parts.items():
            m=metrics(part,model['median_zero'],model['marginal_total_error_multiplier']) if model['supported'] else metrics(part)
            m['support_sufficient']=len(part)>=20
            if model['supported'] and len(part)>=20:
                raw=np.abs(np.asarray(m['centered_1sigma'])-.682689492)
                revised=np.abs(np.asarray(m['training_center_scale_1sigma'])-.682689492)
                raw2=np.abs(np.asarray(m['centered_2sigma'])-.954499736)
                revised2=np.abs(np.asarray(m['training_center_scale_2sigma'])-.954499736)
                m['coverage_improvement_by_axis']=(revised<=raw)&(revised2<=raw2+.02)
                m['calibrated_marginal_coverage_by_axis']=(revised<=.05)&(revised2<=.03)
            evaluated[key][tag]=m
    for key,model in models.items():
        tests=list(evaluated[key].values())
        model['replicated_qso_quality_support']=bool(model['supported'] and all(t.get('support_sufficient') and all(t.get('calibrated_marginal_coverage_by_axis',[False,False])) and all(t.get('coverage_improvement_by_axis',[False,False])) for t in tests))
        model['stellar_error_adoption']=False
    strata={}
    for low in (21,22):
        sample=[r for r in rows if low<=r['magnitude']<low+1]
        sets={'all':sample,'new':[r for r in sample if r['window'] in new_windows]}
        for run in (1,2):sets['single_run' if run==1 else 'multiple_runs']=[r for r in sample if (r['sdss_runs']==1)==(run==1)]
        for kind in ('r_existing','extra_r_anchor','non_r_anchor'):sets[kind]=[r for r in sample if r['kind']==kind]
        for tag,part in sets.items():
            strata[f'{low}__{tag}']=dict(all_blind=metrics(part),
                strict_published=merge_strata(part,('blind_all','Xiao','NSC')),
                conditions=dict(nsc_nights_p16_p50_p84=np.quantile([r['nsc_nights'] for r in part],[.16,.5,.84]) if part else None,
                    nsc_seasons_p16_p50_p84=np.quantile([r['nsc_seasons'] for r in part],[.16,.5,.84]) if part else None,
                    joint_span_p16_p50_p84=np.quantile([r['joint_span'] for r in part],[.16,.5,.84]) if part else None))
    groups=defaultdict(list)
    for r in rows:
        origin,ra,dec=r['identity'];pix=int(hp.ang2pix(32,ra,dec,lonlat=True))
        groups[(int(r['magnitude']),pix)].append(r)
    spatial={}
    for (low,pix),part in sorted(groups.items()):
        if len(part)<30:continue
        m=metrics(part)
        rng=np.random.default_rng(SEED+pix+low)
        motion,_=vectors(part)
        boot=[]
        for start in range(0,128,8):boot.extend(np.median(motion[rng.integers(0,len(part),(8,len(part)))],axis=1))
        bounds=np.quantile(boot,[.025,.975],axis=0).T
        supported=m['sdss_runs']>=3 and m['nsc_nights']>=5
        m.update(pixel=pix,nside=32,pixel_area_sqdeg=hp.nside2pixarea(32,degrees=True),
            median_object_interval95=bounds,physical_support_sufficient=supported,
            qso_zero_goal_by_axis=np.maximum(abs(bounds[:,0]),abs(bounds[:,1]))<=.5,
            caveat='Object intervals do not certify unmeasured common spatial modes. Display pixel is not calibration parameter area.')
        spatial[f'{low}__{pix}']=m
    report=dict(frozen_manifest_sha256=digest(manifest),models=models,validation=evaluated,
        strata=strata,spatial=spatial,slurm_job=os.environ['SLURM_JOB_ID'],
        definition=dict(split='Original optical identity SHA256 modulo5: 0–2 train,3 validate,4 confirm; eight new windows excluded from training.',
            training_windows='Historical old coverage; all regions previously examined. New-window transfer is a frozen diagnostic confirmation, not unseen-sky validation.',
            empirical_scale='Marginal total reported PM error multiplier from QSO center-adjusted absolute-normalized median / 0.67448975. Original formal and reference components preserved.',
            unchanged_fit=True,stellar_transfer='Not calibrated on faint Gaia; source color/morphology transfer to stars needs separate evidence.',
            recommended_use='Conditional research usability. Source identity and unmeasured common systematic error remain separate.'))
    check_aggregate(report);save_json(output/'v20_quality_calibration_summary.json',report)
    print('QUALITY_MODELS_DONE',len(rows),len(models),len(spatial),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('registration',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute nodes')
    run(**vars(a))
