"""Finite joint-pixel pilot; adoption is separate from conditional information."""
import argparse
from pathlib import Path
from time import monotonic
import os
import numpy as np
from .blind_catalogue_quality import read
from .early_pixel_targets_v23 import prepare,TargetFactory
from .early_pixel_model_v23 import JointPixels,sky_at
from .full_coordinate_sources import write_rows
from .standard_motion import icrs_motion
from .io import digest,save_json
from .public_summary import check_aggregate


def fit_case(row,factory,position=None,profile=True):
    start=monotonic();result=dict(frozen_index=int(row['frozen_index']),category=row['category'],r0=row['r0'],late_status=row['status'])
    if row['status']!='conditional_late_association':return dict(result,status='no_unique_supported_late_trajectory')
    cuts,ref,reg,meta,unsupported=factory.make(row,position=position,require_tie=False)
    result.update(early_physical_runs=len(cuts),early_fields=meta,early_exits=unsupported,late_detections=len(row['late_time']),late_year_span=float(np.ptp(row['late_time'])),all_frame_ties_independently_supported=bool(cuts and all(v['independent_frame_tie_pass'] for v in meta)))
    if not cuts:return dict(result,status='no_early_pixel_opportunity',elapsed_seconds=monotonic()-start)
    latexy=np.asarray(row['late_xy']);latetime=np.asarray(row['late_time']);latecov=np.asarray(row['late_covariance']);initial=np.asarray(row['late_only_state']);model=JointPixels(cuts,latexy,latetime,latecov);fit=model.fit(initial)
    result.update(status='computed',joint_state_local=fit['state'],joint_chi2=fit['chi2'],joint_success=fit['success'],joint_identifiable=fit['fully_identified'],competing_joint_minima=fit['competing_minima'],joint_start_chi2=fit['start_chi2'])
    if not fit['fully_identified'] or not fit['success']:return dict(result,status='unidentified_or_numerically_incomplete',elapsed_seconds=monotonic()-start)
    early_epoch=float(np.mean([c.time for c in cuts]));early_initial=initial.copy();early_initial[[0,2]]=sky_at(initial,early_epoch,2015.5)
    early=JointPixels(cuts,[],[],[],epoch=early_epoch);earlyfit=early.fit(early_initial)
    whit=np.linalg.inv(np.linalg.cholesky(latecov));late_res=np.einsum('nij,nj->ni',whit,sky_at(initial,latetime,2015.5)-latexy);null=float(np.sum(late_res**2))
    flux=[]
    for i,c in enumerate(cuts):
        w=1/c.sigma**2;bg=float(np.sum(c.data*w)/w.sum());null+=float(np.sum(((c.data-bg)/c.sigma)**2));point=sky_at(fit['state'],c.time,2015.5)+fit['parameters'][4+2*i:6+2*i];_,f,b,e=c.profile(point);flux.append(dict(time=c.time,run=c.run,flux=f,flux_error=e,SNR=f/e,background=b))
    delta=null-fit['chi2'];result.update(target_vs_early_empty_delta_chi2=delta,per_run_flux=flux,profile_flux_SNR=float(np.sqrt(sum(max(0,v['SNR'])**2 for v in flux))),early_only_state=earlyfit['state'],early_only_epoch=early_epoch,early_only_identified_axes=earlyfit['identifiable_state_axes'],competing_early_minima=earlyfit['competing_minima'])
    c0=model.common_response(fit,ref,np.asarray(row['late_reference_draws']));registration=model.common_response(fit,reg,np.zeros_like(row['late_reference_draws']));conditional=np.asarray(fit['conditional_covariance']);c0cov=np.cov(c0,rowvar=False);regcov=np.cov(registration,rowvar=False);cov=conditional+c0cov+regcov
    output=icrs_motion(fit['state'][[0,2]],fit['state'][[1,3]],tuple(position if position is not None else [row['ra'],row['dec']]),cov)
    latepm=np.diag(np.asarray(row['late_only_conditional_covariance']))[[1,3]];gain=1-np.diag(conditional)[[1,3]]/latepm
    result.update(standard_ICRS=output,conditional_covariance=conditional,reference_covariance=c0cov,frame_registration_covariance=regcov,synchronous_reference_state_draws=c0,synchronous_frame_registration_state_draws=registration,conditional_slope_variance_improvement=gain)
    early_size=None
    if earlyfit['full_covariance'] is not None and np.all(earlyfit['identifiable_state_axes'][[0,2]]):
        ec=earlyfit['conditional_covariance'][np.ix_([0,2],[0,2])];early_size=float(np.sqrt(max(0,np.linalg.eigvalsh(ec).max())*2.278868566));result['early_position_68_ellipse_major_mas']=early_size
    basepass=bool(delta>=25 and result['profile_flux_SNR']>=5 and early_size is not None and early_size<=200 and max(output['pm_error_masyr'])<=10 and max(gain)>=.2 and fit['competing_minima']==0 and earlyfit['competing_minima']==0 and np.all(earlyfit['identifiable_state_axes'][[0,2]]))
    if profile and basepass:
        result['joint_conditional_profiles']=model.profiles(fit,axes=(1,3));result['early_position_profiles']=early.profiles(earlyfit,axes=(0,2));intervals=[v for group in result['early_position_profiles'].values() for v in group.values()];closed=len(intervals)==4 and all(x is not None and np.isfinite(x) for v in intervals for x in v);basepass&=closed
    elif basepass:basepass=False # Screening without profiles never declares recovery.
    result.update(information_gate_pass=basepass,adoption_gate_pass=bool(basepass and result['all_frame_ties_independently_supported']),elapsed_seconds=monotonic()-start,
        scope='Free4 state plus one50mas nuisance per early physical run; only late positions once with frozen late-only robust weights. Existing target SDSS centroid never enters likelihood. Reference and independent-star frame-tie draws propagated separately, each internally synchronous, no spurious coupling by draw index. Pixel-noise Gaussian approximation tested by independent injections/empty positions. Conditional information pass is distinct from accepted reference/identity/astrophysical certification.')
    return result


def pilot(base,root):
    file=root/'pixels/private_frozen_forty_late.json.gz'
    if not file.exists():prepare(base,root)
    rows=read(file);factory=TargetFactory(base,root);selected=[]
    for name in ['no_cached_SDSS','positive_early_control','SDSS_quality_distance']:
        candidates=sorted([r for r in rows if r['category']==name and r['status']=='conditional_late_association'],key=lambda r:r['frozen_index'])
        if candidates:selected.append(candidates[0])
    results=[]
    for row in selected:
        result=fit_case(row,factory);results.append(result);print('V23_REAL_PIXEL_PILOT',row['frozen_index'],result['status'],result.get('early_physical_runs'),result.get('information_gate_pass'),result.get('elapsed_seconds'),flush=True)
    out=root/'pixels';write_rows(out/'private_real_pilot.json.gz',results)
    summary=dict(frozen_forty_late_sha256=digest(file),cases=[{k:v for k,v in r.items() if k in ['category','r0','status','early_physical_runs','late_detections','target_vs_early_empty_delta_chi2','profile_flux_SNR','early_position_68_ellipse_major_mas','conditional_slope_variance_improvement','information_gate_pass','adoption_gate_pass','all_frame_ties_independently_supported','elapsed_seconds']} for r in results],slurm_job=os.environ['SLURM_JOB_ID'],scope='Three metadata/frozen-index chosen real pilots; no tuning from target PM. Not the40-object validation, injection or empty-position acceptance. Independently unsupported frame ties prohibit adoption even with localization information.')
    check_aggregate(summary);save_json(out/'v23_real_pixel_pilot_summary.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    pilot(a.base,a.root)
