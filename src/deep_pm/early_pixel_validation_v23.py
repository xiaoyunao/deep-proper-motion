"""Fixed forty, independent backgrounds, conditional injections and run holdout.

No result from this finite experiment changes the production catalogue. Failed
frame registration blocks adoption, but never replaces the nontrivial pixel
information gate in the blank-position experiment.
"""
import argparse
from collections import Counter
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path
from time import monotonic
import os
import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares
from scipy.stats import beta, t as student_t
from .astrometry import project,unproject
from .blind_catalogue_quality import read
from .coordinate_metric import AngularTree,fit_vector_motion
from .early_pixel_targets_v23 import TargetFactory
from .early_pixel_model_v23 import JointPixels,sky_at,numerical_jacobian
from .early_pixel_trial_v23 import fit_case
from .full_coordinate_sources import write_rows
from .io import digest,save_json
from .public_summary import check_aggregate

FACTORY=None
ROOT=None
ROWS=None
NONLINEAR=set()
COMPATIBLE_COMPLETED=dict(implementation_sha256='784f2f530d68fe14279369f79390209d3efc7503862d94f198ce7105a869543f',trial_sha256='919a04e8f44bd8114a7036a2a6a21e008de2e8d15d804d98b2180b6299a255d6',model_sha256='6a57583ee4690221a6da3acc2d08d609bb7e60fee7fba2c3d5e80eeaaaeb0d97')


def controls(base,root,rows,factory):
    path=root/'pixels/private_background_registration.json'
    if path.exists():return read(path)
    # Positions and image/crowding opportunity are registered before fitting any
    # control. Different controls have >=10 arcsec separation, not repeated noise.
    rng=np.random.default_rng(230027);center=(211.363708,28.534444);positions=[]
    for a in factory.assets:
        with fits.open(a['catalogue'],memmap=False) as h:positions.append(np.column_stack([h[1].data['RA'],h[1].data['DEC']]))
    cat=np.concatenate(positions);tree=AngularTree(project(*cat.T,center),center)
    templates=[r for r in rows if r['status']=='conditional_late_association'];chosen=[];xychosen=[];counts=Counter()
    for j in range(60000):
        if all(counts[k]>=n for k,n in [('blank',300),('injection_isolated',150),('injection_neighbor',150)]):break
        template=templates[j%len(templates)];anchor=project([template['ra']],[template['dec']],center)[0];point=anchor+rng.uniform(-120000,120000,2)
        if xychosen and np.min(np.linalg.norm(np.array(xychosen)-point,axis=1))<10000:continue
        near=float(tree.query(point)[0]);kind='blank' if counts['blank']<300 else ('injection_isolated' if near>=3000 else 'injection_neighbor')
        if kind=='blank' and near<3000:continue
        if kind=='injection_isolated' and (near<3000 or counts[kind]>=150):continue
        if kind=='injection_neighbor' and (near<1200 or near>=3000 or counts[kind]>=150):continue
        position=unproject([point],center)[0];cuts,ref,reg,meta,exits=factory.make(template,position=position)
        if not cuts:continue
        good=True
        for cut,m in zip(cuts,meta):
            field=factory.getfield(m['field_tag']);x,y=cut.base_pixel;xx=int(round(x));yy=int(round(y))
            if kind!='injection_neighbor' and np.any(field.objects[yy-3:yy+4,xx-3:xx+4]):good=False;break
        if not good:continue
        number=len(chosen);speed=[0,25,50,100][number%4];angle=rng.uniform(0,2*np.pi);mag=float(rng.uniform(22,23));truth=np.array([rng.uniform(-100,100),speed*np.cos(angle),rng.uniform(-100,100),speed*np.sin(angle)])
        # Test the actual early moving footprint, including central masks. The
        # synthetic late schedule is explicitly conditional on a real template;
        # it is not used to infer new NSC CCD completeness or source numbers.
        for cut,m in zip(cuts,meta):
            x,y=cut.point(sky_at(truth,cut.time,2015.5));field=factory.getfield(m['field_tag']);xx=int(round(x));yy=int(round(y))
            if xx<4 or yy<4 or xx+4>=field.pixel.shape[1] or yy+4>=field.pixel.shape[0] or field.bad[yy-3:yy+4,xx-3:xx+4].any():good=False;break
            if np.min(np.hypot(cut.gx-x,cut.gy-y))>.8:good=False;break
        if not good:continue
        chosen.append(dict(control_index=number,kind=kind,position_deg=position,template_frozen_index=template['frozen_index'],r=mag,truth_state=truth,nearest_released_SDSS_source_mas=near,early_runs=[m['run'] for m in meta],seed=230027+number*7919));xychosen.append(point);counts[kind]+=1
    proof=dict(controls=chosen,counts=dict(counts),seed=230027,attempts=j+1,frozen_late_sha256=digest(root/'pixels/private_frozen_forty_late.json.gz'),implementation_sha256=digest(Path(__file__)),scope='Registered before control fits. Distinct real early backgrounds >=10arcsec; actual masks and moving early footprints. Blank and isolated controls exclude released sources within3arcsec and central OBJECT masks. Neighbor injections have1.2-3arcsec catalogued neighbors. Late sampling is a conditional real-template schedule, not an NSC footprint/completeness simulation; colour, confusion below catalogue, noise correlations and absolute frame bias remain limitations.')
    save_json(path,proof);print('V23_CONTROL_REGISTRATION',dict(counts),j+1,flush=True);return proof


def synthetic(control,template,factory):
    row=dict(template);position=np.asarray(control['position_deg']);row.update(ra=position[0],dec=position[1],r0=control['r'],category=control['kind']);truth=np.asarray(control['truth_state']);rng=np.random.default_rng(control['seed'])
    cuts,ref,reg,meta,exits=factory.make(row,position=position);cov=np.asarray(row['late_covariance']);times=np.asarray(row['late_time']);noise=np.array([rng.multivariate_normal([0,0],c) for c in cov]);xy=sky_at(truth,times,2015.5)+noise
    late=fit_vector_motion(xy,times,cov,2015.5,robust=False);row.update(late_xy=xy,late_only_state=late.coeff.T.ravel(),late_only_conditional_covariance=late.cov)
    injected=[];negative=[]
    for cut,m in zip(cuts,meta):
        if control['kind']=='blank':injected.append(cut);continue
        jitter=rng.normal(0,cut.floor,2);x,y=cut.point(sky_at(truth,cut.time,2015.5)+jitter);field=factory.getfield(m['field_tag']);image=cut.psf.image(x,y);negative.append(float(-image[image<0].sum()));p=np.maximum(cut.psf.sample(x,y,cut.gx,cut.gy),0)/(1+negative[-1]);flux=10**(.4*(22.5-control['r']));signal=flux*p
        # Existing real background noise is not generated a second time.
        # Add source shot noise in electrons with official gain and calib.
        conversion=field.calib[cut.gx.astype(int)]/field.a['gain'];electrons=signal/conversion;source=rng.poisson(np.maximum(electrons,0))*conversion
        injected.append(replace(cut,data=cut.data+source,sigma=np.sqrt(cut.sigma**2+signal*conversion)))
    return row,(injected,ref,reg,meta,exits),negative


def run_holdouts(row,factory):
    if row['status']!='conditional_late_association':return []
    cuts,ref,reg,meta,exits=factory.make(row);out=[]
    if len(cuts)<2:return out
    for j,held in enumerate(cuts):
        train=[c for i,c in enumerate(cuts) if i!=j];model=JointPixels(train,row['late_xy'],row['late_time'],row['late_covariance']);fit=model.fit(row['late_only_state']);entry=dict(held_physical_run=held.run,training_early_runs=len(train),training_numerically_identified=fit['fully_identified'])
        if not fit['fully_identified']:out.append(entry);continue
        predicted=sky_at(fit['state'],held.time,2015.5)
        # Independent held-run localization initialized at the original optical
        # position (zero), never at its training prediction or final all-run fit.
        fun=lambda p:held.profile(p)[0];answer=least_squares(fun,[0.,0.],jac=lambda p:numerical_jacobian(fun,p,[.5,.5]),x_scale=[100,100],max_nfev=120);normal=answer.jac.T@answer.jac
        _,flux,bg,error=held.profile(answer.x);entry.update(held_flux_SNR=flux/error)
        if np.linalg.matrix_rank(normal)==2 and answer.success and flux/error>=5:
            c=np.linalg.inv(normal)+np.eye(2)*held.floor**2;dt=held.time-2015.5;design=np.array([[1,dt,0,0],[0,0,1,dt]]);prediction_cov=design@fit['conditional_covariance']@design.T;difference=predicted-answer.x;total=c+prediction_cov;entry.update(conditional_position_residual_mas=difference,conditional_position_covariance=total,conditional_mahalanobis2=float(difference@np.linalg.solve(total,difference)))
        out.append(entry)
    return out


def nonlinear_check(row,factory):
    cuts,ref,reg,meta,exits=factory.make(row)
    if not cuts:return dict(status='no_early_opportunity')
    model=JointPixels(cuts,row['late_xy'],row['late_time'],row['late_covariance']);fit=model.fit(row['late_only_state'])
    if not fit['fully_identified']:return dict(status='unidentified')
    response=model.common_response(fit,ref,row['late_reference_draws']);checks=[]
    for k in [0,13,27]:
        shifted=[replace(c,offset_mas=c.offset_mas+ref[k,i]) for i,c in enumerate(cuts)];other=JointPixels(shifted,np.asarray(row['late_xy'])-np.asarray(row['late_reference_draws'])[k],row['late_time'],row['late_covariance']);actual=other.fit(fit['parameters'],starts=False)
        checks.append(dict(draw=k,actual_state_change=actual['state']-fit['state'],linear_state_change=response[k],maximum_position_disagreement_mas=float(np.max(abs((actual['state']-fit['state']-response[k])[[0,2]]))),maximum_PM_disagreement_masyr=float(np.max(abs((actual['state']-fit['state']-response[k])[[1,3]])))))
    return dict(status='computed',draw_checks=checks)


def task(work):
    mode,value=work;folder=ROOT/'pixels/validation';path=folder/f'private_{mode}_{value if mode=="real" else value["control_index"]:04d}.json';ticket=dict(implementation_sha256=digest(Path(__file__)),trial_sha256=digest(Path(__file__).with_name('early_pixel_trial_v23.py')),model_sha256=digest(Path(__file__).with_name('early_pixel_model_v23.py')),late_sha256=digest(ROOT/'pixels/private_frozen_forty_late.json.gz'))
    if path.exists():
        prior=read(path)
        old=prior['ticket'];compatible=all(old.get(k)==v for k,v in COMPATIBLE_COMPLETED.items()) and old.get('late_sha256')==ticket['late_sha256']
        if old!=ticket and not compatible:raise ValueError('Changed validation implementation/input')
        # The only numerical change is exactly-zero PSF support outside the
        # observed cut. All completed old fits evaluated a valid PSF or exactly
        # zero interpolation there already; interrupted profiles were unsaved.
        return prior['result']
    if mode=='real':
        row=next(r for r in ROWS if r['frozen_index']==value);result=fit_case(row,FACTORY);result['held_run_predictions']=run_holdouts(row,FACTORY)
        if value in NONLINEAR and row['status']=='conditional_late_association':result['nonlinear_common_validation']=nonlinear_check(row,FACTORY)
    else:
        template=next(r for r in ROWS if r['frozen_index']==value['template_frozen_index']);row,prepared,negative=synthetic(value,template,FACTORY);result=fit_case(row,FACTORY,position=value['position_deg'],prepared=prepared);result.update(control_index=value['control_index'],truth_state=value['truth_state'],PSF_negative_full_flux_fraction=negative,conditional_simulation=True)
    result['experiment_mode']=mode;save_json(path,dict(ticket=ticket,result=result));print('V23_PIXEL_CASE',mode,result.get('frozen_index'),result.get('control_index'),result['status'],result.get('information_gate_pass'),round(result.get('elapsed_seconds',0),2),flush=True);return result


def aggregate(results,registration,root):
    summary=dict(registered_controls=registration['counts'],control_registration_sha256=digest(root/'pixels/private_background_registration.json'),groups={},slurm_job=os.environ['SLURM_JOB_ID'])
    for category in sorted({r['category'] for r in results}):
        group=[r for r in results if r['category']==category];computed=[r for r in group if r['status']=='computed'];entry=dict(initial=len(group),status=dict(Counter(r['status'] for r in group)),information_pass=sum(r.get('information_gate_pass',False) for r in group),adoption_pass=sum(r.get('adoption_gate_pass',False) for r in group),actual_early_run_counts=dict(Counter(r.get('early_physical_runs',0) for r in group)))
        if category=='blank':
            n=len(group);k=entry['information_pass'];entry['information_false_accept_one_sided95_upper']=float(beta.ppf(.95,k+1,n-k)) if k<n else 1.;entry['null_fraction_scope']='Same nontrivial information gate, NOT adoption gate (registration failures block all adoption). Conditional actual backgrounds, synthetic late schedule.'
        if category.startswith('injection'):
            for label,part in [('all_computed',computed),('information_selected',[r for r in computed if r.get('information_gate_pass')])]:
                stat=dict(n=len(part))
                if part:
                    errors=np.array([np.asarray(r['joint_state_local'])[[1,3]]-np.asarray(r['truth_state'])[[1,3]] for r in part]);se=errors.std(axis=0,ddof=1)/np.sqrt(len(part)) if len(part)>1 else np.full(2,1e20);stat.update(mean_PM_bias_masyr=errors.mean(axis=0),mean_bias_interval95=np.column_stack([errors.mean(axis=0)-student_t.ppf(.975,max(1,len(part)-1))*se,errors.mean(axis=0)+student_t.ppf(.975,max(1,len(part)-1))*se]),NMAD_PM_masyr=1.4826*np.median(abs(errors-np.median(errors,axis=0)),axis=0),conditional_Wald_1sigma_coverage=np.mean(abs(errors)<=np.array([np.sqrt(np.diag(r['conditional_covariance']))[[1,3]] for r in part]),axis=0))
                    for level in ['1.0','3.841458820694124']:
                        covered=[]
                        for r in part:
                            profiles=r.get('joint_conditional_profiles',{});interval=[profiles.get(str(a),{}).get(level) for a in [1,3]]
                            if all(v is not None and all(x is not None for x in v) for v in interval):covered.append([interval[j][0]<=r['truth_state'][a]<=interval[j][1] for j,a in enumerate([1,3])])
                        stat['profile_'+level]=dict(n=len(covered),axis_coverage=np.mean(covered,axis=0) if covered else None)
                entry[label]=stat
        holds=[h for r in group for h in r.get('held_run_predictions',[])];valid=[h for h in holds if 'conditional_position_residual_mas' in h];entry['whole_early_run_holdout']=dict(attempted=len(holds),independent_positions_with_SNR5=len(valid),conditional_position_P68_mas=float(np.quantile([np.linalg.norm(h['conditional_position_residual_mas']) for h in valid],.68)) if valid else None,conditional_ellipse68_coverage=float(np.mean([h['conditional_mahalanobis2']<=2.278868566 for h in valid])) if valid else None,conditional_ellipse95_coverage=float(np.mean([h['conditional_mahalanobis2']<=5.991464547 for h in valid])) if valid else None)
        summary['groups'][category]=entry
    summary.update(scope='Frozen40 physical objects; all exits retained. Injections add moving spatial PSFs and source Poisson to distinct real backgrounds, conditional on measured late schedules and fixed C0/frame point offsets. Conditional profile coverage does not certify absolute frame errors or NSC footprint completeness. Common covariance propagated separately; nonlinear perturbation checks remain private and summarized independently. No HST/target PM used in selection. Absolute adoption also requires independently passed registration; no415 expansion if that or faint-control validation fails.',frozen_forty_415_decision='Do not expand415 unless forty, registration, injection, null and independent-run criteria all pass.')
    checks=[v for r in results for v in r.get('nonlinear_common_validation',{}).get('draw_checks',[])]
    summary['nonlinear_common_response_check']=dict(actual_refits=len(checks),maximum_position_difference_mas=max([v['maximum_position_disagreement_mas'] for v in checks],default=None),maximum_PM_difference_masyr=max([v['maximum_PM_disagreement_masyr'] for v in checks],default=None))
    check_aggregate(summary);save_json(root/'pixels/v23_pixel_validation_summary.json',summary)


def run(base,root,workers):
    global FACTORY,ROOT,ROWS,NONLINEAR
    start=monotonic();ROOT=root;ROWS=read(root/'pixels/private_frozen_forty_late.json.gz');FACTORY=TargetFactory(base,root);registered=controls(base,root,ROWS,FACTORY)
    NONLINEAR={min(r['frozen_index'] for r in ROWS if r['category']==c and r['status']=='conditional_late_association') for c in sorted({r['category'] for r in ROWS}) if any(r['category']==c and r['status']=='conditional_late_association' for r in ROWS)}
    # Preload immutable fields/reference before fork; private child model arrays
    # remain bounded. BLAS is single-threaded in the Slurm launcher.
    for r in ROWS:
        if r['status']=='conditional_late_association':FACTORY.make(r)
    work=[('real',r['frozen_index']) for r in ROWS]+[('control',r) for r in registered['controls']]
    (root/'pixels/validation').mkdir(exist_ok=True)
    with get_context('fork').Pool(workers) as pool:results=list(pool.imap_unordered(task,work,chunksize=1))
    aggregate(results,registered,root);print('V23_PIXEL_VALIDATION_COMPLETE',len(results),monotonic()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    if a.workers>int(os.environ.get('SLURM_CPUS_PER_TASK',1)):p.error('Workers exceed actual allocation')
    run(a.base,a.root,a.workers)
