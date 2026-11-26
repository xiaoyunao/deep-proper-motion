"""Frozen optical and bright-star identities under the complete reference update.

Fixed-member fits preserve quality exits; final blind optical recovery is a
separate known-position readout. No colour/DCR candidate is applied here.
"""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import os
import numpy as np
import pandas as pd
from .__main__ import tomllib
from .astrometry import project
from .blind_catalogue_quality import read
from .cluster_motion import mean
from .continuous_reference import PRECISION,SPATIAL,read_offsets,reference_frame
from .coordinate_metric import native_covariance,projected_covariance,fit_vector_motion,reference_covariance,AngularTree,old_influence
from .dark_precision import query_unique
from .full_coordinate_sources import write_rows
from .full_reference_run import local_support
from .io import digest,save_json
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .public_summary import check_aggregate
from .standard_motion import icrs_motion
from .usability_cluster import summarize,REFERENCE


def run(root):
    base=root.parent;dest=root/'ngc';dest.mkdir(exist_ok=True)
    original=base/'v2.0/ngc_motion/private_cluster_diagnostics.json.gz';frozen=read(original)
    selected=frozen['selected'];oldmap={tuple(r['source_key']):r for r in frozen['entries']}
    bright={k for k,r in oldmap.items() if r.get('gaia')}
    dark={tuple(k) for tag,keys in selected.items() if tag.startswith('old__target__') and float(tag.rsplit('__',1)[1])>=21 for k in keys}
    target_bright={tuple(k) for tag,keys in selected.items() if tag.startswith('old__target__') and float(tag.rsplit('__',1)[1])<21 for k in keys}
    if len(dark)!=197:raise ValueError('Frozen197 original dark identities changed')
    chosen=bright|dark|target_bright;plan=read(root/'registration/private_full_coordinate_plan.json');lookup={e['core']:e for e in plan['cores']}
    graph=read(root/'graph/v22_global_graph_summary.json');config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    stellar=read(base/'v2.1/stellar-attribution/private_stellar_positions.json.gz');measure=defaultdict(list)
    for m in stellar['measurements']:measure[(tuple(m['source_key']),m['calibration_group'],m['time'])].append(m)
    details={};exit_counts=Counter();bright_old=[];bright_new=[];bright_se=[];private=[]
    for core in sorted({k[0] for k in chosen}):
        entry=lookup[core];component=next(r['component'] for r in graph['components'] if r['center_deg']==entry['center'])
        file=root/'reference'/f'component-{component:04d}'/'private_c0_unfiltered.npz';center=reference_frame(file);solution=read_offsets(file)
        if not np.allclose(center,entry['center'],atol=1e-12,rtol=0):raise ValueError('Saved bright-position predictions require explicit chart re-expression')
        data,xy,t,sigma,keep=quality_detections(Path(entry['source']),center,config,settings['production']);keys=calibration_groups(data);idx=pd.Index(data.detection_id)
        names=np.array(sorted(solution));draws=np.stack([solution[g][1] for g in names]);support=local_support(data.ra_deg,data.dec_deg,center,keys,names,draws)[0]
        for key in sorted(k for k in chosen if k[0]==core):
            prior=oldmap[key];old=prior['row'];ix=idx.get_indexer(old['detection_ids'])
            if np.any(ix<0):exit_counts['missing_original_detection']+=1;continue
            if any(g not in solution for g in keys[ix]):exit_counts['missing_reference_group']+=1;continue
            cp=projected_covariance(data.ra_deg.iloc[ix],data.dec_deg.iloc[ix],center,native_covariance(sigma[ix]));zero=np.array([solution[g][0] for g in keys[ix]])
            model=fit_vector_motion(xy[ix]-zero,t[ix],cp,config['reference_epoch']);common,response=reference_covariance(model.influence,np.stack([solution[g][1] for g in keys[ix]],axis=1));out=icrs_motion(model.coeff[0],model.coeff[1],center,model.cov+common)
            j=np.asarray(out['state_jacobian']);response=response@j.T
            localpass=bool(support[ix].all());formal=bool(np.max(out['pm_error_masyr'])<=settings['production']['max_axis_pm_error_masyr'])
            exit_counts['fixed_new_native_support_exit']+=not localpass;exit_counts['fixed_new_formal_error_exit']+=not formal
            row=dict(pm_icrs_masyr=out['pm_icrs_masyr'],pm_error_masyr=out['pm_error_masyr']);details[key]=dict(row=row,independent_icrs=(j@model.cov@j.T)[np.ix_([1,3],[1,3])],reference_draws=response[:,[1,3]],graph_sha256=digest(file))
            record=dict(source_key=key,new=row,local_support_pass=localpass,formal_error_pass=formal)
            if key in bright:
                predictions=[]
                for k in ix:
                    m=measure.get((key,keys[k],float(t[k])),[])
                    if len(m)!=1:break
                    predictions.append(xy[k]-np.asarray(m[0]['raw_graph']))
                if len(predictions)==len(ix):
                    truth=fit_vector_motion(np.asarray(predictions),t[ix],cp,model.epoch,frozen_weight=model.weight);p=icrs_motion(truth.coeff[0],truth.coeff[1],center,truth.cov)
                    old_h=old_influence(old,t[ix],sigma[ix]);old_expected=np.einsum('pni,ni->p',old_h,np.asarray(predictions));old_truth=icrs_motion(old_expected[[0,2]],old_expected[[1,3]],center,np.eye(4))
                    bright_new.append(np.asarray(out['pm_icrs_masyr'])-p['pm_icrs_masyr']);bright_old.append(np.asarray(old['pm_icrs_masyr'])-old_truth['pm_icrs_masyr']);bright_se.append(out['pm_error_masyr'])
                    record['new_residual_from_same_estimator_apparent_Gaia_prediction']=bright_new[-1]
                else:exit_counts['bright_missing_saved_full_apparent_prediction']+=1
            private.append(record)
        print('NGC_FIXED_CORE',core,flush=True)
    def cohort(keys):
        part=[k for k in keys if k in details];weights=[1/np.mean(np.asarray(oldmap[k]['row']['pm_error_masyr'])**2) for k in part]
        return dict(frozen_denominator=len(keys),new_computable=len(part),old=mean([oldmap[k] for k in part],weights,REFERENCE),new=mean([details[k] for k in part],weights,REFERENCE))
    phot_path=base/'v2.0/ngc_photometry/private_photometric_parent.csv.gz'
    # Resolve this existing metadata product by its fixed v2.1 run inputs if the
    # historical storage name differs; never regenerate a photometric sample.
    if not phot_path.exists():
        matches=list((base/'v2.0').glob('*/private_photometric_parent.csv.gz'))
        if len(matches)!=1:raise ValueError('Need unambiguous frozen photometric parent')
        phot_path=matches[0]
    from .ngc5466_demo import TARGET
    parent=pd.read_csv(phot_path);phot=parent.loc[parent.cmd_v20&parent.radius.between(*TARGET,inclusive='left')&parent.r0.between(22,23,inclusive='left')]
    if len(phot)!=415:raise ValueError('Frozen415 optical denominator changed')
    import duckdb
    db=duckdb.connect(str(root/'catalogue/catalogue.duckdb'),read_only=True);accepted=db.execute("SELECT * FROM accepted WHERE ra_deg BETWEEN 208 AND 214 AND dec_deg BETWEEN 27 AND 31").fetchdf();db.close()
    tree=AngularTree(project(accepted.ra_deg,accepted.dec_deg,(211.3637,28.5344)),(211.3637,28.5344));goals=project(phot.ra,phot.dec,(211.3637,28.5344));good,near=query_unique(tree,goals,700,300);claims=Counter(near[good]);good&=np.array([claims.get(j,0)==1 for j in near])
    report=dict(frozen_cluster_sha256=digest(original),fixed_original_dark=cohort(sorted(dark)),fixed_optical_target_bright=cohort(sorted(target_bright)),fixed_optical_target_bright_and_dark=cohort(sorted(target_bright|dark)),exits=dict(exit_counts),
        bright_same_physical_members=dict(n=len(bright_new),old_residual_to_same_estimator_apparent_five_parameter_prediction=summarize(bright_old),new_residual_to_same_estimator_apparent_five_parameter_prediction=summarize(bright_new,bright_se),scope='Saved Gaia5-parameter apparent positions include known annual parallax; evaluated only after fits. Old and new estimators each applied to the same frozen apparent5-parameter predicted positions. New axis coverage conditions on Gaia prediction, excludes correlated Gaia truth uncertainty; no faint-Gaia acceptance.'),
        frozen_optical415=dict(denominator=415,final_unique_known_position_matches=int(good.sum()),known_position_gate_mas=700,by_half_mag={str(low):dict(optical=int(((phot.r0>=low)&(phot.r0<low+.5)).sum()),matched=int((good&(phot.r0>=low)&(phot.r0<low+.5)).sum())) for low in (22,22.5)}),
        slurm_job=os.environ['SLURM_JOB_ID'],scope='Fixed original optical/bright identities, original per-source inverse-variance weights held identical for old/new cohort means; quality failures retained with explicit counts. Common reference perturbations combined before covariance, not divided bysqrtN. Photometric candidates include foreground/background, not cluster-member truth. New blind optical matching is known-position conditional recovery; main frozen catalogue unchanged.')
    check_aggregate(report);save_json(dest/'v22_fixed_ngc_bright_dark_summary.json',report);write_rows(dest/'private_fixed_ngc_readout.json.gz',private)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.root)
