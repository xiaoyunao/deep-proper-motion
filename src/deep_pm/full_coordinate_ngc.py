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
from .astrometry import project,unproject
from .blind_catalogue_quality import read
from .cluster_motion import mean
from .continuous_reference import PRECISION,SPATIAL
from .coordinate_metric import native_covariance,projected_covariance,fit_vector_motion,reference_covariance,AngularTree,old_influence
from .dark_precision import query_unique
from .full_coordinate_sources import write_rows,component_map,reference_context
from .full_reference_run import local_support
from .io import digest,save_json
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .public_summary import check_aggregate
from .standard_motion import icrs_motion
from .usability_cluster import summarize,REFERENCE


def reexpress_positions(points,old_center,new_center):
    """Saved apparent positions retain their original chart until converted."""
    points=np.asarray(points)
    if np.allclose(old_center,new_center,atol=1e-12,rtol=0):return points
    return project(*unproject(points,old_center).T,new_center)


def run(root,history_root=None):
    base=history_root if history_root is not None else root.parent;dest=root/'ngc';dest.mkdir(exist_ok=True)
    original=base/'v2.0/ngc_motion/private_cluster_diagnostics.json.gz';frozen=read(original)
    selected=frozen['selected'];oldmap={tuple(r['source_key']):r for r in frozen['entries']}
    bright={k for k,r in oldmap.items() if r.get('gaia')}
    dark={tuple(k) for tag,keys in selected.items() if tag.startswith('old__target__') and float(tag.rsplit('__',1)[1])>=21 for k in keys}
    target_bright={tuple(k) for tag,keys in selected.items() if tag.startswith('old__target__') and float(tag.rsplit('__',1)[1])<21 for k in keys}
    if len(dark)!=197:raise ValueError('Frozen197 original dark identities changed')
    chosen=bright|dark|target_bright;plan=read(root/'registration/private_full_coordinate_plan.json');lookup={e['core']:e for e in plan['cores']}
    mapping,mapping_sha=component_map(str(root));config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    stellar=read(base/'v2.1/stellar-attribution/private_stellar_positions.json.gz');measure=defaultdict(list)
    for m in stellar['measurements']:measure[(tuple(m['source_key']),m['calibration_group'],m['time'])].append(m)
    details={};exit_counts=Counter();bright_old=[];bright_new=[];bright_se=[];private=[]
    for core in sorted({k[0] for k in chosen}):
        entry=lookup[core]
        data,old_xy,t,sigma,keep=quality_detections(Path(entry['source']),entry['center'],config,settings['production']);keys=calibration_groups(data);idx=pd.Index(data.detection_id);contexts={}
        for key in sorted(k for k in chosen if k[0]==core):
            prior=oldmap[key];old=prior['row'];ix=idx.get_indexer(old['detection_ids'])
            if np.any(ix<0):exit_counts['missing_original_detection']+=1;continue
            physical=mapping.reindex(keys[ix])
            if physical.isna().any():exit_counts['missing_reference_group']+=1;continue
            components=physical.astype(int).unique()
            if len(components)!=1:exit_counts['original_members_span_independent_reference_components']+=1;continue
            component=int(components[0])
            if component not in contexts:
                file=root/'reference'/f'component-{component:04d}'/'private_c0_unfiltered.npz'
                center,solution,names,draws,graph_sha=reference_context(str(file))
                xy=old_xy if np.allclose(center,entry['center'],atol=1e-12,rtol=0) else project(data.ra_deg,data.dec_deg,center)
                support=local_support(data.ra_deg,data.dec_deg,center,keys,names,draws)[0]
                contexts[component]=(center,solution,xy,support,graph_sha)
            center,solution,xy,support,graph_sha=contexts[component]
            cp=projected_covariance(data.ra_deg.iloc[ix],data.dec_deg.iloc[ix],center,native_covariance(sigma[ix]));zero=np.array([solution[g][0] for g in keys[ix]])
            model=fit_vector_motion(xy[ix]-zero,t[ix],cp,config['reference_epoch']);common,response=reference_covariance(model.influence,np.stack([solution[g][1] for g in keys[ix]],axis=1));out=icrs_motion(model.coeff[0],model.coeff[1],center,model.cov+common)
            j=np.asarray(out['state_jacobian']);response=response@j.T
            localpass=bool(support[ix].all());formal=bool(np.max(out['pm_error_masyr'])<=settings['production']['max_axis_pm_error_masyr'])
            exit_counts['fixed_new_native_support_exit']+=not localpass;exit_counts['fixed_new_formal_error_exit']+=not formal
            row=dict(pm_icrs_masyr=out['pm_icrs_masyr'],pm_error_masyr=out['pm_error_masyr']);details[key]=dict(row=row,independent_icrs=(j@model.cov@j.T)[np.ix_([1,3],[1,3])],reference_draws=response[:,[1,3]],graph_sha256=graph_sha)
            record=dict(source_key=key,new=dict(out,reference_epoch=config['reference_epoch']),
                original_detection_ids=old['detection_ids'],reference_component=component,reference_model_sha256=graph_sha,
                independent_icrs_covariance=j@model.cov@j.T,synchronous_reference_icrs_state_draws=response,
                local_support_pass=localpass,formal_error_pass=formal)
            if key in bright:
                predictions=[]
                for k in ix:
                    m=measure.get((key,keys[k],float(t[k])),[])
                    if len(m)!=1:break
                    predictions.append(old_xy[k]-np.asarray(m[0]['raw_graph']))
                if len(predictions)==len(ix):
                    new_predictions=reexpress_positions(predictions,entry['center'],center)
                    truth=fit_vector_motion(new_predictions,t[ix],cp,model.epoch,frozen_weight=model.weight);p=icrs_motion(truth.coeff[0],truth.coeff[1],center,truth.cov)
                    old_h=old_influence(old,t[ix],sigma[ix]);old_expected=np.einsum('pni,ni->p',old_h,np.asarray(predictions));old_truth=icrs_motion(old_expected[[0,2]],old_expected[[1,3]],entry['center'],np.eye(4))
                    bright_new.append(np.asarray(out['pm_icrs_masyr'])-p['pm_icrs_masyr']);bright_old.append(np.asarray(old['pm_icrs_masyr'])-old_truth['pm_icrs_masyr']);bright_se.append(out['pm_error_masyr'])
                    record['new_residual_from_same_estimator_apparent_Gaia_prediction']=bright_new[-1]
                    if plan.get('production_blind'):
                        record['bright_fixed_weights']=model.weight
                        record['bright_fixed_center']=center
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
    report=dict(frozen_cluster_sha256=digest(original),physical_reference_map_sha256=mapping_sha,fixed_original_dark=cohort(sorted(dark)),fixed_optical_target_bright=cohort(sorted(target_bright)),fixed_optical_target_bright_and_dark=cohort(sorted(target_bright|dark)),exits=dict(exit_counts),
        bright_same_physical_members=dict(n=len(bright_new),old_residual_to_same_estimator_apparent_five_parameter_prediction=summarize(bright_old),new_residual_to_same_estimator_apparent_five_parameter_prediction=summarize(bright_new,bright_se),scope='Saved Gaia5-parameter apparent positions include known annual parallax; evaluated only after fits. Old and new estimators each applied to the same frozen apparent5-parameter predicted positions. New axis coverage conditions on Gaia prediction, excludes correlated Gaia truth uncertainty; no faint-Gaia acceptance.'),
        frozen_optical415=dict(denominator=415,final_unique_known_position_matches=int(good.sum()),known_position_gate_mas=700,by_half_mag={str(low):dict(optical=int(((phot.r0>=low)&(phot.r0<low+.5)).sum()),matched=int((good&(phot.r0>=low)&(phot.r0<low+.5)).sum())) for low in (22,22.5)}),
        slurm_job=os.environ['SLURM_JOB_ID'],scope='Fixed original optical/bright identities, original per-source inverse-variance weights held identical for old/new cohort means; quality failures retained with explicit counts. Common reference perturbations combined before covariance, not divided bysqrtN. Photometric candidates include foreground/background, not cluster-member truth. New blind optical matching is known-position conditional recovery; main frozen catalogue unchanged.')
    if plan.get('production_blind'):
        report['bright_in_final_blind_catalogue']=final_bright(root,private)
    check_aggregate(report);save_json(dest/'v22_fixed_ngc_bright_dark_summary.json',report);write_rows(dest/'private_fixed_ngc_readout.json.gz',private)


def final_bright(root,rows):
    """Identify frozen bright controls retained with exactly the same physical members."""
    import duckdb
    from .full_coordinate_catalogue import member_key
    from .stream_json import array_items
    expected=[]
    fixed={}
    for row in rows:
        residual=row.get('new_residual_from_same_estimator_apparent_Gaia_prediction')
        if residual is None:
            continue
        pm=row['new']['pm_icrs_masyr']
        # Position predictions used the same five-parameter annual-parallax
        # series and frozen estimator as the known-member control above.
        key=member_key(row['original_detection_ids'])
        fixed[key]=row
        expected.append((key,*pm,*residual))
    control=pd.DataFrame(expected,columns=['key','fixed_e','fixed_n','res_e','res_n'])
    if control.key.duplicated().any():
        raise ValueError('Duplicate bright validation member set')
    con=duckdb.connect(str(root/'catalogue/catalogue.duckdb'),read_only=True)
    con.register('control',control)
    matched=con.execute("""SELECT c.key,a.core,a.slot,a.magnitude,a.pm_e,a.pm_n,a.se_e,a.se_n,a.sdss_runs,
            c.fixed_e,c.fixed_n,c.res_e,c.res_n FROM accepted a JOIN control c
            ON a.member_sha256=c.key""").fetchdf()
    con.close()
    if len(matched)>len(control):
        raise ValueError('A bright physical counterpart was claimed twice')
    selected=matched.loc[matched.magnitude.lt(21)].copy()
    # A changed origin or ordering can change numerical summation. Report the
    # direct PM check before using that fixed estimator's apparent truth.
    delta=selected[['pm_e','pm_n']].to_numpy()-selected[['fixed_e','fixed_n']].to_numpy()
    same=np.all(np.isclose(delta,0.,atol=1e-5,rtol=0),axis=1)
    selected["same_point"]=same
    verified_keys=set()
    for core, group in selected.loc[same].groupby('core'):
        folder=root/'sources'/core
        source=folder/'private_standard_motion.json.gz'
        if read(folder/'v22_source_core_summary.json')['private_output_sha256'][source.name]!=digest(source):
            raise ValueError('Changed bright source weights')
        slots=dict(zip(group.slot,group.key))
        for slot, actual in enumerate(array_items(source)):
            if slot not in slots:continue
            key=slots.pop(slot)
            prior=fixed[key]
            if member_key(actual['detection_ids'])!=key:
                raise ValueError('Bright source index and physical members disagree')
            weights=dict(zip(prior['original_detection_ids'],prior['bright_fixed_weights']))
            if (actual['reference_model_sha256']==prior['reference_model_sha256'] and
                np.allclose(actual['projection_center_deg'],prior['bright_fixed_center'],rtol=0,atol=1e-12) and
                np.allclose(actual['robust_weight'],[weights[k] for k in actual['detection_ids']],rtol=1e-7,atol=1e-10)):
                verified_keys.add(key)
            if not slots:break
        if slots:raise ValueError('Missing final source in immutable bright selection')
    verified=selected.loc[selected.key.isin(verified_keys)]
    def subset(frame):
        residual=frame[['res_e','res_n']].to_numpy()
        return summarize(residual,frame[['se_e','se_n']].to_numpy())
    return dict(fixed_apparent_predictions=len(control),exact_member_matches=len(matched),
                usable_observed_r_lt21=len(selected),changed_estimator=len(selected)-len(verified),
                verified_identical_estimator=subset(verified),
                single_run=subset(verified.loc[verified.sdss_runs.eq(1)]),
                multiple_runs=subset(verified.loc[verified.sdss_runs.ge(2)]),
                closure_sha256=digest(root/'catalogue/v22_catalogue_ownership_summary.json'),
                scope='Frozen bright Gaia controls with annual parallax in observed epochs. '
                      'These local stellar checks condition on the external positions and measured membership. '
                      'Changed estimators remain in the exit count. Faint Gaia PMs are excluded.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--history-root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.root,a.history_root)
