"""v2.2 full-sphere, full-2D source fits using unfiltered common reference graphs.

A core is an origin of computation. J2015.5 ownership is reconciled globally
later; a source moving across an internal core boundary is never discarded here.
"""
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
from .allband_blind_year import r_magnitude
from .astrometry import project,seasons
from . import stereographic
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets,reference_frame,verify_directory
from .coordinate_metric import native_covariance,projected_covariance,fit_vector_motion,reference_covariance,AngularTree,old_influence,covariance_of_estimator,design_matrix
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .full_reference_run import local_support
from .io import digest,save_json,json_default
from .nsc_information import night_labels
from .nsc_science_catalogue import select_catalogue
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups
from .public_summary import check_aggregate
from .standard_motion import icrs_motion
from .year_opportunities import utc_year


def information(data,time,night,keys,model,fit,covariance,utc_years=None):
    """Full2D Fisher information after eliminating the two position nuisances."""
    w=np.linalg.inv(covariance)*model.weight[:,None,None]
    jacobian=np.asarray(fit['state_jacobian'])
    transformed_design=design_matrix(time,model.epoch)@np.linalg.inv(jacobian)
    pos=transformed_design[:,:,[0,2]];motion=transformed_design[:,:,[1,3]]
    pp=np.einsum('nip,nij,njq->pq',pos,w,pos)
    pm=np.einsum('nip,nij,njq->pq',pos,w,motion)
    nuisance=np.linalg.solve(pp,pm)
    residual=motion-pos@nuisance
    fisher=residual.transpose(0,2,1)@w@residual
    diagonal=fisher[:,[0,1],[0,1]];total=diagonal.sum(axis=0)
    maximum={};effective={}
    for label,tags in [('physical_group',keys),('observing_night',np.char.add(data.survey.to_numpy(str),night.astype(str)))]:
        part=np.array([diagonal[tags==g].sum(axis=0) for g in sorted(set(tags))]);fraction=part/total
        maximum[label]=fraction.max(axis=0);effective[label]=1/np.sum(fraction**2,axis=0)
    nsc=data.survey.eq('NSC').to_numpy();tlate=time[nsc]
    if utc_years is None:utc_years=utc_year(data.observation_time.to_numpy())
    return dict(detections=len(data),nsc_exposures=data.exposure_id[nsc].nunique(),nsc_nights=len(set(night[nsc])),
        nsc_seasons=len(set(seasons(tlate,90))),nsc_span_years=float(np.ptp(tlate)),joint_span_years=float(np.ptp(time)),
        sdss_physical_runs=data.exposure_id[~nsc].nunique(),
        sdss_unique_UTC_years=len(set(utc_years[~nsc])),nsc_unique_UTC_years=len(set(utc_years[nsc])),
        UTC_year_scope='Distinct calendar years; independent long-term support additionally requires actual span/night/season/leverage.',
        full2D_profiled_ICRS_slope_information=total,largest_information_fraction=maximum,effective_information_groups=effective,
        independent_pm_variance=np.diag(jacobian@fit['independent_tangent_covariance']@jacobian.T)[[1,3]],
        reference_pm_variance=np.diag(jacobian@fit['reference_tangent_covariance']@jacobian.T)[[1,3]],
        leverage_definition='Full2D GLS profiled Fisher contribution, localICRS slope axes; conditional W0, not independent-night information')


def write_rows(path,rows):
    pending=path.with_suffix('.pending')
    with gzip.open(pending,'wt') as f:json.dump(rows,f,default=json_default,allow_nan=False)
    pending.replace(path)


def saved_reference_response(row,data,time,sigma):
    """Recover the saved estimator's synchronous state draws without refitting.

    Only spectral validation rows save these large arrays. All science rows
    keep member identities, estimator weights and the full covariance, so the
    same responses remain recoverable from signed raw inputs.
    """
    ix=pd.Index(data.detection_id).get_indexer(row['detection_ids'])
    if np.any(ix<0):raise ValueError('Saved fit lost its physical measurement')
    center=stereographic.from_record(row);cp=projected_covariance(data.ra_deg.iloc[ix],data.dec_deg.iloc[ix],center,native_covariance(sigma[ix]))
    w=np.linalg.inv(cp)*np.asarray(row['robust_weight'])[:,None,None]
    design=design_matrix(time[ix],row['reference_epoch'])
    h=np.einsum('pq,qni->pni',np.asarray(row['independent_tangent_covariance']),np.einsum('nip,nij->pnj',design,w))
    # Find the exact versioned graph by component; never realign unrelated
    # reference components as if their draw labels represented the same noise.
    return h,ix


def external_values(source,oldfit,spectra,center):
    """Published ground PMs for same spectral positions; no calibration use."""
    result={};goals=project(spectra.ra,spectra.dec,center)
    for method,path in [('NSC',source/'official_nsc.csv'),('Xiao',oldfit/'legacy.csv')]:
        frame=pd.read_csv(path,dtype={'id':str}).rename(columns=str.lower)
        if method=='Xiao':frame=frame.loc[frame.pm_source.astype(str).str.strip().eq('DESI')]
        frame=frame.loc[np.isfinite(frame[['ra','dec']]).all(axis=1)].reset_index(drop=True)
        if frame.empty:continue
        tree=AngularTree(project(frame.ra,frame.dec,center),center)
        found,near=query_unique(tree,goals,1000,300)
        for j in np.flatnonzero(found):
            row=frame.iloc[near[j]]
            err=row[['pmraerr','pmdecerr'] if method=='NSC' else ['pmra_err','pmdec_err']].to_numpy(float)
            pm=row[['pmra','pmdec']].to_numpy(float)
            if method=='NSC' and (not np.isfinite(float(row.deltamjd)) or float(row.deltamjd)<=0):continue
            if np.isfinite(pm).all() and abs(pm).max()<1e5 and np.isfinite(err).all() and (err>0).all():result.setdefault(int(j),{})[method]=(pm,err)
    return result


@lru_cache(maxsize=8)
def reference_context(graph):
    graph=Path(graph);sha=digest(graph)
    center=reference_frame(graph);solution=read_offsets(graph);names=np.asarray(sorted(solution))
    draws=np.stack([solution[k][1] for k in names])
    return center,solution,names,draws,sha


@lru_cache(maxsize=4)
def component_map(root):
    path=Path(root)/'graph/private_group_components.parquet'
    frame=pd.read_parquet(path)
    meta_path=Path(root)/'graph/v22_global_graph_summary.json';meta=read(meta_path)
    ready={r['component'] for r in meta['components'] if r['status']=='ready'}
    frame=frame.loc[frame.component.isin(ready)]
    from hashlib import sha256
    signature=sha256((digest(path)+digest(meta_path)).encode()).hexdigest()
    return frame.set_index('group').component,signature


# Reviewed TAN implementation in 0320f59; only the wide-chart branch,
# post-freeze validation and retention of control rows changed for v2.4.
ARCHIVED_TAN = {
    'implementation_sha256': 'f7916c4cbec7f8219035791ef52a66a1224d058147628a7e0aa9aaeee9b40513',
    'metric_sha256': '4565ddf93372c55e823093b6e2e21a7c4315cd7e80f40a0835b862f0dce46e00',
    'support_sha256': '2e6c273a565b022a11b8272398a6d5f9e575fce91ec5498300ab1fe5dfaf391f',
    'standard_output_sha256': 'bd72e9ef7058f04a1bfccca64a14094233ba9e562f63758a16d26fb72af0ded0',
}


def reuse_archived_source(entry,ticket,dest):
    prior=Path(entry['predecessor'])
    receipt=prior/'v22_source_core_summary.json'
    original=read(receipt)
    old=original['ticket']
    keys=('graphs','source_receipt_sha256','precision_sha256','spatial_sha256',
          'selection_sha256','funnel_sha256','method')
    if any(old.get(k)!=ticket[k] for k in keys) or any(old.get(k)!=v for k,v in ARCHIVED_TAN.items()):
        return None
    proof=prior.parent.parent/'catalogue/index'/entry['core']/'private_index_receipt.json'
    archived=digest(receipt)
    if read(proof)['ticket']['source_summary_sha256']!=archived:
        raise ValueError('Frozen ownership index and archived source version disagree')
    start=monotonic()
    files={}
    for name in original['private_output_sha256']:
        target=dest/name
        if name=='private_standard_motion.json.gz' or (name=='private_r_control.json.gz' and ticket['save_r_comparison']):
            src=(prior/name).resolve(strict=True)
            sha=original['private_output_sha256'][name]
            if digest(src)!=sha:raise ValueError('Changed verified predecessor science rows')
            if target.is_symlink() and target.resolve(strict=True)==src:pass
            elif target.exists() or target.is_symlink():raise ValueError('Would replace an existing source output')
            else:target.symlink_to(src)
            files[name]=sha
        else:
            # Spectroscopic validation is performed after NEW ownership freezes.
            if target.is_symlink():raise ValueError('Unsafe link on a new empty evaluation product')
            if target.suffix=='.gz':write_rows(target,[])
            else:save_json(target,[])
            files[name]=digest(target)
    result=dict(original,ticket=ticket,fixed_blind_qso_entries=0,fixed_qso_status={},new_blind_spectral_matches=0,
                private_output_sha256=files,reused_archived_source_sha256=archived,
                elapsed_seconds=monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                slurm_job=os.environ['SLURM_JOB_ID'],scope='Same signed input, all actual calibrated group models and frozen TAN source method. Blind scientific origin rows reused; ownership and validation evaluated on the unified catalogue.')
    save_json(dest/'v22_source_core_summary.json',result)
    return result


def one(task):
    entry,root,components=task;root=Path(root);dest=root/'sources'/entry['core'];dest.mkdir(parents=True,exist_ok=True)
    source=Path(entry['source']);oldfit=Path(entry['fit']) if entry.get('fit') else None;config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    loaded=None
    if components is None:
        mapping,mapping_sha=component_map(str(root));source_sha=digest(source/'run_manifest.json')
        dispatch=dest/'private_dispatch.json'
        if dispatch.exists():
            registered=read(dispatch)
            if registered['source_sha256']!=source_sha or registered['group_map_sha256']!=mapping_sha:raise ValueError('Changed component dispatch inputs')
            components=registered['components']
        else:
            loaded=quality_detections(source,entry['center'],config,settings['production'])
            target_groups=calibration_groups(loaded[0])
            components=sorted(set(mapping.reindex(pd.unique(target_groups)).dropna().astype(int)))
            save_json(dispatch,dict(source_sha256=source_sha,group_map_sha256=mapping_sha,components=components,
                definition='Actual target physical groups, including those with no local reference galaxies. Not restricted to local reference entry.'))
    ticket=dict(source_receipt_sha256=digest(source/'run_manifest.json'),graphs={str(c):reference_context(str(root/'reference'/f'component-{c:04d}'/'private_c0_unfiltered.npz'))[4] for c in components},
        implementation_sha256=digest(Path(__file__)),metric_sha256=digest(Path(__file__).with_name('coordinate_metric.py')),
        selection_sha256=digest(Path(__file__).with_name('nsc_science_catalogue.py')),funnel_sha256=digest(Path(__file__).with_name('pilot_production.py')),
        support_sha256=digest(Path(__file__).with_name('full_reference_run.py')),
        standard_output_sha256=digest(Path(__file__).with_name('standard_motion.py')),
        precision_sha256=digest(PRECISION),spatial_sha256=digest(SPATIAL),method='full-coordinate-v22-native50mas-full2D-sphere-nativeSE12')
    production=entry.get('production_blind',False)
    if production:ticket['chart_sha256']=digest(Path(stereographic.__file__))
    if production:
        ticket['spectral_evaluation']='after-global-closure'
        ticket['save_r_comparison']=entry.get('save_r_comparison',False)
    path=dest/'v22_source_core_summary.json'
    if path.exists():
        result=read(path)
        if result['ticket']!=ticket:raise ValueError('Changed source fit checkpoint')
        for name,sha in result['private_output_sha256'].items():
            if digest(dest/name)!=sha:raise ValueError('Changed fitted catalogue')
        return result
    if production and entry.get('predecessor'):
        reused=reuse_archived_source(entry,ticket,dest)
        if reused is not None:return reused
    start=monotonic();verify_directory(source,('detections.csv','spectra_union.csv'))
    records=[];controls=[];fixed=[];support_counts={};funnels={};used=set()
    oldrows=[] if production else read(oldfit/'private_standard_motion.json.gz')
    oldblind=[] if production else read(oldfit/'private_blind_qso.json')
    if not components:
        fixed=[dict(core=entry['core'],source_slot=int(r['source_slot']),qso_local_index=int(r['qso_local_index']),magnitude=r['magnitude'],old_values=r['values'],status='no_solved_reference_component') for r in oldblind]
    b1_by_slot={}
    if oldblind:
        if loaded is None:loaded=quality_detections(source,entry['center'],config,settings['production'])
        original_data,_,original_time,original_sigma,_=loaded
        original_lookup=pd.Index(original_data.detection_id)
        for target in oldblind:
            slot=int(target['source_slot']);old=oldrows[slot];ix=original_lookup.get_indexer(old['detection_ids'])
            if (ix<0).any():continue
            h=old_influence(old,original_time[ix],original_sigma[ix])
            weight=np.asarray(old['robust_weight'])
            reproduced=covariance_of_estimator(h,native_covariance(original_sigma[ix]),weight)
            np.testing.assert_allclose(reproduced,old['independent_tangent_covariance'],rtol=2e-8,atol=2e-6)
            cp_old=projected_covariance(original_data.ra_deg.iloc[ix],original_data.dec_deg.iloc[ix],entry['center'],native_covariance(original_sigma[ix]))
            independent=covariance_of_estimator(h,cp_old,weight);j=np.asarray(old['state_jacobian'])
            covariance=j@(independent+np.asarray(old['reference_tangent_covariance']))@j.T
            b1_by_slot[slot]=dict(B1_values=(old['pm_icrs_masyr'],np.sqrt(np.diag(covariance)[[1,3]])),B1_covariance=covariance,
                B1_definition='Frozen old point/members/IRLS/H/J and archived old reference common term; native covariance including floor projected in original plane; C/w conditional noise')
    for item in fixed:item.update(b1_by_slot.get(item['source_slot'],{}))
    # Recompute every original blind identity, even if the new quality gate fails.
    # Final old-owner/exterior restrictions are applied to its frozen identity
    # pool in the global scientific readout, never by new quality selection.
    physical_count=legacy_count=0
    for component in components:
        graph=root/'reference'/f'component-{component:04d}'/'private_c0_unfiltered.npz'
        center,solution,names,draws,graph_sha=reference_context(str(graph))
        if loaded is None:loaded=quality_detections(source,entry['center'],config,settings['production'])
        data,_,t,sigma,keep=loaded
        xy=project(data.ra_deg,data.dec_deg,center)
        keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy();night=night_labels(data);year=utc_year(data.observation_time.to_numpy())
        cp=projected_covariance(data.ra_deg,data.dec_deg,center,native_covariance(sigma));inverse=np.linalg.inv(cp)
        support,se=local_support(data.ra_deg,data.dec_deg,center,keys,names,draws)
        core=inside(data.ra_deg,data.dec_deg,entry['spec']);on=core&pd.Series(keys).isin(names).to_numpy()
        support_counts[str(component)]=dict(core_group_present=int(on.sum()),core_local12mas_supported=int((core&support).sum()),
            core_native_se_p16_p50_p84=np.quantile(se[on],[.16,.5,.84],axis=0) if on.any() else None)
        @lru_cache(maxsize=32768)
        def cached(members):
            ix=np.asarray(members,int);zero=np.array([solution[g][0] for g in keys[ix]])
            model=fit_vector_motion(xy[ix]-zero,t[ix],cp[ix],config['reference_epoch'])
            common,response=reference_covariance(model.influence,np.stack([solution[g][1] for g in keys[ix]],axis=1))
            fit=icrs_motion(model.coeff[0],model.coeff[1],center,model.cov+common)
            if isinstance(center,stereographic.Stereographic):
                fit['projection_kind']='stereographic'
            fit.update(reference_epoch=config['reference_epoch'],tangent_state=model.coeff.T.ravel(),
                independent_tangent_covariance=model.cov,reference_tangent_covariance=common,
                robust_weight=model.weight,span_years=float(np.ptp(t[ix])))
            j=np.asarray(fit['state_jacobian'])
            fit.update(independent_icrs_covariance=j@model.cov@j.T,reference_icrs_covariance=j@common@j.T)
            return model,fit,response
        def fit_source(ix):
            model,fit,_=cached(tuple(ix));return model,fit['pm_error_masyr']
        def output_fit(ix):return cached(tuple(ix))[1]
        def distance(ix,model):return np.sqrt(np.einsum('ni,nij,nj->n',model.residual,inverse[ix],model.residual))
        control_maps={}
        baseline,candidates,funnel,_=select_catalogue(data,xy,t,sigma,keep,solution,keys,center,entry['spec'],config,settings,
            fit_source,output_fit,sky_center=center,quality_distance=distance,calibration_keep=support,control_maps=control_maps)
        funnels[str(component)]=funnel
        for candidate in candidates:
            ix=candidate['index'];ids=data.detection_id.iloc[ix].astype(str).tolist()
            if used.intersection(ids):raise ValueError('Cross-component detection reuse')
            used.update(ids);model,fit,response=cached(tuple(ix));local=data.iloc[ix].reset_index(drop=True)
            info=information(local,t[ix],night[ix],keys[ix],model,fit,cp[ix],year[ix])
            mag,origin=r_magnitude(ix,data,nsc)
            flags=['spatial_zero_uncertified','night_correlation_unqualified','identity_unconfirmed_outside_spectral_audit']
            shape=local.morphology[local.survey.eq('NSC')].to_numpy(float)
            if not np.isfinite(shape).all() or np.any((shape<0)|(shape>1)):flags.append('unknown_or_invalid_morphology')
            if candidate['kind']=='non_r_anchor':flags.append('non_r_anchor_requires_identity_validation')
            if max(info['largest_information_fraction']['physical_group'])>=.8:flags.append('one_group_supplies_at_least_80pct_leverage')
            if info['nsc_seasons']<3:flags.append('fewer_than_three_nsc_seasons')
            records.append(dict(kind=candidate['kind'],magnitude=float(mag) if np.isfinite(mag) else None,magnitude_source=origin,
                reference_component=str(component),reference_model_sha256=graph_sha,projection_center_deg=center,
                physical_groups=keys[ix].tolist(),exposures=data.exposure_id.iloc[ix].tolist(),nights_utc18=night[ix],
                detector_ids=data.detector_id.iloc[ix].astype(str).tolist(),
                observation_mjd_tt=data.observation_time.iloc[ix].tolist(),time_scale='tt',
                bands_per_detection=data.band.iloc[ix].astype(str).tolist(),instruments_per_detection=data.instrument.iloc[ix].fillna('unknown').astype(str).tolist(),
                field_ids=[str(v) if pd.notna(v) else None for v in data.field_id.iloc[ix]] if 'field_id' in data else [None]*len(ix),
                bands=sorted(set(local.band[local.survey.eq('NSC')])),instruments=sorted(set(local.instrument[local.survey.eq('NSC')].fillna('unknown'))),
                detection_ids=ids,information=info,quality_flags=flags,native_reference_se_mas=se[ix],**fit))
        legacy_count+=len(control_maps['legacy'])
        physical_count+=len(control_maps['physical'])
        for control_kind,table in control_maps.items():
            if production and not entry.get('save_r_comparison',False):continue
            for row in table.values():
                ix=row[4];controls.append(dict(control_kind=control_kind,detection_ids=data.detection_id.iloc[ix].tolist(),reference_component=str(component),**output_fit(ix)))
        lookup=pd.Index(data.detection_id)
        for target in oldblind:
            slot=int(target['source_slot']);before=oldrows[slot];ix=lookup.get_indexer(before['detection_ids'])
            result=dict(core=entry['core'],source_slot=slot,qso_local_index=int(target['qso_local_index']),magnitude=target['magnitude'],
                old_values=target['values'],component=str(component),old_detection_ids=before['detection_ids'],**b1_by_slot.get(slot,{}))
            if (ix<0).any():result['status']='missing_normalised_member'
            elif any(g not in solution for g in keys[ix]):result['status']='unsolved_reference_group'
            else:
                model,fit,response=cached(tuple(ix));resid=distance(ix,model)
                result.update(status='computed',new_values=(fit['pm_icrs_masyr'],fit['pm_error_masyr']),new_fit=fit,
                    local_support_pass=bool(support[ix].all()),formal_error_pass=bool(max(fit['pm_error_masyr'])<=10),residual_pass=bool(np.quantile(resid,.68)<=3),
                    synchronous_reference_state_draws=response,reference_model_sha256=graph_sha)
            fixed.append(result)
        cached.cache_clear()
    # Catalogue is frozen before any new spectral lookup.
    write_rows(dest/'private_standard_motion.json.gz',records);write_rows(dest/'private_r_control.json.gz',controls)
    write_rows(dest/'private_fixed_blind_qso.json.gz',fixed)
    spectra=pd.DataFrame() if production else pd.read_csv(source/'spectra_union.csv');blind=[]
    if len(spectra) and records:
        # A compact common local chart is only an indexing convenience; all
        # target/source distances are spherical. Not used for PM fitting.
        center=entry['center'];goals=project(spectra.ra,spectra.dec,center)
        points=project(*np.asarray([r['position_deg'] for r in records]).T,center)
        found,near=query_unique(AngularTree(goals,center),points,500,300);claims=Counter(near[found])
        published=external_values(source,oldfit,spectra,center)
        for tag in ('legacy','physical'):
            selected_controls=[r for r in controls if r['control_kind']==tag]
            cpoints=project(*np.asarray([r['position_deg'] for r in selected_controls]).T,center) if selected_controls else np.empty((0,2))
            cf,cn=query_unique(AngularTree(goals,center),cpoints,500,300);cc=Counter(cn[cf])
            for cj in np.flatnonzero(cf):
                ck=int(cn[cj])
                if cc[ck]==1:
                    c=selected_controls[cj];published.setdefault(ck,{})['new_'+tag+'_r']=(c['pm_icrs_masyr'],c['pm_error_masyr'])
        for j in np.flatnonzero(found):
            k=int(near[j])
            if claims[k]!=1:continue
            qso=spectra.iloc[k];row=records[j];rmag=float(qso.reference_r)
            h,ix=saved_reference_response(row,data,t,sigma)
            graph=root/'reference'/f"component-{int(row['reference_component']):04d}"/'private_c0_unfiltered.npz'
            _,reference,_,_,_=reference_context(str(graph))
            state_draws=-np.einsum('pni,dni->dp',h,np.stack([reference[g][1] for g in keys[ix]],axis=1))
            native_draws=state_draws@np.asarray(row['state_jacobian']).T
            np.testing.assert_allclose(np.cov(native_draws,rowvar=False),row['reference_icrs_covariance'],rtol=2e-7,atol=2e-7)
            blind.append(dict(identity=(str(qso.origin),float(qso.ra),float(qso.dec)),source_slot=int(j),core=entry['core'],
                magnitude=rmag if np.isfinite(rmag) else None,kind=row['kind'],information=row['information'],bands=row['bands'],instruments=row['instruments'],
                reference_component=row['reference_component'],reference_model_sha256=row['reference_model_sha256'],
                synchronous_reference_icrs_state_draws=native_draws,values=dict(new_all=(row['pm_icrs_masyr'],row['pm_error_masyr']),**published.get(k,{}))))
    save_json(dest/'private_blind_qso.json',blind)
    files=['private_standard_motion.json.gz','private_r_control.json.gz','private_fixed_blind_qso.json.gz','private_blind_qso.json']
    result=dict(ticket=ticket,core=entry['core'],window=entry['window'],research_solutions=len(records),r_control_solutions=legacy_count,physical_r_control_solutions=physical_count,
        by_class=dict(Counter(r['kind'] for r in records)),support=support_counts,funnel=funnels,
        fixed_blind_qso_entries=len(fixed),fixed_qso_status=dict(Counter(r['status'] for r in fixed)),new_blind_spectral_matches=len(blind),
        private_output_sha256={name:digest(dest/name) for name in files},elapsed_seconds=monotonic()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Origin-core output, not final owner count. Sphere association, physical native12mas per-target support, full2D GLS and synchronous graph covariance. Fixed old QSO quality failures retained. Global source/detection/owner/spectral uniqueness must close before final scientific statistics.')
    check_aggregate(result);save_json(path,result);print('FULL_COORDINATE_SOURCE',entry['core'],len(records),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--core',action='append');p.add_argument('--workers',type=int,default=2);p.add_argument('--component',action='append',type=int)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    plan=read(args.root/'registration/private_full_coordinate_plan.json')
    # Dispatch from actual target groups, including restored regions without
    # local reference galaxies. Local12mas support remains per measurement.
    selected=[r for r in plan['cores'] if args.core is None or r['core'] in args.core]
    tasks=[(entry,str(args.root),args.component) for entry in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(one,tasks,chunksize=1):print('SOURCE_CHECKPOINT',result['core'],flush=True)
