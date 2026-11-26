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
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets,reference_frame,verify_directory
from .coordinate_metric import native_covariance,projected_covariance,fit_vector_motion,reference_covariance,AngularTree
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
    dt=time-time.mean();w=np.linalg.inv(covariance)*model.weight[:,None,None]
    pp=w.sum(axis=0);pm=np.einsum('n,nij->ij',dt,w)
    mean=np.linalg.solve(pp,pm)
    residual=dt[:,None,None]*np.eye(2)-mean
    fisher=residual.transpose(0,2,1)@w@residual
    velocity_jac=np.asarray(fit['state_jacobian'])[np.ix_([1,3],[1,3])]
    inv=np.linalg.inv(velocity_jac);fisher=inv.T[None]@fisher@inv[None]
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
        sdss_independent_UTC_years=len(set(utc_years[~nsc])),nsc_independent_UTC_years=len(set(utc_years[nsc])),
        full2D_profiled_ICRS_slope_information=total,largest_information_fraction=maximum,effective_information_groups=effective,
        independent_pm_variance=np.diag(fit['independent_tangent_covariance'])[[1,3]],
        reference_pm_variance=np.diag(fit['reference_tangent_covariance'])[[1,3]],
        leverage_definition='Full2D GLS profiled Fisher contribution, localICRS slope axes; conditional W0, not independent-night information')


def write_rows(path,rows):
    pending=path.with_suffix('.pending')
    with gzip.open(pending,'wt') as f:json.dump(rows,f,default=json_default,allow_nan=False)
    pending.replace(path)


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
            if np.isfinite(pm).all() and np.isfinite(err).all() and (err>0).all():result.setdefault(int(j),{})[method]=(pm,err)
    return result


def one(task):
    entry,root,components=task;root=Path(root);dest=root/'sources'/entry['core'];dest.mkdir(parents=True,exist_ok=True)
    source=Path(entry['source']);oldfit=Path(entry['fit']);config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    ticket=dict(source_receipt_sha256=digest(source/'run_manifest.json'),graphs={str(c):digest(root/'reference'/f'component-{c:04d}'/'private_c0_unfiltered.npz') for c in components},
        implementation_sha256=digest(Path(__file__)),metric_sha256=digest(Path(__file__).with_name('coordinate_metric.py')),
        selection_sha256=digest(Path(__file__).with_name('nsc_science_catalogue.py')),funnel_sha256=digest(Path(__file__).with_name('pilot_production.py')))
    path=dest/'v22_source_core_summary.json'
    if path.exists():
        result=read(path)
        if result['ticket']!=ticket:raise ValueError('Changed source fit checkpoint')
        for name,sha in result['private_output_sha256'].items():
            if digest(dest/name)!=sha:raise ValueError('Changed fitted catalogue')
        return result
    start=monotonic();verify_directory(source,('detections.csv','spectra_union.csv'))
    records=[];controls=[];fixed=[];support_counts={};funnels={};used=set()
    oldrows=read(oldfit/'private_standard_motion.json.gz');oldblind=read(oldfit/'private_blind_qso.json')
    # Recompute every original blind identity, even if the new quality gate fails.
    # Final old-owner/exterior restrictions are applied to its frozen identity
    # pool in the global scientific readout, never by new quality selection.
    for component in components:
        graph=root/'reference'/f'component-{component:04d}'/'private_c0_unfiltered.npz';center=reference_frame(graph)
        solution=read_offsets(graph);names=np.asarray(sorted(solution))
        draws=np.stack([solution[k][1] for k in names])
        data,xy,t,sigma,keep=quality_detections(source,center,config,settings['production'])
        keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy();night=night_labels(data);year=utc_year(data.observation_time.to_numpy())
        cp=projected_covariance(data.ra_deg,data.dec_deg,center,native_covariance(sigma));inverse=np.linalg.inv(cp)
        support,se=local_support(data.ra_deg,data.dec_deg,center,keys,names,draws)
        core=inside(data.ra_deg,data.dec_deg,entry['spec']);on=core&np.isin(keys,names)
        support_counts[str(component)]=dict(core_group_present=int(on.sum()),core_local12mas_supported=int((core&support).sum()),
            core_native_se_p16_p50_p84=np.quantile(se[on],[.16,.5,.84],axis=0) if on.any() else None)
        @lru_cache(maxsize=32768)
        def cached(members):
            ix=np.asarray(members,int);zero=np.array([solution[g][0] for g in keys[ix]])
            model=fit_vector_motion(xy[ix]-zero,t[ix],cp[ix],config['reference_epoch'])
            common,response=reference_covariance(model.influence,np.stack([solution[g][1] for g in keys[ix]],axis=1))
            fit=icrs_motion(model.coeff[0],model.coeff[1],center,model.cov+common)
            fit.update(reference_epoch=config['reference_epoch'],tangent_state=model.coeff.T.ravel(),
                independent_tangent_covariance=model.cov,reference_tangent_covariance=common,
                robust_weight=model.weight,span_years=float(np.ptp(t[ix])))
            return model,fit,response
        def fit_source(ix):
            model,fit,_=cached(tuple(ix));return model,fit['pm_error_masyr']
        def output_fit(ix):return cached(tuple(ix))[1]
        def distance(ix,model):return np.sqrt(np.einsum('ni,nij,nj->n',model.residual,inverse[ix],model.residual))
        control_maps={}
        baseline,candidates,funnel,_=select_catalogue(data,xy,t,sigma,keep,solution,keys,center,entry['spec'],config,settings,
            fit_source,output_fit,sky_center=center,quality_distance=distance,calibration_keep=support,control_maps=control_maps)
        funnels[str(component)]=funnel;graph_sha=digest(graph)
        for candidate in candidates:
            ix=candidate['index'];ids=data.detection_id.iloc[ix].astype(str).tolist()
            if used.intersection(ids):raise ValueError('Cross-component detection reuse')
            used.update(ids);model,fit,response=cached(tuple(ix));local=data.iloc[ix].reset_index(drop=True)
            info=information(local,t[ix],night[ix],keys[ix],model,fit,cp[ix],year[ix])
            mag,origin=r_magnitude(ix,data,nsc)
            flags=['spatial_zero_uncertified','night_correlation_unqualified','identity_unconfirmed_outside_spectral_audit']
            if candidate['kind']=='non_r_anchor':flags.append('non_r_anchor_requires_identity_validation')
            if max(info['largest_information_fraction']['physical_group'])>=.8:flags.append('one_group_supplies_at_least_80pct_leverage')
            if info['nsc_seasons']<3:flags.append('fewer_than_three_nsc_seasons')
            records.append(dict(kind=candidate['kind'],magnitude=float(mag) if np.isfinite(mag) else None,magnitude_source=origin,
                reference_component=str(component),reference_model_sha256=graph_sha,projection_center_deg=center,
                physical_groups=keys[ix].tolist(),exposures=data.exposure_id.iloc[ix].tolist(),nights_utc18=night[ix],
                bands=sorted(set(local.band[local.survey.eq('NSC')])),instruments=sorted(set(local.instrument[local.survey.eq('NSC')].fillna('unknown'))),
                detection_ids=ids,information=info,quality_flags=flags,native_reference_se_mas=se[ix],**fit))
        for control_kind,table in control_maps.items():
            for row in table.values():
                ix=row[4];controls.append(dict(control_kind=control_kind,detection_ids=data.detection_id.iloc[ix].tolist(),reference_component=str(component),**output_fit(ix)))
        lookup=pd.Index(data.detection_id)
        for target in oldblind:
            slot=int(target['source_slot']);before=oldrows[slot];ix=lookup.get_indexer(before['detection_ids'])
            result=dict(core=entry['core'],source_slot=slot,qso_local_index=int(target['qso_local_index']),magnitude=target['magnitude'],
                old_values=target['values'],component=str(component),old_detection_ids=before['detection_ids'])
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
    spectra=pd.read_csv(source/'spectra_union.csv');blind=[]
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
            blind.append(dict(identity=(str(qso.origin),float(qso.ra),float(qso.dec)),source_slot=int(j),core=entry['core'],
                magnitude=rmag if np.isfinite(rmag) else None,kind=row['kind'],information=row['information'],bands=row['bands'],instruments=row['instruments'],
                reference_component=row['reference_component'],values=dict(new_all=(row['pm_icrs_masyr'],row['pm_error_masyr']),**published.get(k,{}))))
    save_json(dest/'private_blind_qso.json',blind)
    files=['private_standard_motion.json.gz','private_r_control.json.gz','private_fixed_blind_qso.json.gz','private_blind_qso.json']
    result=dict(ticket=ticket,core=entry['core'],window=entry['window'],research_solutions=len(records),r_control_solutions=sum(r['control_kind']=='legacy' for r in controls),physical_r_control_solutions=sum(r['control_kind']=='physical' for r in controls),
        by_class=dict(Counter(r['kind'] for r in records)),support=support_counts,funnel=funnels,
        fixed_blind_qso_entries=len(fixed),fixed_qso_status=dict(Counter(r['status'] for r in fixed)),new_blind_spectral_matches=len(blind),
        private_output_sha256={name:digest(dest/name) for name in files},elapsed_seconds=monotonic()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Origin-core output, not final owner count. Sphere association, physical native12mas per-target support, full2D GLS and synchronous graph covariance. Fixed old QSO quality failures retained. Global source/detection/owner/spectral uniqueness must close before final scientific statistics.')
    check_aggregate(result);save_json(path,result);print('FULL_COORDINATE_SOURCE',entry['core'],len(records),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--core',action='append');p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    plan=read(args.root/'registration/private_full_coordinate_plan.json');mapping=pd.read_parquet(args.root/'graph/private_group_components.parquet')
    # Domain intersection is a dispatch superset. Per-detection membership and
    # native-SE checks in each complete graph decide actual application support.
    selected=[r for r in plan['cores'] if args.core is None or r['core'] in args.core]
    tasks=[]
    for entry in selected:
        shard=pd.read_parquet(args.root/'input'/entry['core']/'reference.parquet',columns=['group'])
        components=sorted(set(mapping.loc[mapping['group'].isin(shard['group']),'component'].astype(int)))
        tasks.append((entry,str(args.root),components))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(one,tasks,chunksize=1):print('SOURCE_CHECKPOINT',result['core'],flush=True)
