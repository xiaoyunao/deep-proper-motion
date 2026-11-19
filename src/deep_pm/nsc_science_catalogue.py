"""Continuous all-NSC-band research catalogue, with frozen-quality r control.

No QSO or Gaia motion enters discovery. A research-quality fit is not a certified
star: unknown membership, spatial zero and nightly covariance remain explicit.
"""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import resource
from time import monotonic

import healpy as hp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .allband_blind_year import r_magnitude
from .allband_qso import evaluated_fit,physical_candidates
from .astrometry import unproject
from .astrometry_audit import augment_tracks
from .continuous_reference import PRECISION,SPATIAL,bind,read_offsets,reference_frame,verify_directory
from .dark_precision import query_unique
from .exposure_catalogues import inside
from .exposure_pilot import contrast_samples,matched_legacy
from .independent_summary import standard_block
from .io import save_json,json_default,digest
from .nsc_comparison import qso_matches,comparison
from .nsc_information import information,night_labels,summarize_information
from .pilot_production import make_tracks,quality_detections,source_funnel,valid_epochs
from .precision_v03 import calibration_groups,calibration_support
from .provenance import run_stage
from .public_summary import check_aggregate
from .regional_validation import official_table


def select_catalogue(data,xy,time,sigma,keep,solution,keys,center,spec,config,settings):
    """r seeds stay r seeds; the non-r route cannot borrow an eligible r anchor."""
    prod=settings['production'];nsc=data.survey.eq('NSC').to_numpy();visit=data.exposure_id.to_numpy()
    support=calibration_support(keys,solution);physical=physical_candidates(data)
    core=lambda position:inside(*unproject(position,center)[0],spec)
    # The legacy-quality r control remains untouched. For new all-band products,
    # NSC r must pass the same physical selection used for every other band.
    strict_r=keep&(~nsc|physical);full=strict_r|(nsc&physical&support)
    tracks=make_tracks(xy,visit,keep,prod['match_radius_mas']);seeds=make_tracks(xy,visit,keep&nsc,prod['match_radius_mas'])
    funnel_r,baseline=source_funnel(data,xy,time,sigma,keep,tracks,solution,config,settings,9.,initial=seeds,observation_groups=keys,core_members=core)
    strict_tracks=make_tracks(xy,visit,strict_r,prod['match_radius_mas']);strict_seeds=make_tracks(xy,visit,strict_r&nsc,prod['match_radius_mas'])
    _,strict=source_funnel(data,xy,time,sigma,strict_r,strict_tracks,solution,config,settings,9.,initial=strict_seeds,observation_groups=keys,core_members=core)
    expanded=augment_tracks(strict_tracks,data,xy,full,strict_r,prod)
    funnel_all,products=source_funnel(data,xy,time,sigma,full,expanded,solution,config,settings,9.,initial=strict_seeds,reference_keep=strict_r,observation_groups=keys,core_members=core)
    original_members=[set(v[4]) for v in baseline.values()]
    # Membership overlap, rather than an output row number, identifies controls
    # when physical flags change track numbering.
    old_owner={int(i):j for j,ix in enumerate(original_members) for i in ix}
    candidates=[];reserved=set();counts=Counter()
    for slot,row in products.items():
        index=row[4];votes=Counter(old_owner[i] for i in index if i in old_owner)
        match=votes.most_common(1)[0][0] if votes else None
        kind='r_existing' if match is not None and len(original_members[match]&set(index))>=.6*len(original_members[match]) else 'extra_r_anchor'
        candidates.append(dict(kind=kind,index=index));reserved.update(index)
    tracks_all=make_tracks(xy,visit,full&support,prod['match_radius_mas']);counts['all_filter_tracks']=len(tracks_all)
    for index in tracks_all:
        if not nsc[index].any() or nsc[index].all():counts['non_r_missing_survey']+=1;continue
        if np.any(nsc[index]&data.band.iloc[index].eq('r').to_numpy()):continue
        counts['non_r_before_quality']=counts.get('non_r_before_quality',0)+1
        if any(i in reserved for i in index):counts['non_r_shared_detection_rejected']+=1;continue
        if not valid_epochs(index,nsc,time,prod):counts['non_r_time_exit']+=1;continue
        shape=data.morphology.iloc[index[nsc[index]]].to_numpy(float);shape=shape[np.isfinite(shape)&(shape>=0)&(shape<=1)]
        if not len(shape) or np.median(shape)<.8:counts['non_r_morphology_exit']+=1;continue
        fit=evaluated_fit(index,data,xy,time,sigma,keys,solution,center,config['reference_epoch'])
        if not core(fit['tangent_state'][[0,2]]):counts['non_r_ownership_exit']+=1;continue
        if np.max(fit['pm_error_masyr'])>prod['max_axis_pm_error_masyr']:counts['non_r_formal_error_exit']+=1;continue
        # Same frozen robust residual gate as source_funnel, evaluated in the
        # original tangent axes. Robust downweighting never deletes measurements.
        from .precision_v03 import fit_with_calibration
        model,_=fit_with_calibration(xy[index],time[index],sigma[index],config['reference_epoch'],[solution[k] for k in keys[index]])
        if np.quantile(np.linalg.norm(model.residual/sigma[index],axis=1),.68)>3:counts['non_r_residual_exit']+=1;continue
        candidates.append(dict(kind='non_r_anchor',index=index));reserved.update(index)
    return baseline,candidates,dict(legacy_r=funnel_r,physical_all_r_anchor=funnel_all,non_r=dict(counts),strict_r_solutions=len(strict)),(keep,strict_r,full)


def stationary_controls(work,source,legacy,data,xy,time,sigma,keys,solution,center,spec,config,settings,masks):
    """Same physical QSO identities for r/full and external comparisons."""
    from .astrometry import project
    qso=pd.read_csv(source/'spectra_union.csv');qso=qso.loc[inside(qso.ra,qso.dec,spec)].reset_index(drop=True);points=project(qso.ra,qso.dec,center)
    labels=pd.read_csv(source/'external_labels.csv',dtype=str).set_index('detection_id').original_object_id
    release=labels,official_table(source,{})
    archive=pd.read_csv(legacy);archive=archive.loc[inside(archive.RA,archive.DEC,spec,.06)];old,_=matched_legacy(archive,points,center)
    nsc=data.survey.eq('NSC').to_numpy();zero=np.zeros((len(qso),2));private={};lookup={}
    for name,mask in zip(('r_legacy_quality','r_physical_quality','all_nsc'),masks):
        lab,truth=qso_matches(data,xy,mask,points);blocks=[]
        contrast_samples(truth,lab,data,xy,time,sigma,mask,nsc,settings,config,(keys,),(solution,),center,qso.reference_r.to_numpy(),qso.origin.to_numpy(),zero,zero,release,old,[],{'C0':{}},blocks)
        private[name]=blocks;lookup[name]={r['validation_local_index']:standard_block(r,center) for r in blocks}
    save_json(work/'private_known_qso_blocks.json',private)
    cohorts={}
    for low in (20,21,22):
        for label,select in [('all',lambda r:True),('one_sdss_run',lambda r:len([e for e in r['exposures'] if e.startswith('SDSS:')])==1),('multiple_sdss_runs',lambda r:len([e for e in r['exposures'] if e.startswith('SDSS:')])>1)]:
            rows=[];external=[]
            for k in sorted(set.intersection(*(set(v) for v in lookup.values()))):
                a=lookup['r_legacy_quality'][k]
                if not low<=a['magnitude']<low+1 or not select(a):continue
                row={name:(table[k]['motion']['C0'],table[k]['error_icrs']['C0']) for name,table in lookup.items()};rows.append(row)
                if {'NSC','Xiao'}<=a['motion'].keys():external.append(dict(row,**{m:(a['motion'][m],a['error_icrs'][m]) for m in ('NSC','Xiao')}))
            contrasts=(('all_nsc','r_physical_quality'),('r_physical_quality','r_legacy_quality'),('all_nsc','Xiao'),('all_nsc','NSC'))
            cohorts[f'r{low}__{label}']=dict(same_objects=comparison(rows,settings,contrasts),strict_published_common=comparison(external,settings,contrasts))
    return cohorts,qso,lookup


def run(work,source,shared,spec,legacy):
    start=monotonic();config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());settings.update(bootstrap_draws=400,bootstrap_absolute_zero=True,min_publish_count=5)
    graph=shared/'private_c0.npz';center=reference_frame(graph);solution,gauges=read_offsets(graph,with_gauge=True)
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production']);nsc=data.survey.eq('NSC').to_numpy();keys=calibration_groups(data);night=night_labels(data)
    votes=pd.Series(keys[keep&inside(data.ra_deg,data.dec_deg,spec)]).map(gauges).value_counts();gauge=min(votes.index,key=lambda k:(-votes[k],k)) if len(votes) else None
    solution={k:v for k,v in solution.items() if gauges[k]==gauge}
    baseline,candidates,funnel,masks=select_catalogue(data,xy,time,sigma,keep,solution,keys,center,spec,config,settings)
    rows=[];records=[];used=set();band_counts=Counter();support=calibration_support(keys,solution);graph_hash=digest(graph)
    for row in candidates:
        index=row['index']
        if any(int(i) in used for i in index):raise ValueError('A physical measurement claimed by two all-band sources')
        used.update(map(int,index));mag,origin=r_magnitude(index,data,nsc)
        fit=evaluated_fit(index,data,xy,time,sigma,keys,solution,center,config['reference_epoch'])
        # Information diagnostics only need this source. Avoid rescanning the
        # million-row survey column for every catalogue entry.
        local=data.iloc[index].reset_index(drop=True);ix=np.arange(len(index))
        details=information(ix,ix,local,time[index],sigma[index],night[index],keys[index],fit)
        late=index[nsc[index]];shape=data.morphology.iloc[late].to_numpy(float);flags=[]
        if not np.isfinite(shape).all() or np.any((shape<0)|(shape>1)):flags.append('unknown_or_invalid_morphology')
        if details['nsc_seasons']<3:flags.append('fewer_than_three_nsc_seasons')
        flags.extend(['night_correlation_unqualified','spatial_zero_uncertified','identity_unconfirmed_outside_spectral_audit'])
        bands=sorted(set(data.band.iloc[late]));band_counts.update(bands)
        source_record=dict(kind=row['kind'],magnitude=mag if np.isfinite(mag) else None,magnitude_source=origin,reference_component=gauge,
            reference_epoch=config['reference_epoch'],reference_model_sha256=graph_hash,
            bands=bands,instruments=sorted(set(data.instrument.iloc[late].fillna('unknown').astype(str))),
            detection_ids=data.detection_id.iloc[index].tolist(),physical_groups=keys[index],exposures=data.exposure_id.iloc[index].tolist(),nights_utc18=night[index],
            information=details,quality_flags=flags,**fit)
        records.append(source_record);rows.append(dict(row,fit=fit,magnitude=mag,information=details))
    # The private full catalogue is frozen before any spectral target lookup.
    with gzip.open(work/'private_standard_motion.json.gz','wt') as stream:json.dump(records,stream,default=json_default,allow_nan=False)
    with gzip.open(work/'private_r_control.json.gz','wt') as stream:
        json.dump([dict(detection_ids=data.detection_id.iloc[v[4]].tolist(),**evaluated_fit(v[4],data,xy,time,sigma,keys,solution,center,config['reference_epoch'])) for v in baseline.values()],stream,default=json_default,allow_nan=False)
    known,qso,lookup=stationary_controls(work,source,legacy,data,xy,time,sigma,keys,solution,center,spec,config,settings,masks)
    from .astrometry import project
    points=np.asarray([r['fit']['position_deg'] for r in rows]).reshape(-1,2);qpoints=project(qso.ra,qso.dec,center)
    found,nearest=query_unique(cKDTree(qpoints),project(*points.T,center),500,300) if len(rows) and len(qso) else (np.zeros(len(rows),bool),np.zeros(len(rows),int))
    claims=Counter(nearest[found]);blind=[]
    for j in np.flatnonzero(found):
        k=int(nearest[j])
        if claims[k]!=1:continue
        row=rows[j];members=row['index'];qm=lookup['all_nsc'].get(k)
        blind.append(dict(kind=row['kind'],magnitude=float(qso.reference_r.iloc[k]),values={'blind_all':(row['fit']['pm_icrs_masyr'],row['fit']['pm_error_masyr'])},
            qso_local_index=k,source_slot=int(j),exposures=sorted(set(data.exposure_id.iloc[members])),nsc_epoch_days=sorted(set(night[members[nsc[members]]]))))
        if qm:
            blind[-1]['values'].update({m:(qm['motion'][m],qm['error_icrs'][m]) for m in ('C0','NSC','Xiao') if m in qm['motion']})
    save_json(work/'private_blind_qso.json',blind)
    byclass={};blinds={}
    for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
        for low in (20,21,22):
            part=[r for r in rows if r['kind']==kind and low<=r['magnitude']<low+1]
            byclass[f'{kind}__r{low}']=dict(research_fits=len(part),information=summarize_information([r['information'] for r in part]),
                source_level_band_support=dict(Counter(b for r in part for b in set(data.band.iloc[r['index'][nsc[r['index']]]]))),
                formal_error_p16_p50_p84=np.quantile([r['fit']['pm_error_masyr'] for r in part],[.16,.5,.84],axis=0) if part else None)
            validation=[r for r in blind if r['kind']==kind and low<=r['magnitude']<low+1];common=[r['values'] for r in validation if {'blind_all','C0','NSC','Xiao'}<=r['values'].keys()]
            blinds[f'{kind}__r{low}']=dict(position_identified_qso=len(validation),all_spectral=comparison([{'blind_all':r['values']['blind_all']} for r in validation],settings),
                strict_published_common=comparison(common,settings,(('blind_all','Xiao'),('blind_all','NSC'),('blind_all','C0'))))
    positions=points;pix=hp.ang2pix(128,*positions.T,lonlat=True) if len(rows) else []
    summary=dict(owned_r_control=len(baseline),owned_all_nsc=len(rows),by_class=byclass,blind_spectral=blinds,known_stationary=known,funnel=funnel,
        source_level_band_support=dict(band_counts),nside128_output_counts=dict(Counter(map(str,pix))),
        input_detections=len(data),qualified_nsc_detected=int((physical_candidates(data)&nsc).sum()),qualified_supported_nsc_detected=int((physical_candidates(data)&nsc&support).sum()),
        reference_gauge=gauge,elapsed_seconds=monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Research catalogue only. SDSS published r positions for all real runs; every physically qualified NSC band eligible. Legacy-quality r control and physically filtered r control separate. Independent non-r route cannot include eligible NSC r or share a reserved detection. Same spectral objects compared; all private discovery frozen before spectral lookup. Full ICRS 4x4 and independent/reference tangent components retained. W0 diagonal+50mas does not certify nightly errors or local zero. Counts are research-usable solutions, not purity-certified stars or image completeness. No QSO or faint Gaia motion trains/chooses models; no DCR adopted.')
    check_aggregate(summary);save_json(work/'nsc_science_catalogue_summary.json',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','source','shared','geometry','legacy'):p.add_argument(n,type=Path)
    p.add_argument('region');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'source_receipt.json',verify_directory(a.source,('detections.csv','spectra_union.csv','external_labels.csv','official_nsc.csv')))
    bind(a.work,'reference_receipt.json',verify_directory(a.shared,('private_c0.npz',)));bind(a.work,'geometry.toml',a.geometry);bind(a.work,'legacy.csv',a.legacy);bind(a.work,'production.toml',SPATIAL)
    with run_stage(a.work,PRECISION,a.region,'nsc-science-catalogue-v15',inputs=('source_receipt.json','reference_receipt.json','geometry.toml','legacy.csv','production.toml'),outputs=('nsc_science_catalogue_summary.json','private_standard_motion.json.gz','private_r_control.json.gz','private_known_qso_blocks.json','private_blind_qso.json')) as go:
        if go:run(a.work,a.source,a.shared,tomllib.loads(a.geometry.read_text())['fields'][a.region],a.legacy)


if __name__=='__main__':main()
