"""Replay fixed core selection; retain object-level losses only on PScratch."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .allband_qso import evaluated_fit,physical_candidates
from .astrometry import project
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets,reference_frame,verify_directory
from .exposure_catalogues import inside
from .io import digest,save_json,json_default
from .nsc_science_catalogue import select_catalogue
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups,calibration_support,fit_with_calibration
from .production_replay_readout import spectral_opportunity
from .public_summary import check_aggregate
from .selection_semantics import morphology_state


ORDER=('outside_core','pointlike_morphology','no_sdss_position','anchor_competition','time_baseline','m3_support','quality','final')


def one(task):
    row,destination,footprint=task
    work=Path(destination)/row['core'];work.mkdir(parents=True,exist_ok=True)
    folder,source,shared=map(Path,(row['fit'],row['source'],row['graph']))
    output=work/'v20_selection_core_summary.json'
    ticket=dict(input_sha256=digest(source/'run_manifest.json'),fit_sha256=digest(folder/'run_manifest.json'),
                reference_sha256=digest(shared/'private_c0.npz'),implementation_sha256=digest(Path(__file__)))
    if output.exists():
        old=read(output)
        if old['ticket']!=ticket:raise ValueError('Selection cache does not match frozen inputs')
        return old
    start=monotonic()
    verify_directory(source,('detections.csv','spectra_union.csv'))
    verify_directory(folder,('private_standard_motion.json.gz','private_blind_qso.json'))
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text())
    spec=tomllib.loads(Path(row['geometry']).read_text())['fields'][row['core']]
    center=reference_frame(shared/'private_c0.npz')
    solution,gauges=read_offsets(shared/'private_c0.npz',with_gauge=True)
    solution={k:v for k,v in solution.items() if gauges[k]==row['reference_gauge']}
    data,xy,time,sigma,keep=quality_detections(source,center,config,settings['production'])
    keys=calibration_groups(data);nsc=data.survey.eq('NSC').to_numpy()
    @lru_cache(maxsize=32768)
    def solve(members):
        ix=np.asarray(members,int)
        return fit_with_calibration(xy[ix],time[ix],sigma[ix],config['reference_epoch'],[solution[k] for k in keys[ix]])
    def fit(ix):return solve(tuple(ix))
    def final(ix):return evaluated_fit(ix,data,xy,time,sigma,keys,solution,center,config['reference_epoch'],model=fit(ix)[0])
    audit={}
    baseline,candidates,funnel,masks=select_catalogue(data,xy,time,sigma,keep,solution,keys,center,spec,config,settings,fit,final,selection_audit=audit)
    frozen=read(folder/'private_standard_motion.json.gz')
    expected={tuple(r['detection_ids']):r for r in frozen}
    actual={tuple(data.detection_id.iloc[r['index']]):r for r in candidates}
    if set(expected)!=set(actual) or len(expected)!=len(frozen):raise ValueError('Trace changed frozen physical members')
    original=read(folder/'nsc_science_catalogue_summary.json')
    if funnel!=original['funnel']:raise ValueError('Trace changed frozen selection counters')
    summary=dict(ticket=ticket,window=row['window'],core=row['core'],input_detections=len(data),
                 exact_source_members_replayed=True,exact_funnel_replayed=True,science_sources=len(frozen),
                 funnel=funnel,morphology={},source_morphology={},qso={},scope='Sixteen metadata-selected core replay. Input/anchor morphology is independent of known-position-control fit success. Original C0/W0 selection and scientific outputs remain frozen.')
    for route,entries in audit.items():
        active=[e for e in entries if e['status']!='outside_core' and 'morphology' in e]
        summary['morphology'][route]=dict(anchors=len(active),states=dict(Counter(e['morphology']['state'] for e in active)),
            legacy_pass_valid_only_fail=sum(e['morphology']['legacy_pass'] and not e['morphology']['valid_only_pass'] for e in active),
            by_exit={state:dict(Counter(e['morphology']['state'] for e in active if e['status']==state)) for state in sorted({e['status'] for e in active})})
    anchor_lookup={tuple(data.detection_id.iloc[e['selected_indices']]):e for e in audit['all_r_anchor'] if e['status']=='final'}
    for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
        selected=[r for r in frozen if r['kind']==kind]
        summary['source_morphology'][kind]=dict(n=len(selected),anchor_states=dict(Counter(anchor_lookup[tuple(r['detection_ids'])]['morphology']['state'] if tuple(r['detection_ids']) in anchor_lookup else 'non_r_valid_only_route' for r in selected)))
    spectra=pd.read_csv(source/'spectra_union.csv')
    spectra=spectra.loc[inside(spectra.ra,spectra.dec,spec)].reset_index(drop=True)
    _,opportunity=spectral_opportunity(Path(footprint),spectra,set(),return_masks=True)
    targets=project(spectra.ra,spectra.dec,center)
    raw=pd.read_csv(source/'detections.csv',usecols=['ra_deg','dec_deg','survey'])
    finite=np.isfinite(raw[['ra_deg','dec_deg']].to_numpy(float)).all(axis=1)
    raw=raw.loc[finite]
    raw_xy=project(raw.ra_deg,raw.dec_deg,center)
    raw_late=raw.survey.eq('NSC').to_numpy()
    raw_counts={s:cKDTree(raw_xy[mask]).query_ball_point(targets,1000,return_length=True) for s,mask in [('NSC',raw_late),('SDSS',~raw_late)]}
    del raw,raw_xy
    physical=physical_candidates(data);support=calibration_support(keys,solution)
    qualified=(keep&(~nsc|physical))|(nsc&physical)
    selected=np.flatnonzero(qualified)
    neighbors=cKDTree(xy[selected]).query_ball_point(targets,1000)
    blind={int(v['qso_local_index']):v for v in read(folder/'private_blind_qso.json')}
    related=defaultdict(list)
    for route in ('all_r_anchor','non_r'):
        for e in audit[route]:
            for i in e.get('anchor_indices',e.get('track_indices',[])):
                if nsc[i]:related[int(i)].append((route,e))
    private=[]
    maximum=0.
    for k,goal in spectra.iterrows():
        if not 21<=goal.reference_r<23:continue
        ix=selected[np.asarray(neighbors[k],int)]
        late=ix[nsc[ix]];early=ix[~nsc[ix]]
        options={id(e):(route,e) for i in late for route,e in related[int(i)] if e['status'] not in ('outside_core','contains_r','missing_survey')}
        flags=[]
        found=blind.get(k)
        if found is not None:
            status='recovered';saved=frozen[found['source_slot']];member=tuple(saved['detection_ids'])
            replay=final(actual[member]['index'])
            delta=float(np.max(abs(np.asarray(saved['pm_icrs_masyr'])-replay['pm_icrs_masyr'])))
            maximum=max(maximum,delta)
            np.testing.assert_array_equal(saved['pm_icrs_masyr'],replay['pm_icrs_masyr'])
        elif raw_counts['NSC'][k]==0:status='no_nsc_detection'
        elif raw_counts['SDSS'][k]==0:status='no_sdss_detection'
        elif not len(late):status='nsc_detection_physical_quality'
        elif not len(early):status='sdss_detection_quality_or_position'
        elif not options:status='no_eligible_anchor_or_track'
        else:
            # Several plausible tracks remain an explicit ambiguity, rather
            # than choosing a trajectory for having a convenient final PM.
            values=list(options.values())
            values.sort(key=lambda r:ORDER.index(r[1]['status']) if r[1]['status'] in ORDER else -1,reverse=True)
            route,event=values[0];status=event['status']
            if len(options)>1:flags.append('multiple_anchor_or_track_candidates')
            if status=='final':status='final_spectral_identity_or_owner'
        if len(late):
            if not support[late].all():flags.append('some_measurements_lack_reference_support')
            if len({data.exposure_id.iat[i] for i in late})<2:flags.append('fewer_than_two_nsc_exposures')
        if not len(early):flags.append('no_qualified_sdss_position')
        morph=[e['morphology'] for _,e in options.values() if 'morphology' in e]
        identity=dict(origin=str(goal.origin),ra=float(goal.ra),dec=float(goal.dec),original_table_row=int(k),
                      original_table_sha256=digest(source/'spectra_union.csv'))
        if 'target' in goal and pd.notna(goal['target']):identity['native_target_id']=str(goal['target'])
        private.append(dict(identity=identity,magnitude=float(goal.reference_r),geometric_opportunity=bool(opportunity[k]),
            primary_exit=status,overlap_flags=flags,anchor_morphology=morph,
            raw_nsc_detections=int(raw_counts['NSC'][k]),raw_sdss_detections=int(raw_counts['SDSS'][k]),
            qualified_nsc_detections=len(late),qualified_sdss_detections=len(early),
            sdss_morphology=data.morphology.iloc[early].dropna().tolist(),
            recovered_kind=frozen[found['source_slot']]['kind'] if found else None))
    for low in (21,22):
        part=[r for r in private if low<=r['magnitude']<low+1]
        opportunity_rows=[r for r in part if r['geometric_opportunity']]
        summary['qso'][str(low)]={name:dict(denominator=len(rows),primary_exit=dict(Counter(r['primary_exit'] for r in rows)),
            overlap_flags=dict(Counter(f for r in rows for f in r['overlap_flags'])),
            direct_anchor_morphology=dict(Counter(m['state'] for r in rows for m in r['anchor_morphology'])))
            for name,rows in [('core_positions',part),('geometric_opportunity',opportunity_rows),('measured_both_surveys',[r for r in opportunity_rows if r['raw_nsc_detections'] and r['raw_sdss_detections']])]}
        for value in summary['qso'][str(low)].values():assert sum(value['primary_exit'].values())==value['denominator']
    with gzip.open(work/'private_selection_trace.json.gz','wt') as stream:
        import json
        json.dump(dict(ticket=ticket,qso=private,anchors=audit),stream,default=json_default,allow_nan=False)
    summary.update(maximum_qso_pm_replay_difference_masyr=maximum,elapsed_seconds=monotonic()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(summary);save_json(output,summary)
    print('SELECTION_CORE_DONE',row['core'],len(private),flush=True)
    return summary


def run(registration,output,workers):
    plan=read(registration);frozen=Path(plan['frozen_manifest'])
    if digest(frozen)!=plan['frozen_manifest_sha256']:raise ValueError('Changed frozen catalogue')
    footprints=read(frozen)['region_footprints']
    output.mkdir(parents=True,exist_ok=True)
    tasks=[(r,str(output),str(Path(footprints[r['window']]).parent)) for r in plan['selected']]
    with ProcessPoolExecutor(max_workers=min(workers,len(os.sched_getaffinity(0)))) as pool:results=list(pool.map(one,tasks,chunksize=1))
    report=dict(registration_sha256=digest(registration),cores=results,slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(report);save_json(output/'v20_selection_summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('registration',type=Path);p.add_argument('output',type=Path);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute nodes')
    run(**vars(args))
