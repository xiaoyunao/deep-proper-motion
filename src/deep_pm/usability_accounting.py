"""Reconcile old-area physical members and trace metadata-ranked r losses."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np

from .__main__ import tomllib
from .allband_qso import physical_candidates
from .astrometry import unproject
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION,SPATIAL,read_offsets,reference_frame,verify_directory
from .exposure_catalogues import inside
from .io import digest,save_json,json_default
from .nsc_ownership_overlay import STANDARD
from .pilot_production import make_tracks,quality_detections,source_funnel
from .precision_v03 import calibration_groups,fit_with_calibration
from .public_summary import check_aggregate


def key(row):
    return hashlib.sha256('\0'.join(sorted(row['detection_ids'])).encode()).hexdigest()


def physical_trace(entry,core):
    folder=Path(entry['fit']);source=(folder/'source_receipt.json').resolve().parent
    graph=(folder/'reference_receipt.json').resolve().parent/'private_c0.npz'
    config=tomllib.loads(PRECISION.read_text());settings=tomllib.loads(SPATIAL.read_text());prod=settings['production']
    spec=tomllib.loads((folder/'geometry.toml').read_text())['fields'][core]
    center=reference_frame(graph);offsets,gauges=read_offsets(graph,with_gauge=True)
    meta=read(folder/'nsc_science_catalogue_summary.json');gauge=meta['reference_gauge']
    offsets={k:v for k,v in offsets.items() if gauges[k]==gauge}
    data,xy,time,sigma,keep=quality_detections(source,center,config,prod)
    nsc=data.survey.eq('NSC').to_numpy();keys=calibration_groups(data);visit=data.exposure_id.to_numpy()
    strict=keep&(~nsc|physical_candidates(data))
    tracks=make_tracks(xy,visit,strict,prod['match_radius_mas']);seeds=make_tracks(xy,visit,strict&nsc,prod['match_radius_mas'])
    @lru_cache(maxsize=32768)
    def solve(members):
        ix=np.asarray(members,int)
        return fit_with_calibration(xy[ix],time[ix],sigma[ix],config['reference_epoch'],[offsets[g] for g in keys[ix]])
    events=[]
    _,results=source_funnel(data,xy,time,sigma,strict,tracks,offsets,config,settings,9.,initial=seeds,
        observation_groups=keys,core_members=lambda pos:inside(*unproject(pos,center)[0],spec),
        fit_source=lambda ix:solve(tuple(ix)),audit_records=events)
    if len(results)!=meta['funnel']['strict_r_solutions']:raise ValueError('Physical-r trace does not reproduce frozen count')
    records={hashlib.sha256('\0'.join(sorted(data.detection_id.iloc[e['anchor_indices']])).encode()).hexdigest():e for e in events if e['status']!='outside_core'}
    if len(records)!=sum(e['status']!='outside_core' for e in events):raise ValueError('Non-unique physical anchor identity')
    return records,len(results),digest(source/'run_manifest.json'),digest(graph)


def trace_task(task):
    row,output=task;work=Path(output)/'r_trace'/row['core'];work.mkdir(parents=True,exist_ok=True)
    file=work/'v20_r_count_transition_summary.json'
    ticket=dict(implementation_sha256=digest(Path(__file__)),before_fit_sha256=digest(Path(row['before']['fit'])/'run_manifest.json'),after_fit_sha256=digest(Path(row['after']['fit'])/'run_manifest.json'))
    if file.exists():
        prior=read(file)
        if prior['ticket']!=ticket:raise ValueError('Changed r-transition trace inputs')
        return prior
    start=monotonic()
    old,nold,old_input,old_graph=physical_trace(row['before'],row['core'])
    new,nnew,new_input,new_graph=physical_trace(row['after'],row['core'])
    prior={k for k,v in old.items() if v['status']=='final'};now={k for k,v in new.items() if v['status']=='final'}
    def reason(e):
        if e is None:return 'anchor_membership_or_core_geometry'
        if e['status']=='quality':return 'formal_pm_error' if e['formal_error_exit'] else 'normalized_residual'
        return e['status']
    gone=Counter(reason(new.get(k)) for k in prior-now)
    added=Counter(reason(old.get(k)) for k in now-prior)
    if len(now)-len(prior)!=row['physical_r_delta']:raise ValueError('Physical-r changes do not reconcile')
    result=dict(ticket=ticket,core=row['core'],window=row['window'],old_physical_r=nold,new_physical_r=nnew,
        exact_anchor_id_common_success=len(now&prior),exited=len(prior-now),entered=len(now-prior),
        exit_reasons=dict(gone),entry_previous_reasons=dict(added),normalized_input_same=old_input==new_input,
        old_reference_sha256=old_graph,new_reference_sha256=new_graph,
        elapsed_seconds=monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result);save_json(file,result)
    print('R_COUNT_TRACED',row['core'],nold,nnew,dict(gone),dict(added),flush=True)
    return result


def index(frozen):
    result={}
    for part in read(frozen/'private_continuation_manifest.json')['partitions']:
        owned=Path(part['owned']);fits=read(owned/'fits.json');overlay=read(owned/'private_owner_overlay.json')
        moves={}
        for e in overlay['transfers']:
            if e['filename']==STANDARD:moves.setdefault(e['origin_core'],{})[e['source_slot']]=e['owner_core']
        for core in fits:result[core]=dict(path=fits[core],moves=moves.get(core,{}))
    return result


def retained(entry,core,scope):
    if entry is None:return {}
    folder=Path(entry['path']);verify_directory(folder,(STANDARD,))
    rows=read(folder/STANDARD)
    result={}
    for slot,row in enumerate(rows):
        owner=entry['moves'].get(slot,core)
        if owner not in scope:continue
        value=key(row)
        if value in result:raise ValueError('Duplicate physical source in one core')
        result[value]=row
    return result


def member_block(task):
    number,tasks,scope,output,ticket=task
    dest=Path(output)/f'private_member_block_{number:03}.json.gz'
    if dest.exists():
        old=read(dest)
        if old['ticket']!=ticket:raise ValueError('Changed member audit inputs')
        return old
    common=0;before_n=0;after_n=0;old_pending={};new_pending={}
    for core,before,after in tasks:
        old=retained(before,core,scope);new=retained(after,core,scope)
        before_n+=len(old);after_n+=len(new);common+=len(old.keys()&new.keys())
        old_pending.update({k:dict(core=core,record=old[k]) for k in old.keys()-new.keys()})
        new_pending.update({k:dict(core=core,record=new[k]) for k in new.keys()-old.keys()})
    result=dict(ticket=ticket,cores=len(tasks),common_same_origin=common,before=before_n,after=after_n,old=old_pending,new=new_pending,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    with gzip.open(dest,'wt') as f:json.dump(result,f,default=json_default,allow_nan=False)
    print('MEMBER_BLOCK_DONE',number,len(tasks),before_n,after_n,flush=True)
    return result


def run(plan_file,output,workers):
    plan=read(plan_file);before=Path(plan['previous']);after=Path(plan['current'])
    for field,frozen in [('previous',before),('current',after)]:
        if digest(frozen/'private_continuation_manifest.json')!=plan[field+'_manifest_sha256']:raise ValueError('Changed accounting scope')
    output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:traces=list(pool.map(trace_task,[(r,str(output)) for r in plan['selected']]))
    old,new=index(before),index(after);scope=set(old)
    affected=[core for core in old if old[core]['path']!=new[core]['path']]
    incoming=[core for core,r in new.items() if core not in old and any(owner in scope for owner in r['moves'].values())]
    tasks=[(core,old.get(core),new[core]) for core in sorted(affected+incoming)]
    ticket=dict(plan_sha256=digest(plan_file),implementation_sha256=digest(Path(__file__)))
    blocks=[(i,tasks[start:start+64],scope,str(output),ticket) for i,start in enumerate(range(0,len(tasks),64))]
    with ProcessPoolExecutor(max_workers=workers) as pool:results=list(pool.map(member_block,blocks))
    older={};newer={};common=sum(r['common_same_origin'] for r in results)
    for r in results:older.update(r['old']);newer.update(r['new'])
    moved=older.keys()&newer.keys();old_only={k:r for k,r in older.items() if k not in moved};new_only={k:r for k,r in newer.items() if k not in moved}
    nold=sum(r['before'] for r in results);nnew=sum(r['after'] for r in results)
    expected_delta=sum(r['allband_delta'] for r in plan['changes'])
    if nnew-nold!=expected_delta or len(new_only)-len(old_only)!=expected_delta:raise ValueError('Old-area final physical member accounting failed')
    save_json(output/'private_allband_membership_transitions.json',dict(ticket=ticket,old_only=old_only,new_only=new_only))
    report=dict(old_scope_cores=len(scope),replayed_origins=len(affected),new_origins_with_old_owner=len(incoming),
        before_replayed_owned_sources=nold,after_replayed_owned_sources=nnew,exact_common_physical_members=common+len(moved),
        entered=len(new_only),exited=len(old_only),allband_net_change=expected_delta,
        physical_r_traces=traces,physical_r_traced_net_change=sum(r['new_physical_r']-r['old_physical_r'] for r in traces),
        physical_r_full_net_change=sum(r['physical_r_delta'] for r in plan['changes']),
        unchanged_reused_origins=len(scope)-len(affected),maximum_worker_peak_rss_mib=max([r['peak_rss_mib'] for r in results]+[r['peak_rss_mib'] for r in traces]),slurm_job=os.environ['SLURM_JOB_ID'],
        peak_rss_driver_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        scope='All changed reference-domain source files plus physical transfers into old owners. Exact physical-member identity, not proximity or PM agreement. Physical-r failed/successful traces on eight preregistered metadata-ranked cores; remaining count changes retain aggregate accounting without invented causes.')
    check_aggregate(report);save_json(output/'v20_replay_accounting_summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan_file',type=Path);p.add_argument('output',type=Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute nodes')
    run(**vars(a))
