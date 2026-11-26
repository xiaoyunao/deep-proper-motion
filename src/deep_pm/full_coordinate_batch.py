"""Memory-bounded, per-core isolated replay of accepted reference components.

No worker retains a previous giant input's allocator across cores. Scheduling
uses only file size and measured reference-input RSS, never QSO performance.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys
import resource
from time import monotonic,sleep

import numpy as np
import pandas as pd
from .full_coordinate_sources import component_map
from .precision_v03 import calibration_groups

from .blind_catalogue_quality import read
from .full_coordinate_sources import one
from .io import digest,save_json
from .public_summary import check_aggregate

DISPATCH_DEFINITION='All normalized target physical groups. Metadata-only dispatch, including groups with no local reference galaxies; physical quality selection remains in source normalization.'


def physical_dispatch(entry,root,mapping,mapping_sha):
    """Reuse only an actual-target dispatch bound to unchanged signed inputs."""
    source_sha=digest(Path(entry['source'])/'run_manifest.json')
    path=root/'sources'/entry['core']/'private_dispatch.json'
    if path.exists():
        previous=read(path)
        if previous['source_sha256']!=source_sha or previous['group_map_sha256']!=mapping_sha:raise ValueError('Changed physical-dispatch input')
        if previous.get('definition')==DISPATCH_DEFINITION:return previous['components'],previous,True
    physical=pd.read_csv(Path(entry['source'])/'detections.csv',usecols=['survey','exposure_id','detector_id'],low_memory=False).drop_duplicates()
    groups=calibration_groups(physical);actual=sorted(set(mapping.reindex(pd.unique(groups)).dropna().astype(int)))
    return actual,dict(source_sha256=source_sha,group_map_sha256=mapping_sha,components=actual,definition=DISPATCH_DEFINITION),False


def cached_check(task):
    """Validate completed files in a warm pool; never fit a missing checkpoint."""
    entry,root=task;root=Path(root);folder=root/'sources'/entry['core'];path=folder/'v22_source_core_summary.json'
    before=digest(path);start=monotonic();proof=read(path)
    mapping,mapping_sha=component_map(str(root));actual,payload,reused=physical_dispatch(entry,root,mapping,mapping_sha)
    if set(proof['ticket']['graphs'])!=set(map(str,actual)):
        return dict(core=entry['core'],reused=False,source_sha256=before)
    dispatch=folder/'private_dispatch.json'
    if not reused:
        if dispatch.exists():
            preserved=folder/'private_dispatch_before_actual_cache_check.json'
            if preserved.exists() and read(preserved)!=read(dispatch):raise ValueError('Repeated changed preliminary dispatch')
            if not preserved.exists():save_json(preserved,read(dispatch))
        save_json(dispatch,payload)
    # Existing proof is mandatory: the normal one() cache branch verifies its
    # complete scientific ticket and every output hash before returning.
    if not path.exists():raise ValueError('Completed cache disappeared')
    one((entry,str(root),None))
    if digest(path)!=before:raise ValueError('Read-only source cache validation changed the scientific receipt')
    return dict(core=entry['core'],reused=True,source_sha256=before,elapsed_seconds=monotonic()-start,
        process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        scope='Warm cache-only worker high-water RSS may include earlier metadata checks; no missing-source fit allowed.')


def selected_cores(root,components):
    plan=read(root/'registration/private_full_coordinate_plan.json')
    meta=read(root/'graph/v22_global_graph_summary.json')
    # Accepted domains retain their inherited chart and remain disconnected.
    # This prefilter is only allowed when that identity is exact, not when a
    # connected-domain extension merged previous physical reference graphs.
    centers=[r['center_deg'] for r in meta['components'] if r['component'] in components]
    old=[k for k,v in plan['graphs'].items() if any(np.allclose(v['center'],c,atol=1e-12,rtol=0) for c in centers)]
    if len(old)!=len(components):raise ValueError('Component scheduler requires explicit physical dispatch for a merged reference domain')
    return [i for i,e in enumerate(plan['cores']) if e['old_graph'] in old]


def source_memory_bound(csv_gib,old_gzip_gib):
    """Resource-only upper envelope, admitted against actual completed fits."""
    return 2+8*csv_gib+128*old_gzip_gib


def memory_queue(root,plan,indices,mode):
    pending=[];legacy={};ratios=[];signatures=[]
    scientific_sha=digest(Path(__file__).with_name('full_coordinate_sources.py'))
    for i in indices:
        e=plan['cores'][i];size=Path(e['source'],'detections.csv').stat().st_size/(1024**3)
        measured=read(root/'input'/e['core']/'v22_reference_input_core_summary.json')['peak_rss_mib']/1024
        conservative=2+1.5*min(45*size+1,measured)
        bound=min(conservative,source_memory_bound(size,Path(e['fit'],'private_standard_motion.json.gz').stat().st_size/1024**3))
        legacy[i]=conservative;pending.append((i,bound if mode=='receipt-validated' else conservative,size))
        path=root/'sources'/e['core']/'v22_source_core_summary.json';dispatch=path.parent/'private_dispatch.json'
        if path.exists() and dispatch.exists():
            proof=read(path);physical=read(dispatch)
            if proof['ticket']['implementation_sha256']==scientific_sha and set(proof['ticket']['graphs'])==set(map(str,physical['components'])):
                ratios.append(proof['peak_rss_mib']/1024/bound);signatures.append(digest(path))
    admitted=len(ratios)>=50 and max(ratios)<=.8
    if mode=='receipt-validated' and not admitted:pending=[(i,legacy[i],size) for i,_,size in pending]
    audit=dict(requested=mode,used='receipt-validated' if mode=='receipt-validated' and admitted else 'reference-input-conservative',
        completed_source_cost_receipts=len(ratios),maximum_observed_rss_over_selected_bound=max(ratios) if ratios else None,
        receipt_set_sha256=sha256('\n'.join(signatures).encode()).hexdigest(),
        admission_minimum_completed_cores=50,admission_maximum_rss_fraction=.8,
        formula='min(original reference-input bound, 2 + 8*CSV_GiB + 128*old_science_gzip_GiB); all units GiB',
        scope='Only scheduling reservations/order change. Source implementation, inputs, reference, RNG, members, point and covariance are unchanged. Observed cost envelope is conditional, not a guaranteed process memory limit. Fall back if a subsequent core exceeds 80percent of its reservation.')
    return pending,legacy,audit


def run(root,components,workers,memory_gib,hours,memory_model='reference-input-conservative',worker_start='subprocess'):
    if not read(root/'admission/v22_complete_component_admission_summary.json')['numerical_admission_passed']:raise ValueError('Complete-component admission missing')
    plan=read(root/'registration/private_full_coordinate_plan.json');indices=selected_cores(root,components)
    key='-'.join(map(str,components));job=os.environ['SLURM_JOB_ID'];dest=root/'batch'/f'component-{key}'/job;dest.mkdir(parents=True,exist_ok=True)
    if workers>int(os.environ.get('SLURM_CPUS_PER_TASK',workers)):raise ValueError('Workers exceed allocated CPUs')
    pending,legacy,audit=memory_queue(root,plan,indices,memory_model)
    if any(memory>memory_gib for _,memory,_ in pending):raise ValueError('One core exceeds allocation memory model')
    fallbacks=0
    print('SOURCE_MEMORY_ADMISSION',audit,flush=True)
    pending.sort(key=lambda r:(-r[1],-r[2],r[0]));running=[];done=[];failed=[];start=monotonic();deadline=start+hours*3600
    cache_start=monotonic();cache_checks=[]
    if memory_model=='receipt-validated':
        # Fork only cache/three-column metadata checks. New fits still get an
        # isolated subprocess so giant scientific allocators cannot accumulate.
        candidates=[i for i,_,_ in pending if (root/'sources'/plan['cores'][i]['core']/'v22_source_core_summary.json').exists()]
        from deep_pm.full_coordinate_batch import cached_check as stable_cached_check
        with ProcessPoolExecutor(max_workers=min(8,workers)) as pool:
            futures={pool.submit(stable_cached_check,(plan['cores'][i],str(root))):i for i in candidates}
            for future in as_completed(futures):
                item=future.result();cache_checks.append(item)
                if item['reused']:done.append(item['core'])
                print('SOURCE_CACHE_VERIFIED',item['core'],item['reused'],'verified',len(done),flush=True)
        finished=set(done);pending=[item for item in pending if plan['cores'][item[0]]['core'] not in finished]
        save_json(dest/'private_cache_validation.json',cache_checks)
    cache_elapsed=monotonic()-cache_start
    while pending or running:
        # Never start work late in the allocation; finished core receipts resume.
        if len(failed)>=3 and not done:deadline=monotonic() # universal startup failure: stop further dispatch
        if monotonic()<deadline:
            while pending and len(running)<workers:
                available=memory_gib-sum(r['memory'] for r in running)
                slot=next((j for j,r in enumerate(pending) if r[1]<=available),None)
                if slot is None:break
                i,memory,size=pending.pop(slot);e=plan['cores'][i];log=(dest/(e['core']+'.log')).open('w')
                cmd=[sys.executable,'-um','deep_pm.full_coordinate_batch',str(root),'--single-index',str(i)]
                for c in components:cmd+=['--component',str(c)]
                if worker_start=='fork':
                    from multiprocessing import get_context
                    from deep_pm.full_coordinate_batch import isolated_fork_core as stable_fork_core
                    process=get_context('fork').Process(target=stable_fork_core,args=(str(root),i,log.fileno()))
                    process.start()
                else:process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                running.append(dict(index=i,core=e['core'],memory=memory,process=process,log=log))
                print('SOURCE_DISPATCH',e['core'],'estimated_GiB',round(memory,2),flush=True)
        for item in list(running):
            process=item['process']
            code=process.poll() if hasattr(process,'poll') else process.exitcode
            if code is not None and not hasattr(process,'poll'):process.join();process.close()
            if code is None:continue
            item['log'].close();running.remove(item)
            (done if code==0 else failed).append(item['core'])
            print('SOURCE_FINISHED',item['core'],code,'done',len(done),'failed',len(failed),'queued',len(pending),flush=True)
            if code==0 and audit['used']=='receipt-validated':
                actual=read(root/'sources'/item['core']/'v22_source_core_summary.json')['peak_rss_mib']/1024
                if actual>.8*item['memory']:
                    audit['used']='reference-input-conservative';fallbacks+=1
                    pending=[(i,legacy[i],size) for i,_,size in pending]
                    pending.sort(key=lambda r:(-r[1],-r[2],r[0]))
                    for active in running:active['memory']=legacy[active['index']]
                    print('SOURCE_MEMORY_FALLBACK',item['core'],actual,flush=True)
        if not running and (not pending or monotonic()>=deadline):break
        sleep(2)
    result=dict(components=components,registered_cores=len(indices),completed_core_checks=len(done),failed_cores=failed,
        undispatched_cores=len(pending),maximum_workers=workers,memory_budget_gib=memory_gib,elapsed_seconds=monotonic()-start,
        memory_reservation_audit=audit,memory_reservation_fallbacks=fallbacks,worker_start_method=worker_start,
        warm_cache_checks=len(cache_checks),warm_cache_reused=sum(v['reused'] for v in cache_checks),
        warm_cache_elapsed_seconds=cache_elapsed,
        implementation_sha256=digest(Path(__file__)),slurm_job=job,
        scope='Every worker validates source/reference/ticket and all private output hashes; isolated process per core prevents retained giant-input memory. Completion is per origin core, before global ownership/detection/spectral closure. Failed cores retain logs and must recover; no silent numerical failure acceptance.')
    check_aggregate(result);save_json(dest/'v22_source_batch_summary.json',result)
    if failed:raise RuntimeError('Affected source cores failed; inspect per-core logs before retry')


def single_core(root,index):
    entry=read(root/'registration/private_full_coordinate_plan.json')['cores'][index]
    # Scheduling domains never define a star's available calibration.
    # Inspect all actual target physical groups, including exposures with
    # no local selected reference galaxies. This also closes previously
    # accepted single-component preliminary replay after all graphs exist.
    mapping,mapping_sha=component_map(str(root))
    actual,dispatch,reused=physical_dispatch(entry,root,mapping,mapping_sha)
    if reused:print('REUSED_SIGNED_PHYSICAL_DISPATCH',entry['core'],flush=True)
    available=[c for c in actual if (root/'reference'/f'component-{c:04d}'/'private_c0_unfiltered.npz').exists()]
    if available!=actual:
        print('SOURCE_DEFERRED_UNSOLVED_PHYSICAL_COMPONENT',entry['core'],actual,flush=True)
        raise SystemExit(3)
    folder=root/'sources'/entry['core'];prior=folder/'v22_source_core_summary.json'
    if prior.exists() and set(read(prior)['ticket']['graphs'])!=set(map(str,actual)):
        archive=root/'partial-source-archives'/entry['core'];archive.parent.mkdir(exist_ok=True)
        if archive.exists():raise ValueError('Repeated partial component archive requires inspection')
        folder.rename(archive);print('PRESERVED_PRELIMINARY_COMPONENT_SOURCE',entry['core'],flush=True)
    folder.mkdir(parents=True,exist_ok=True)
    if not reused or not (folder/'private_dispatch.json').exists():save_json(folder/'private_dispatch.json',dispatch)
    one((entry,str(root),None))


def isolated_fork_core(root,index,log_descriptor):
    # Fresh child of the thin dispatcher; it never processes a second core.
    # Inherited modules avoid repeated shared-filesystem import startup.
    os.dup2(log_descriptor,1);os.dup2(log_descriptor,2)
    single_core(Path(root),index)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--component',action='append',type=int,required=True)
    p.add_argument('--single-index',type=int);p.add_argument('--workers',type=int,default=12);p.add_argument('--memory-gib',type=float,default=160);p.add_argument('--hours',type=float,default=3.5)
    p.add_argument('--memory-model',choices=['reference-input-conservative','receipt-validated'],default='reference-input-conservative');p.add_argument('--worker-start',choices=['subprocess','fork'],default='subprocess');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    if a.single_index is not None:single_core(a.root,a.single_index)
    else:run(a.root,a.component,a.workers,a.memory_gib,a.hours,a.memory_model,a.worker_start)
