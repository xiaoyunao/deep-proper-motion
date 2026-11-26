"""Memory-bounded, per-core isolated replay of accepted reference components.

No worker retains a previous giant input's allocator across cores. Scheduling
uses only file size and measured reference-input RSS, never QSO performance.
"""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic,sleep

import numpy as np
import pandas as pd
from .full_coordinate_sources import component_map
from .precision_v03 import calibration_groups

from .blind_catalogue_quality import read
from .full_coordinate_sources import one
from .io import digest,save_json
from .public_summary import check_aggregate


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


def run(root,components,workers,memory_gib,hours):
    if not read(root/'admission/v22_complete_component_admission_summary.json')['numerical_admission_passed']:raise ValueError('Complete-component admission missing')
    plan=read(root/'registration/private_full_coordinate_plan.json');indices=selected_cores(root,components)
    key='-'.join(map(str,components));job=os.environ['SLURM_JOB_ID'];dest=root/'batch'/f'component-{key}'/job;dest.mkdir(parents=True,exist_ok=True)
    pending=[]
    for i in indices:
        e=plan['cores'][i];size=Path(e['source'],'detections.csv').stat().st_size/(1024**3)
        measured=read(root/'input'/e['core']/'v22_reference_input_core_summary.json')['peak_rss_mib']/1024
        # Input extraction and source fitting share the same standardisation.
        # Use the smaller of a conservative file estimate and the measured
        # allocator high-water bound, then 50percent fit/output headroom.
        memory=2+1.5*min(45*size+1,measured)
        if memory>memory_gib:raise ValueError('One core exceeds allocation memory model')
        pending.append((i,memory,size))
    pending.sort(key=lambda r:(-r[1],-r[2],r[0]));running=[];done=[];failed=[];start=monotonic();deadline=start+hours*3600
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
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                running.append(dict(index=i,core=e['core'],memory=memory,process=process,log=log))
                print('SOURCE_DISPATCH',e['core'],'estimated_GiB',round(memory,2),flush=True)
        for item in list(running):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();running.remove(item)
            (done if code==0 else failed).append(item['core'])
            print('SOURCE_FINISHED',item['core'],code,'done',len(done),'failed',len(failed),'queued',len(pending),flush=True)
        if not running and (not pending or monotonic()>=deadline):break
        sleep(2)
    result=dict(components=components,registered_cores=len(indices),completed_core_checks=len(done),failed_cores=failed,
        undispatched_cores=len(pending),maximum_workers=workers,memory_budget_gib=memory_gib,elapsed_seconds=monotonic()-start,
        implementation_sha256=digest(Path(__file__)),slurm_job=job,
        scope='Every worker validates source/reference/ticket and all private output hashes; isolated process per core prevents retained giant-input memory. Completion is per origin core, before global ownership/detection/spectral closure. Failed cores retain logs and must recover; no silent numerical failure acceptance.')
    check_aggregate(result);save_json(dest/'v22_source_batch_summary.json',result)
    if failed:raise RuntimeError('Affected source cores failed; inspect per-core logs before retry')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--component',action='append',type=int,required=True)
    p.add_argument('--single-index',type=int);p.add_argument('--workers',type=int,default=12);p.add_argument('--memory-gib',type=float,default=160);p.add_argument('--hours',type=float,default=3.5);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    if a.single_index is not None:
        entry=read(a.root/'registration/private_full_coordinate_plan.json')['cores'][a.single_index]
        # Scheduling domains never define a star's available calibration.
        # Inspect all actual target physical groups, including exposures with
        # no local selected reference galaxies. This also closes previously
        # accepted single-component preliminary replay after all graphs exist.
        mapping,_=component_map(str(a.root))
        physical=pd.read_csv(Path(entry['source'])/'detections.csv',usecols=['survey','exposure_id','detector_id'],low_memory=False).drop_duplicates()
        groups=calibration_groups(physical)
        actual=sorted(set(mapping.reindex(pd.unique(groups)).dropna().astype(int)))
        available=[c for c in actual if (a.root/'reference'/f'component-{c:04d}'/'private_c0_unfiltered.npz').exists()]
        if available!=actual:
            print('SOURCE_DEFERRED_UNSOLVED_PHYSICAL_COMPONENT',entry['core'],actual,flush=True)
            raise SystemExit(3)
        folder=a.root/'sources'/entry['core'];prior=folder/'v22_source_core_summary.json'
        if prior.exists() and set(read(prior)['ticket']['graphs'])!=set(map(str,actual)):
            archive=a.root/'partial-source-archives'/entry['core'];archive.parent.mkdir(exist_ok=True)
            if archive.exists():raise ValueError('Repeated partial component archive requires inspection')
            folder.rename(archive);print('PRESERVED_PRELIMINARY_COMPONENT_SOURCE',entry['core'],flush=True)
        one((entry,str(a.root),None))
    else:run(a.root,a.component,a.workers,a.memory_gib,a.hours)
