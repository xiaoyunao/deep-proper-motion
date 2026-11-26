"""Checkpoint final-dispatch core indices while unrelated source cores finish."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic,sleep
import os
from .blind_catalogue_quality import read
from .full_coordinate_catalogue import index_core
from .usability_accounting import index as old_index


def run(root,workers,hours):
    plan=read(root/'registration/private_full_coordinate_plan.json');old=old_index(root.parent/'v1.9/batch/science')
    pending=set(range(len(plan['cores'])));running={};done=set();deadline=monotonic()+hours*3600
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            if monotonic()<deadline:
                for i in sorted(pending):
                    if len(running)>=2*workers:break
                    e=plan['cores'][i];folder=root/'sources'/e['core'];summary=folder/'v22_source_core_summary.json';dispatch=folder/'private_dispatch.json'
                    if not summary.exists() or not dispatch.exists():continue
                    proof=read(summary);physical=read(dispatch)
                    if set(proof['ticket']['graphs'])!=set(map(str,physical['components'])):continue
                    running[pool.submit(index_core,(i,e,str(root),{e['core']:old[e['core']]}))]=i
                pending-=set(running.values())
            for f in list(running):
                if f.done():f.result();done.add(running.pop(f));print('STREAM_FINAL_INDEX',len(done),'remaining',len(pending),flush=True)
            if not running and (not pending or monotonic()>=deadline):break
            sleep(10)
    print('STREAM_INDEX_CHECKPOINT',len(done),len(pending),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=4);p.add_argument('--hours',type=float,default=3);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    run(a.root,a.workers,a.hours)
