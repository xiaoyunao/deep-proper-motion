"""Administrative Slurm and PScratch accounting; no scientific data execution."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess

from .io import save_json
from .public_summary import check_aggregate

JOBS=['61964815','61965574','61965664','61965720','61965915','61966119','61966448','61966558','61967287','61967420','61967758','61967856']


def run(root,output,extra_jobs):
    jobs=JOBS+extra_jobs
    rows=subprocess.check_output(['sacct','-j',','.join(jobs),'--format=JobIDRaw,JobName,State,ElapsedRaw,AllocTRES,TotalCPU,MaxRSS','--parsable2','--noheader'],text=True)
    heads={};steps={}
    for line in rows.splitlines():
        job,name,state,elapsed,tres,cpu,rss=line.split('|')[:7]
        row=dict(job=job,name=name,state=state,elapsed_seconds=int(elapsed),allocated_tres=tres,total_cpu=cpu,maximum_rss=rss)
        if '.' in job:steps[job]=row
        else:heads[job]=row
    missing=set(jobs)-heads.keys()
    if missing:raise ValueError(f'Missing actual Slurm records: {sorted(missing)}')
    total=0.;performance=0.
    for job,row in heads.items():
        tres=dict(v.split('=',1) for v in row['allocated_tres'].split(',') if '=' in v)
        # Perlmutter CPU:256 logical CPUs per full node; shared AllocTRES billing
        # already reflects the allocated CPU/memory fraction. Iris is authoritative.
        billing=float(tres.get('billing',tres.get('cpu',0)))
        cost=billing/256*row['elapsed_seconds']/3600
        row['estimated_billed_cpu_node_hours']=cost;total+=cost
        row['budget_class']='performance' if job in ('61965574','61968365') else 'science_or_recovery'
        if row['budget_class']=='performance':performance+=cost
    allocated_kib=int(subprocess.check_output(['du','-sk',str(root)],text=True).split()[0])
    files=sum(len(names) for _,_,names in os.walk(root))
    result=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),jobs=heads,steps=steps,
        estimated_cpu_node_hours=total,performance_cpu_node_hours=performance,
        budget_cpu_node_hours=10,performance_cap_cpu_node_hours=2,
        pscratch_task_allocated_bytes=allocated_kib*1024,task_file_count=files,
        accounting='Actual sacct allocations and elapsed wall time, including failed/OOM jobs. Estimate = AllocTRES billing/256*elapsed_hours; final NERSC Iris ledger remains authoritative. Requested limits are not charged elapsed time.',
        storage='Task root created for v2.0. Snapshot is task increment, not user filesystem quota or measured all-process storage peak; original historical data are excluded.',
        peak_memory='Per-step sacct MaxRSS and measured process peaks are supplied; no claim of a simultaneously measured aggregate memory peak.',
        current_jobs_all_terminal=all(v['state'] in ('COMPLETED','FAILED','OUT_OF_MEMORY','CANCELLED','TIMEOUT') for v in heads.values()))
    if total>10 or performance>2:raise ValueError('Actual resource budget exceeded')
    check_aggregate(result);save_json(output,result)
    print('RESOURCE_LEDGER',total,performance,files,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--job',action='append',default=[],dest='extra_jobs');a=p.parse_args();run(**vars(a))
