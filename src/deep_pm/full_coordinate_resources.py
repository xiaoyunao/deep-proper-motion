"""Actual v2.2 allocation accounting, including failures and bounded stellar work."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
from .io import save_json
from .public_summary import check_aggregate

JOBS='62036814 62036842 62036857 62036896 62036919 62037022 62037151 62037358 62037412 62037821 62037901 62038110 62038508 62039535 62039600 62039618 62039698 62039734 62039746 62039805 62040359 62041280 62041938'.split()
B_FULL='62037358 62037901 62038508 62039618 62039734'.split()
B_STEPS=['62041280.3']


def run(root,output,extra=()):
    jobs=list(dict.fromkeys(JOBS+list(extra)));rows=subprocess.check_output(['sacct','-j',','.join(jobs),'--format=JobIDRaw,State,ElapsedRaw,AllocTRES,TotalCPU,MaxRSS','--parsable2','--noheader'],text=True);parents={};steps={}
    for line in rows.splitlines():
        job,state,elapsed,tres,cpu,rss=line.split('|')[:6];r=dict(state=state,elapsed_seconds=int(elapsed),allocated_tres=tres,total_cpu=cpu,maximum_rss=rss)
        if '.' in job:steps[job]=r
        else:parents[job]=r
    if set(jobs)!=parents.keys():raise ValueError('Missing actual allocation record')
    for job,r in parents.items():
        tres=dict(t.split('=',1) for t in r['allocated_tres'].split(',') if '=' in t);r['billing_logical_cpu_equivalent']=float(tres.get('billing',tres.get('cpu',0)));r['cpu_node_hours']=r['elapsed_seconds']/3600*r['billing_logical_cpu_equivalent']/256
    total=sum(r['cpu_node_hours'] for r in parents.values());b=sum(parents[j]['cpu_node_hours'] for j in B_FULL)
    for j in B_STEPS:
        if j not in steps:raise ValueError('Missing bounded stellar step')
        b+=steps[j]['elapsed_seconds']/3600*parents[j.split('.')[0]]['billing_logical_cpu_equivalent']/256
    allocated=int(subprocess.check_output(['du','-sk',str(root)],text=True).split()[0])*1024
    files=sum(len(f) for _,_,f in os.walk(root))
    out=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),parents=parents,steps=steps,
        estimated_billed_cpu_node_hours=total,stellar_conservative_cpu_node_hours_upper_bound=b,
        total_budget_cpu_node_hours=48,stellar_budget_cpu_node_hours=2,task_allocated_bytes_snapshot=allocated,task_file_count_snapshot=files,
        stellar_upper_bound_definition='Full allocations of all dedicated or mixed old stellar jobs and the whole reference0/1 parent are conservatively charged to B; newer B-only overlap step walltime charged at its full parent allocation rate. B is contained in total, never added twice. This intentionally exceeds actual B CPU consumption.',
        accounting='Actual sacct parent AllocTRES billing/256 times actual elapsed hours, all failure/cancellation/OOM parents included exactly once; steps listed diagnostically, never double billed. Iris is authoritative. Running allocations are snapshots, not final charges.',
        storage='Only new v2.2 PScratch root, current allocated bytes/files; originals and historical caches excluded, not peak or user quota.',
        unallocated_submission_rejections=['shared_interactive sbatch unsupported','QOSMaxSubmitJobPerUserLimit before final12 allocation; waited for completed predecessor'],
        scope='Memory is measured per-step MaxRSS and per-process science receipts, not claimed simultaneous node aggregate. No periodic touch or Home data copy.')
    check_aggregate(out);save_json(output,out)
    if total>=48 or b>=2:raise ValueError('Budget reached: stop new dispatch and preserve completed science')
    print('V22_ACTUAL_RESOURCE_SNAPSHOT',total,b,files,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--job',action='append',default=[]);a=p.parse_args();run(a.root,a.output,a.job)
