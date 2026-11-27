"""Bounded-task allocation accounting; parent jobs include every failed retry."""
import argparse,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
from .io import save_json
from .public_summary import check_aggregate


def run(root,final=False):
    jobs=list(dict.fromkeys((root/'private_allocations.txt').read_text().split()))
    if not jobs or not all(j.isdigit() for j in jobs):raise ValueError('Missing/invalid actual Slurm allocation ledger')
    text=subprocess.check_output(['sacct','-n','-P','-j',','.join(jobs),'--format=JobIDRaw,State,ElapsedRaw,AllocTRES,MaxRSS'],text=True);parents={};steps={}
    for line in text.splitlines():
        j,state,seconds,tres,rss=line.split('|')[:5];r=dict(state=state,elapsed_seconds=int(seconds),allocated_tres=tres,maximum_rss=rss)
        if '.' in j:steps[j]=r
        else:
            values=dict(x.split('=',1) for x in tres.split(',') if '=' in x);r['estimated_billed_cpu_node_hours']=r['elapsed_seconds']/3600*float(values.get('billing',values.get('cpu',0)))/256;parents[j]=r
    if parents.keys()!=set(jobs):raise ValueError('Actual allocation missing from sacct')
    if final and any(r['state'] in ['RUNNING','PENDING','COMPLETING'] for r in parents.values()):raise ValueError('Final ledger requested with a live allocation')
    total=sum(r['estimated_billed_cpu_node_hours'] for r in parents.values());out=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),parents=parents,steps=steps,estimated_billed_cpu_node_hours=total,total_budget_cpu_node_hours=4,terminal=final,
        accounting='All actual parent AllocTRES billing/256 times ElapsedRaw hours, including failures/cancelled/OOM. Steps are diagnostic, never added again; Iris authoritative. Running entries are snapshots, not terminal charges.')
    check_aggregate(out);save_json(root/('v23_resource_final_summary.json' if final else 'v23_resource_snapshot_summary.json'),out)
    print('V23_RESOURCE',total,len(parents),flush=True)
    if total>=4:raise ValueError('Budget reached: stop new dispatch, retain completed science and report remainder')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--final',action='store_true');a=p.parse_args();run(a.root,a.final)
