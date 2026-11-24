"""Start the prospectively frozen continuation from the accepted research method."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Condition
from time import sleep

import pandas as pd

from deep_pm.continuous_reference import FILES,read_offsets,verify_directory
from deep_pm.io import digest,save_json
from deep_pm.public_summary import check_aggregate

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Use a Slurm compute node')
work, registered, frozen, catalogues, sas, legacy = map(Path, sys.argv[1:])
rev=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
selection=registered/'wide_selection_summary.json'
protocol=registered/'next_windows.toml'
previous=frozen/'continuation_science_receipt.json'
reference=frozen/'private_continuation_manifest.json'
verify_directory(catalogues,('spectra_union.csv','spectra_veto.csv'))
proof=json.loads(previous.read_text())
if (sum(x['cores'] for x in proof['graphs'].values())!=2251
        or any(any(x['ownership'][name] for name in ('cross_core_shared_detection_claims','exact_position_duplicate_rows','outside_common_epoch_owner'))
               for x in proof['graphs'].values())):
    raise ValueError('The broad science and association ledger has not passed')
for path in (selection,protocol,reference,legacy):
    if not path.is_file():
        raise ValueError('Missing fixed continuation input')
subprocess.run([sys.executable,'-um','deep_pm.independent_batch',str(work),str(selection),str(protocol)],check=True)
plan=json.loads(selection.read_text())
if digest(previous)!=plan['previous_scope_sha256'] or digest(protocol)!=plan['config_sha256']:
    raise ValueError('Larger footprint inputs changed')
tasks=[tuple(line.split()) for line in (work/'tasks.tsv').read_text().splitlines()]
if len(tasks)!=4*len(plan['first_batch']):
    raise ValueError('The registered continuation geometry changed')
logs=work/'logs'
logs.mkdir(exist_ok=True)
nodes=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split()
counts={v['name']:v['exposures'] for v in plan['first_batch']}
ordered=defaultdict(list)
for task in tasks:
    ordered[task[1]].append(task)
workers_per_node=int(os.environ.get('DEEP_PM_INTAKE_WORKERS_PER_NODE','2'))
step_memory=int(os.environ.get('DEEP_PM_INTAKE_STEP_GIB','56'))
reference_memory=int(os.environ.get('DEEP_PM_INTAKE_REFERENCE_GIB','57'))
xiao_memory=int(os.environ.get('DEEP_PM_INTAKE_XIAO_GIB','39'))
node_memory=int(os.environ.get('DEEP_PM_INTAKE_NODE_GIB','170'))
if min(workers_per_node,step_memory,reference_memory,xiao_memory,node_memory)<=0 or max(step_memory,reference_memory,xiao_memory,110)>node_memory:
    raise ValueError('Intake step exceeds its per-node worker memory budget')
buffer={node:node_memory for node in nodes}
condition=Condition()

@contextmanager
def resources(node,gib):
    with condition:
        condition.wait_for(lambda:buffer[node]>=gib)
        buffer[node]-=gib
    try:
        yield
    finally:
        with condition:
            buffer[node]+=gib
            condition.notify_all()

def command(node,module,arguments,path,mem,attempts):
    cmd=['srun','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c4',
         f'--mem={mem}G','-w',node,sys.executable,'-um',module,*map(str,arguments)]
    with (logs/(path+'.out')).open('a') as out,(logs/(path+'.err')).open('a') as err:
        for i in range(attempts):
            with resources(node,mem):
                rc=subprocess.run(cmd,stdout=out,stderr=err).returncode
            if rc==0:return
            if i<attempts-1:sleep(20*(i+1))
    raise ValueError(f'No complete receipt after scoped retry: {path}')

def completed(folder,name,core,geometry_sha):
    receipt=folder/'run_manifest.json'
    if not receipt.is_file():
        return None
    manifest=json.loads(receipt.read_text())
    if (manifest['commit'],manifest['region'],manifest['config_sha256'])!=(rev,core,geometry_sha):
        raise ValueError('An existing acquisition stamp has a different geometry, core, or version')
    attempts=manifest['stages'].get(name,[])
    return attempts[-1] if attempts and attempts[-1]['state']=='completed' else None

def acquire(node,core,box):
    geom=work/f'batch-{box}.toml'
    config_sha=digest(geom)
    shard=work/'inputs'/core
    ref=work/'refs'/core
    input_stage='exposure-inputs-v09'
    ref_stage='continuous-ref-input-v12'
    input_reused=completed(shard,input_stage,core,config_sha) is not None
    reference_reused=completed(ref,ref_stage,core,config_sha) is not None
    if reference_reused and not input_reused:
        raise ValueError('A cached reference lacks a valid acquired input')
    need=110 if counts[core[:-3]]>500 else step_memory
    if not input_reused:
        command(node,'deep_pm.exposure_inputs',[work/'inputs',core,catalogues,sas,geom,'--workers','3'],core+'-fetch',need,3)
    inp=verify_directory(shard,(*FILES,'acquisition_summary.json'))
    if not reference_reused:
        command(node,'deep_pm.continuous_reference',['extract',ref,geom,core,shard],core+'-refs',reference_memory,2)
    rcp=verify_directory(ref,('reference.csv.gz','reference_summary.json'))
    input_sha=digest(inp)
    ref_proof=completed(ref,ref_stage,core,config_sha)
    if (ref/'source_receipt.json').resolve()!=inp.resolve() or ref_proof['inputs']['source_receipt.json']!=input_sha:
        raise ValueError('The extracted reference uses another input receipt')
    data=json.loads((shard/'acquisition_summary.json').read_text())
    if not data['all_nsc_object_selection']:
        raise ValueError('Unexpected r-only archive search in the continuation')
    return dict(window=box,normalized_detections=data['normalized_detections'],
                input_sha256=input_sha,reference_sha256=digest(rcp),
                input_reused=input_reused,reference_reused=reference_reused)

complete={}
errors={}
def collect(items,pool):
    submitted={pool.submit(acquire,nodes[i%len(nodes)],*t):t for i,t in enumerate(items)}
    for future in as_completed(submitted):
        core,box=submitted[future]
        try:
            complete[core]=future.result()
            print('NEXT_INPUT',core,complete[core]['normalized_detections'],flush=True)
        except Exception as exc:
            errors[core]=type(exc).__name__
            print('NEXT_INPUT_ERROR',core,repr(exc),flush=True)
        save_json(work/'next_intake_progress_summary.json',dict(cores=len(complete),planned=len(tasks),failures=errors))

# Two extremes per window within the 10–20 requested first-tranche cores.
canaries=sorted({max(v,key=lambda t:(counts[t[0][:-3]],t[0])) for v in ordered.values()})
lighter=sorted({min(v,key=lambda t:(counts[t[0][:-3]],t[0])) for v in ordered.values()}-set(canaries))
canaries.extend(lighter[:max(0,20-len(canaries))])
first_tranche = work/'next_intake_canary_summary.json'
with ThreadPoolExecutor(workers_per_node*len(nodes)+1) as pool:
    control=pool.submit(command,nodes[0],'deep_pm.legacy_regions',[work/'catalogues',legacy,protocol],'xiao',xiao_memory,1)
    collect(canaries,pool)
    rows = sum(v['normalized_detections'] for v in complete.values())
    disk_bytes = sum(sum(f.stat().st_size for f in (work/'inputs'/k).rglob('*') if f.is_file())
                     for k in complete)
    old_replay = plan['estimated_resources']['estimated_old_replayed_cores']
    # Calibrate full non-r data volume by metadata exposure mass, not by fitting
    # performance or the proper motions of the registered validation objects.
    measured_exposure_mass = sum(counts[t[0][:-3]] for t in canaries)
    planned_exposure_mass = sum(counts[t[0][:-3]] for t in tasks)
    projected_rows = int(rows*planned_exposure_mass/max(1,measured_exposure_mass))
    projected_disk = int(disk_bytes*planned_exposure_mass/max(1,measured_exposure_mass))
    final_and_reference_bytes = (len(tasks)+old_replay)*28500000 + len(tasks)*25000000
    intake_hours = max(len(tasks)*.012,11.59*projected_rows/419424538)
    planned_hours = plan['estimated_resources']['estimated_cpu_node_hours']
    nominal_intake_hours = plan['estimated_resources']['estimated_cpu_node_hours_by_stage']['intake']
    revised_hours = planned_hours-nominal_intake_hours+intake_hours
    test=dict(cores=len(complete),planned=len(canaries),failures=errors,
        observed_rows=rows,first_tranche_bytes=disk_bytes,
        forecast_rows_with_halos=projected_rows,forecast_additional_peak_bytes=projected_disk+final_and_reference_bytes,
        updated_cpu_node_hours=revised_hours,
        input_proofs=dict(complete),method_version=rev,slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(test)
    save_json(first_tranche,test)
    if errors or test['observed_rows']==0:
        raise ValueError('Continuation canaries did not pass')
    if revised_hours>plan['budget_nodes_hours'] or test['forecast_additional_peak_bytes']>1024**4:
        raise ValueError('First tranche exceeds the total registered budget')
    rest=[t for t in tasks if t not in canaries]
    by_box=defaultdict(list)
    for v in rest:by_box[v[1]].append(v)
    waves=[v[i] for i in range(max(map(len,by_box.values()))) for v in by_box.values() if i<len(v)]
    collect(waves,pool)
    control.result()
verify_directory(work/'catalogues',tuple(f'xiao_{box}.csv' for box in sorted(ordered)))
summary=dict(cores=len(complete),planned_cores=len(tasks),failures=errors,
             windows=dict(Counter(row['window'] for row in complete.values())),reference_previous_sha256=digest(previous),code_commit=rev,
             slurm_job=os.environ['SLURM_JOB_ID'])
check_aggregate(summary)
save_json(work/'next_intake_completion_summary.json',summary)
if errors:raise ValueError('Retain scope and repair unfinished continuation inputs')

registry=json.loads(reference.read_text())
old_groups={}
for value in registry['partitions']:
    tag=value['label']
    ledger = json.loads((Path(value['owned'])/'fits.json').read_text())
    found=(Path(next(iter(ledger.values())))/'reference_receipt.json').resolve().parent
    verify_directory(found,('private_c0.npz',))
    if digest(found/'private_c0.npz')!=proof['graphs'][tag]['reference_graph_sha256']:
        raise ValueError('The previous reference graph changed')
    old_groups[tag]=set(read_offsets(found/'private_c0.npz'))
def reference_candidates(item):
    core,region=item
    frames=pd.read_csv(work/'refs'/core/'reference.csv.gz',
        usecols=['group','is_holdout'],dtype={'group':str,'is_holdout':bool})
    return region,set(frames.loc[~frames.is_holdout,'group'])
by_window=defaultdict(set)
with ThreadPoolExecutor(6) as threads:
    for region,keys in threads.map(reference_candidates,tasks):
        by_window[region].update(keys)
candidates=work/'private_verified_next_training_group_candidates.json'
save_json(candidates,{k:sorted(v) for k,v in by_window.items()})
comparison=dict(windows={k:dict(training_group_candidates=len(v),
        shared_previous_profiles={p:len(v&o) for p,o in old_groups.items()}) for k,v in by_window.items()},
    previous_reference_index_sha256=digest(reference),training_candidate_sha256=digest(candidates),
    slurm_job=os.environ['SLURM_JOB_ID'],
    scope='Actual training-group identities present before global support and bootstrap selection.')
check_aggregate(comparison)
save_json(work/'next_verified_shared_group_summary.json',comparison)
records=work/'private_verified_intake_index.json'
save_json(records,complete)
summary.update(input_reused=sum(v['input_reused'] for v in complete.values()),
    reference_reused=sum(v['reference_reused'] for v in complete.values()),
    normalized_detections_with_halos=sum(v['normalized_detections'] for v in complete.values()),
    verified_index_sha256=digest(records))
check_aggregate(summary)
save_json(work/'next_intake_verified_completion_summary.json',summary)
print('VERIFIED_NEXT_INTAKE',summary,flush=True)
