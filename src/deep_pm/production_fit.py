from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Condition
from time import sleep

import numpy as np
from deep_pm.__main__ import tomllib
from deep_pm.continuous_reference import PRECISION, SPATIAL, read_offsets, verify_directory
from deep_pm.exposure_catalogues import inside
from deep_pm.io import digest,save_json

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Use Slurm compute nodes')
work,reference,intake=map(Path,sys.argv[1:4])
workers_per_node=int(sys.argv[4])
phase='next'
plan_path=reference/'private_continuation_reference_plan.json'
plan=json.loads(plan_path.read_text())
ref_plan_summary=json.loads((reference/'v19_reference_plan_summary.json').read_text())
if digest(plan_path)!=ref_plan_summary['plan_sha256']:
    raise ValueError('Reference plan staleness')
archive=plan['archives']
labels=sorted(plan['solve'])
new=dict(line.split() for line in (intake/'tasks.tsv').read_text().splitlines())
intake_index=intake/'private_verified_intake_index.json'
intake_proof=json.loads((intake/'next_intake_verified_completion_summary.json').read_text())
intake_records=json.loads(intake_index.read_text())
if intake_proof['verified_index_sha256']!=digest(intake_index) or new.keys()!=intake_records.keys():
    raise ValueError('Input receipt ledger changed')
work.mkdir(parents=True,exist_ok=True)
graphs={}; configs={}; categories={}; supported=set()
projections={}; gauges_by_domain={}
for label in labels:
    part=plan['solve'][label]
    folder=reference/label
    graph=folder/'shared'
    receipt=verify_directory(graph,('private_c0.npz','continuous_reference_summary.json'))
    r=read_offsets(graph/'private_c0.npz',with_gauge=True)
    if set(r[0])&supported:
        raise ValueError('Two reference graphs claim a common physical observing group')
    if any(len(value[1])!=48 for value in r[0].values()):
        raise ValueError('Each physical observing group requires all 48 reference draws')
    gauges_by_domain[label]=r[1]
    supported.update(r[0])
    projections[label]=dict(graph_sha256=digest(graph/'private_c0.npz'), fit_receipt_sha256=digest(receipt))
    graphs[label]=graph
    geometry=work/f'production-{label}.toml'
    contents='production_quality_tags = true\n'+(folder/'joint_geometry.toml').read_text()
    if geometry.exists() and geometry.read_text()!=contents:
        raise ValueError('Production geometry changed')
    if not geometry.exists():
        geometry.write_text(contents)
    configs[label]=geometry
    if categories.keys()&part['region_by_core'].keys():
        raise ValueError('Two reference domains claim one core')
    categories.update({core:(label,window) for core,window in part['region_by_core'].items()})
for part in plan['reuse'].values():
    owned=Path(part['owned'])
    old=json.loads((owned/'fits.json').read_text())
    first=Path(next(iter(old.values())))
    graph=(first/'reference_receipt.json').resolve().parent/'private_c0.npz'
    frozen=set(read_offsets(graph))
    if frozen&supported:
        raise ValueError('Refitted and frozen physical reference domains conflict')
    supported.update(frozen)
if not new.keys()<=categories.keys():
    raise ValueError('Reference-closed fit set does not cover all new inputs')
verify_directory(intake/'catalogues',tuple(f'xiao_{w}.csv' for w in sorted(set(new.values()))))
audit={'projections':projections}
progress=work/f'private_{phase}_fits_progress.json'
failures=work/f'{phase}_fit_failures.json'
nodes=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split()
memory_budget=int(os.environ.get("DEEP_PM_FIT_MEMORY_GIB","280"))
available_memory={node:memory_budget for node in nodes}
memory_condition=Condition()

@contextmanager
def reserve_node_memory(node,gib):
    with memory_condition:
        memory_condition.wait_for(lambda:available_memory[node]>=gib)
        available_memory[node]-=gib
    try:
        yield
    finally:
        with memory_condition:
            available_memory[node]+=gib
            memory_condition.notify_all()

def input_paths(core,region):
    if core in new:
        path=intake/'inputs'/core
        if digest(path/'run_manifest.json')!=intake_records[core]['input_sha256']:
            raise ValueError('New input receipt was changed')
        return path,intake/'catalogues'/f'xiao_{region}.csv'
    old=Path(archive[core]['path'])
    proof=verify_directory(old,('nsc_science_catalogue_summary.json',))
    if digest(proof)!=archive[core]['receipt_sha256']:
        raise ValueError('Original immutable fit receipt differs')
    return (old/'source_receipt.json').resolve().parent,(old/'legacy.csv').resolve()

std='private_standard_motion.json.gz'
control='private_r_control.json.gz'

def run(item,node):
    core,box=item
    label,region=categories[core]
    graph=graphs[label]
    source,legacy=input_paths(core,region)
    verify_directory(source,('detections.csv','acquisition_summary.json'))
    measurements=json.loads((source/'acquisition_summary.json').read_text())['normalized_detections']
    step_memory=126 if measurements>12000000 else 58 if measurements>4000000 else 32
    geometry=configs[label]
    out=work/'outputs'/core
    out.mkdir(parents=True,exist_ok=True)
    cmd=['srun','--exclusive','--exact','--cpu-bind=cores','-N1','-n1','-c4',f'--mem={step_memory}G','-w',node,sys.executable,'-um','deep_pm.nsc_science_catalogue',
         str(out),str(source),str(graph),str(geometry),str(legacy),core]
    recorded=out/'run_manifest.json'
    stage=None
    if recorded.is_file():
        manifest=json.loads(recorded.read_text())
        if (manifest['commit']!=os.environ['DPM_COMMIT'] or manifest['region']!=core
                or manifest['config_sha256']!=digest(PRECISION)):
            raise ValueError('Frozen fit manifest belongs to a different version')
        attempts=manifest['stages'].get('nsc-science-catalogue-v15',[])
        if attempts and attempts[-1]['state']=='completed':
            stage=attempts[-1]
    if stage is not None:
        expected={
            'source_receipt.json':source/'run_manifest.json',
            'reference_receipt.json':graph/'run_manifest.json',
            'geometry.toml':geometry,
            'legacy.csv':legacy,
            'production.toml':SPATIAL,
        }
        for alias,item in expected.items():
            bound=out/alias
            if bound.resolve()!=item.resolve() or digest(bound)!=stage['inputs'].get(alias):
                raise ValueError('Cached fit input identity changed')
    else:
        with (out/'stdout.log').open('a') as stdout,(out/'stderr.log').open('a') as stderr:
            with reserve_node_memory(node,step_memory):
                subprocess.run(cmd,stdout=stdout,stderr=stderr,check=True)
    receipt=verify_directory(out,(std,control,'private_known_qso_blocks.json','private_blind_qso.json','nsc_science_catalogue_summary.json'))
    summary=json.loads((out/'nsc_science_catalogue_summary.json').read_text())
    spec=tomllib.loads(geometry.read_text())['fields'][core]
    gauges=gauges_by_domain[label]
    group_counts={}
    outside={}
    for name,key in ((std,'owned_all_nsc'),(control,'owned_r_control')):
        with gzip.open(out/name,'rt') as f:
            rows=json.load(f)
        if len(rows)!=summary[key]:raise ValueError('Signed record and row counts disagree')
        used=set()
        outside[name]=0
        for row in rows:
            outside[name]+=int(not inside(*row['position_deg'],spec))
            state=np.asarray(row['position_deg']+row['pm_icrs_masyr'])
            cov=np.asarray(row['covariance'])
            if state.shape!=(4,) or not np.isfinite(state).all() or not 0<=state[0]<360 or abs(state[1])>90:
                raise ValueError('Invalid ICRS state')
            if cov.shape!=(4,4) or not np.isfinite(cov).all() or not np.allclose(cov,cov.T,atol=1e-9,rtol=1e-9):
                raise ValueError('Invalid conditional covariance')
            if not np.allclose(np.sqrt(np.diag(cov)[[1,3]]),row['pm_error_masyr'],atol=1e-9,rtol=1e-9):
                raise ValueError('Inconsistent conditional covariance')
            ids=row['detection_ids']
            if len(set(ids))!=len(ids) or (name==std and used.intersection(ids)):
                raise ValueError('Repeated physical detection claim')
            used.update(ids)
            if name==std:
                groups=row['physical_groups']
                if (row['reference_epoch']!=2015.5 or row['reference_model_sha256']!=audit['projections'][label]['graph_sha256']
                    or len(groups)!=len(ids) or any(gauges.get(g)!=row['reference_component'] for g in groups)):
                    raise ValueError('Mixed reference and physical groups')
        group_counts[key]=len(rows)
    return core,box,dict(path=str(out),receipt_sha256=digest(receipt),graph=label,region=region,counts=group_counts,
                        boundary_positions=outside,reference_component=summary['reference_gauge'],
                        normalized_detections=measurements, elapsed_seconds=summary['elapsed_seconds'],
                        peak_rss_mib=summary['peak_rss_mib'])

def collect(pool,items,products,errors):
    threads=[pool.submit(run,item,nodes[i%len(nodes)]) for i,item in enumerate(items)]
    for f in as_completed(threads):
        i=threads.index(f);core,box=items[i]
        try:
            _,_,r=f.result()
            products[core]=r
            print('FIT',core,r['counts'],flush=True)
        except Exception as exc:
            errors[core]=type(exc).__name__
            print('FIT_ERROR',core,repr(exc),flush=True)
        save_json(progress,products)
        save_json(failures,errors)


tasks=[(core,region) for core,(graph,region) in categories.items()]
pilot_receipt=json.loads((intake/'next_intake_canary_summary.json').read_text())
pilot=[t for t in tasks if t[0] in pilot_receipt['input_proofs']]
tasks.sort(key=lambda t: -intake_records.get(t[0],{}).get('normalized_detections',500000))
products={};errors={}
with ThreadPoolExecutor(workers_per_node*len(nodes)) as pool:
    collect(pool,pilot,products,errors)
    if errors or not sum(r['counts']['owned_all_nsc'] for r in products.values()):
        raise ValueError('Frozen numerical pilot did not pass')
    save_json(work/f'{phase}_pilot_summary.json',dict(n=len(products),
        row_count=sum(r['normalized_detections'] for r in products.values()),
        largest_worker_rss_mib=max(r['peak_rss_mib'] for r in products.values()),
        longest_worker_seconds=max(r['elapsed_seconds'] for r in products.values()),
        workers_per_node=workers_per_node, slurm_job=os.environ['SLURM_JOB_ID']))
    collect(pool,[t for t in tasks if t not in pilot],products,errors)
if errors:raise ValueError('Retry only the scoped failed fits')
if set(products)!=set(categories):raise ValueError('Missing scoped fit outputs')
union_file=work/f'private_{phase}_fits_index.json'
save_json(union_file,products)
local_fresh={k:v for k,v in products.items() if k in new}
summary=dict(cores=len(products),new_cores=len(local_fresh),affected_old_cores=len(products)-len(local_fresh),
    pre_ownership_new_research=sum(v['counts']['owned_all_nsc'] for v in local_fresh.values()),
    fit_ledger_sha256=digest(union_file),reference_plan_sha256=digest(plan_path),
    reference_graphs={label:projections[label]['graph_sha256'] for label in labels},
    source_code_commit=os.environ['DPM_COMMIT'],slurm_job=os.environ['SLURM_JOB_ID'])
save_json(work/f'{phase}_fit_summary.json',summary)
print('FIT_CLOSURE',phase,summary,flush=True)
