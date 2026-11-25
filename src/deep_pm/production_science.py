import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from deep_pm.__main__ import tomllib
from deep_pm.continuous_reference import verify_directory
from deep_pm.continuous_summary import area
from deep_pm.exposure_catalogues import inside
from deep_pm.io import digest,save_json
from deep_pm.nsc_ownership_overlay import STANDARD, CONTROL
from deep_pm.nsc_science_summary import known_cohorts
from deep_pm.public_summary import check_aggregate

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Use Slurm compute nodes')
parser=argparse.ArgumentParser(description='Signed partition readout, then serial global accumulation.')
for name in ('work','ref','fit','intake'):parser.add_argument(name,type=Path)
parser.add_argument('window_fits',type=Path,nargs='?',help='Required only for uncached native footprint extraction')
parser.add_argument('--phase',choices=('all','partitions','merge'),default='all')
parser.add_argument('--graph',action='append')
args=parser.parse_args()
work,ref,fit,intake,window_fits=(getattr(args,n) for n in ('work','ref','fit','intake','window_fits'))
work.mkdir(parents=True,exist_ok=True)
plan_path=ref/'private_continuation_reference_plan.json'
plan_sha=digest(plan_path)
reference_plan=json.loads(plan_path.read_text())
fit_summary=fit/'next_fit_summary.json'
data=json.loads(fit_summary.read_text())
ledger=fit/'private_next_fits_index.json'
records=json.loads(ledger.read_text())
if len(records)!=data['cores'] or data['fit_ledger_sha256']!=digest(ledger) or data['reference_plan_sha256']!=plan_sha:
    raise ValueError('The signed fit partition is incomplete')
tasks={core:r['region'] for core,r in records.items()}
new_cores=dict(line.split() for line in (intake/'tasks.tsv').read_text().splitlines())
if not new_cores.keys()<=records.keys():
    raise ValueError('Verified fit ledger does not cover all intake cores')
graphs={label:ref/label/'shared' for label in sorted({r['graph'] for r in records.values()})}
for label,folder in graphs.items():
    verify_directory(folder,('private_c0.npz','continuous_reference_summary.json'))
    if data['reference_graphs'][label]!=digest(folder/'private_c0.npz'):
        raise ValueError('The reference graph changed after fit')
nodes=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split()
from time import sleep
from threading import Lock
locks={n:Lock() for n in nodes}

def stage(node,mod,*args,log,allow=False):
    with log.with_suffix('.out').open('a') as out,log.with_suffix('.err').open('a') as err:
        command=['srun','--exclusive','--exact','--cpu-bind=cores','-N1','-n1','-c4',f"--mem={os.environ.get('DPM_SCIENCE_STEP_MEM','155G')}",'-w',node,sys.executable,'-um',mod,*map(str,args)]
        result=subprocess.run(command,stdout=out,stderr=err,check=not allow)
    return result.returncode

def audit_core_ownership(dest,state,paths,specs,raw):
    moves=[]
    totals=Counter()
    spatial=raw['spatial']
    for core,folder in paths.items():
        receipt=verify_directory(folder,(STANDARD,CONTROL))
        if digest(receipt)!=raw['fit_receipt_sha256'][core]:raise ValueError('Original catalogue changed')
        for file,count_key in ((STANDARD,'owned_all_nsc'),(CONTROL,'owned_r_control')):
            expected=spatial[core][count_key]
            with gzip.open(folder/file,'rt') as stream:rows=json.load(stream)
            if len(rows)!=expected:raise ValueError('Catalogue length changed')
            for slot,row in enumerate(rows):
                if inside(*row['position_deg'],specs[core]):continue
                owners=[key for key,bounds in specs.items() if inside(*row['position_deg'],bounds)]
                moves.append(dict(core=core,slot=slot,file=file,record=row,common_epoch_owners=owners))
                totals[file]+=1
    result=dict(source_manifest_sha256=digest(state/'fits.json'),outside_by_type={m:totals[m] for m in (STANDARD,CONTROL)},
        affected_cores=sorted({m['core'] for m in moves}),outside_full_region=sum(not m['common_epoch_owners'] for m in moves),
        internal_owner_changes=sum(len(m['common_epoch_owners'])==1 for m in moves),slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result)
    save_json(dest/'ownership_failure_summary.json',result)
    save_json(dest/'private_ownership_failures.json',moves)
    return result

def collect(graph_name,node):
    actual={c:Path(r['path']) for c,r in records.items() if r['graph']==graph_name}
    shared=Path(graphs[graph_name])
    geometry=shared.parent/'joint_geometry.toml'
    specs=tomllib.loads(geometry.read_text())['fields']
    graph_hash=digest(shared/'private_c0.npz')
    assert set(specs)==set(actual)
    state=work/graph_name
    state.mkdir(exist_ok=True)
    sources={c:str((p/'source_receipt.json').resolve().parent) for c,p in actual.items()}
    saves={c:str(p) for c,p in actual.items()}
    for name,content in [('fits.json',saves),('sources.json',sources)]:
        path=state/name
        if path.exists():
            if json.loads(path.read_text())!=content:raise ValueError('Partition input map differs from existing signed cache')
        else:save_json(path,content)
    checkpoint=state/'private_group_science_receipt.json'
    if checkpoint.exists():
        prior=json.loads(checkpoint.read_text())
        owned=Path(prior['result']['owned'])
        current_index={core:records[core]['receipt_sha256'] for core in actual}
        if (prior['fit_receipt_sha256']!=current_index
                or prior['reference_sha256']!=graph_hash
                or prior['source_manifest_sha256']!=digest(state/'fits.json')):
            raise ValueError('Recorded ready science belongs to another fit or reference version')
        verify_directory(owned,('continuous_science_summary.json','private_owner_overlay.json'))
        for name,checksum in prior['footprint_receipts'].items():
            fp=state/name/'footprint'
            verify_directory(fp,('transfer_footprint_summary.json',))
            if digest(fp/'transfer_footprint_summary.json')!=checksum:
                raise ValueError('Recorded footprint changed after its scientific readout')
        print('REUSE_SCIENCE_PARTITION',graph_name,len(actual),flush=True)
        return prior['result']
    if args.phase=='merge':raise ValueError('Merge requires a completed signed partition; schedule the partition stage first')
    if window_fits is None:raise ValueError('Uncached partition needs the real SDSS window catalogue')
    save_json(state/'private_frozen_plan.json',dict(sources=sources,reference_graph=str(shared/'private_c0.npz'),reference_sha256=graph_hash,
        nominal_area_sqdeg=sum(area(s) for s in specs.values()),source_code_commit=os.environ['DPM_COMMIT']))
    summary=state/'science_preownership_summary.json'
    spectral_path=state/'private_spectral_vectors.json'
    expected_receipts={core:records[core]['receipt_sha256'] for core in actual}
    if not summary.is_file() or not spectral_path.is_file():
        rc=stage(node,'deep_pm.nsc_science_summary',state,geometry,summary,'--private-pool',spectral_path,
                 log=state/'science',allow=True)
    else:
        rc=0
    raw=json.loads(summary.read_text())
    spectral_payload=json.loads(spectral_path.read_text())
    if (raw['reference_graph_sha256']!=graph_hash or raw['fit_receipt_sha256']!=expected_receipts
            or spectral_payload['fit_receipt_sha256']!=expected_receipts
            or spectral_payload['science_summary_sha256']!=digest(summary)):
        raise ValueError('The preownership readout does not match the current fitted inputs')
    del spectral_payload
    if raw['ownership']['cross_core_shared_detection_claims'] or raw['ownership']['exact_position_duplicate_rows']:
        raise ValueError('Repeated source claim needs repair before catalogue release')
    # The readout signs its outputs before raising on an unresolved output owner.
    if rc and not raw['ownership']['outside_common_epoch_owner']:
        raise RuntimeError('Preownership readout failed for an unrelated reason')
    audit=state/'audit'
    proof=audit_core_ownership(audit,state,actual,specs,raw)
    print('OWNERSHIP_AUDIT',graph_name,proof,flush=True)
    stage(node,'deep_pm.nsc_ownership_overlay',state/'owned',state,summary,audit,geometry,'--archive-exterior',log=state/'owned')
    science=state/'owned/continuous_science_summary.json'
    stage(node,'deep_pm.nsc_coverage_conditions',state/'coverage',state,science,log=state/'coverage')
    for box in sorted({tasks[c] for c in actual}):
        w=state/box
        w.mkdir(exist_ok=True)
        cores={c:records[c]['reference_component'] for c in actual if tasks[c]==box}
        geometry=ref/graph_name/'layout'/f'batch-{box}.toml'
        plan=w/'private_window_plan.json'
        save_json(plan,dict(cores=list(cores),core_gauges=cores,reference_sha256=graph_hash,geometry_sha256=digest(geometry)))
        for attempt in range(4):
            result = stage(node,'deep_pm.nsc_transfer_footprint',w/'footprint',plan,geometry,shared,window_fits,
                           log=w/'footprint',allow=True)
            if not result:break
            if attempt == 3:raise RuntimeError(f'Unclosed native footprint for {box}')
            sleep(60*(attempt+1))
    result=dict(label=graph_name,cores=len(actual),science_receipt_sha256=digest(science),
                owned=str(state/'owned'),private_pool=str(state/'private_spectral_vectors.json'),
                geometry=str(shared.parent/'joint_geometry.toml'),region_by_core={k:tasks[k] for k in actual})
    footprints={box:digest(state/box/'footprint/transfer_footprint_summary.json')
                for box in set(result['region_by_core'].values())}
    save_json(checkpoint,dict(result=result,footprint_receipts=footprints,reference_sha256=graph_hash,
        source_manifest_sha256=digest(state/'fits.json'),fit_receipt_sha256={core:records[core]['receipt_sha256'] for core in actual}))
    return result
def process(graph,node):
    with locks[node]:
        return collect(graph,node)

selected=sorted(graphs) if not args.graph else sorted(args.graph)
if not set(selected)<=graphs.keys():raise ValueError('Requested unknown physical reference partition')
if args.graph and args.phase!='partitions':raise ValueError('Partial graph scope is only valid for the partition stage')
with ThreadPoolExecutor(len(nodes)) as pool:
    runs={g:pool.submit(process,g,nodes[i%len(nodes)]) for i,g in enumerate(selected)}
    result={g:r.result() for g,r in runs.items()}
if args.phase=='partitions':
    print('SCIENCE_PARTITIONS_VERIFIED',len(result),flush=True)
    raise SystemExit(0)
for label, entry in reference_plan['reuse'].items():
    if label in result:
        raise ValueError('Duplicate reference domain labels in the combined catalogue')
    result[label]=entry
    original=Path(entry['owned'])
    original_fits=json.loads((original/'fits.json').read_text())
    first=Path(next(iter(original_fits.values())))
    graphs[label]=(first/'reference_receipt.json').resolve().parent
owners=set()
group_union=set()
from deep_pm.continuous_reference import read_offsets
for label,record in result.items():
    region=Path(record['owned'])
    verify_directory(region,('continuous_science_summary.json','private_owner_overlay.json'))
    if digest(region/'continuous_science_summary.json')!=record['science_receipt_sha256']:
        raise ValueError('The ownership summary differs from its signed stage')
    if owners&record['region_by_core'].keys():
        raise ValueError('One output core is claimed by two physical reference domains')
    owners.update(record['region_by_core'])
    groups=set(read_offsets(graphs[label]/'private_c0.npz'))
    if groups&group_union:
        raise ValueError('The final catalogue contains conflicting physical groups')
    group_union.update(groups)
if owners!=set(reference_plan['archives'])|set(new_cores):raise ValueError('Final catalogue ownership is incomplete')
manifest=work/'private_continuation_manifest.json'
footprints=dict(reference_plan['prior_footprints'])
for name in reference_plan['solve']:
    for box in set(result[name]['region_by_core'].values()):
        footprints[box]=str(work/name/box/'footprint/transfer_footprint_summary.json')
save_json(manifest,dict(partitions=list(result.values()),region_footprints=footprints))
science=work/'continuation_production_science_summary.json'
stage(nodes[0],'deep_pm.nsc_wide_readout',manifest,science,work/'private_nside128_index.json.gz',log=work/'readout')
cohorts={};new_known=[];new_blind=[]
for g,r in result.items():
    folder=Path(r['owned']);file=folder/'continuous_science_summary.json'
    aggregate=json.loads(file.read_text())
    if digest(file)!=r['science_receipt_sha256']:raise ValueError('Source changes during final readout')
    ownership=aggregate['ownership']
    if any(ownership[k] for k in ('exact_position_duplicate_rows','outside_common_epoch_owner','cross_core_shared_detection_claims')):
        raise ValueError('Incomplete final ownership audit')
    cohorts[g]=dict(cores=r['cores'],science_receipt_sha256=digest(file),ownership=ownership,
        ownership_resolution=aggregate['ownership_resolution'],acquisition=aggregate['acquisition'],
        reference_graph_sha256=aggregate['reference_graph_sha256'])
    pool=json.loads(Path(r['private_pool']).read_text())
    if pool['fit_receipt_sha256']!=aggregate['fit_receipt_sha256']:raise ValueError('Unverified spectral pool')
    new_known.extend(row for row in pool['known'] if row['core'] in new_cores)
    owners=json.loads((folder/'private_owner_overlay.json').read_text())
    moved={(v['origin_core'],v['source_slot']):v['owner_core'] for v in owners['transfers'] if v['filename']==STANDARD}
    new_blind.extend(row for row in pool['blind'] if moved.get((row['core'],row['source_slot']),row['core']) in new_cores)
from deep_pm.nsc_wide_readout import summarize
spectral=work/'continuation_new_spectral_summary.json'
new_study=dict(summarize(new_known,new_blind),cores=len(new_cores),slurm_job=os.environ['SLURM_JOB_ID'],
    scope='Strict same-source uses known sky positions on the newly registered cores. Blind validation uses independent spectra on the finally owned fits. Xiao2025 uses catalogue DESI motions only.')
check_aggregate(new_study);save_json(spectral,new_study)
proof=dict(graphs=cohorts,slurm_job=os.environ['SLURM_JOB_ID'],source_code_commit=os.environ['DPM_COMMIT'],
           combined_science_sha256=digest(science),new_spectral_sha256=digest(spectral),
           fit_summary_sha256=digest(fit_summary),reference_plan_sha256=plan_sha,
           retained_physical_groups=len(group_union))
check_aggregate(proof);save_json(work/'continuation_science_receipt.json',proof)
print('SCIENCE_VERIFIED',len(owners),len(new_cores),flush=True)
