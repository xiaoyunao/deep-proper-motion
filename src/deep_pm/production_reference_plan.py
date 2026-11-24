"""Close real physical reference connections for the registered wide batch."""
import argparse
from collections import defaultdict
import os
from pathlib import Path

import numpy as np

from .__main__ import tomllib
from .blind_catalogue_quality import read
from .continuous_reference import read_offsets, reference_frame, verify_directory
from .io import digest, save_json
from .public_summary import check_aggregate
from .validation_blocks import connected


def projection(fields, old_centers):
    mids = np.radians([[np.mean(fields[c][axis]) for axis in ('ra','dec')] for c in sorted(fields)])
    lon, lat = mids.T
    vectors = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
    if len(old_centers)==1:
        a,b=np.radians(old_centers[0])
    else:
        x,y,z=vectors.mean(axis=0)
        a,b=np.arctan2(y,x),np.arctan2(z,np.hypot(x,y))
    if np.min(vectors@np.array([np.cos(b)*np.cos(a),np.cos(b)*np.sin(a),np.sin(b)]))<=0:
        raise ValueError('A connected physical component crosses its candidate gnomonic horizon')
    return list(map(float,old_centers[0])) if len(old_centers)==1 else [float(np.degrees(a)%360),float(np.degrees(b))]


def run(work, intake, registered, frozen):
    work.mkdir(parents=True,exist_ok=True)
    selection=read(registered/'wide_selection_summary.json')
    bound=frozen/'continuation_science_receipt.json'
    prior=read(bound)
    if digest(bound)!=selection['previous_scope_sha256']:
        raise ValueError('Frozen archive changed since registration')
    proof_path=intake/'next_intake_verified_completion_summary.json'
    proof=read(proof_path)
    tasks=dict(line.split() for line in (intake/'tasks.tsv').read_text().splitlines())
    index=intake/'private_verified_intake_index.json'
    metadata=read(index)
    if proof['failures'] or selection['new_cores']!=len(tasks) or tasks.keys()!=metadata.keys() or digest(index)!=proof['verified_index_sha256']:
        raise ValueError('Incomplete registered input')
    candidate_path=intake/'private_verified_next_training_group_candidates.json'
    overlap=read(intake/'next_verified_shared_group_summary.json')
    if overlap['training_candidate_sha256']!=digest(candidate_path):
        raise ValueError('Training-group inventory changed')
    possible=read(candidate_path)
    old={}
    archives={}
    footprints={}
    prior_manifest=read(frozen/'private_continuation_manifest.json')
    for part in prior_manifest['partitions']:
        tag=part['label']
        owned=Path(part['owned'])
        verify_directory(owned,('fits.json','continuous_science_summary.json'))
        if digest(owned/'continuous_science_summary.json')!=prior['graphs'][tag]['science_receipt_sha256']:
            raise ValueError('Frozen ownership product changed')
        paths=read(owned/'fits.json')
        previous=read(owned/'continuous_science_summary.json')
        if paths.keys()!=part['region_by_core'].keys() or paths.keys()&archives.keys() or paths.keys()&tasks.keys():
            raise ValueError('Overlap or missing cores in existing archive')
        specs=tomllib.loads(Path(part['geometry']).read_text())['fields']
        shared=(Path(next(iter(paths.values())))/'reference_receipt.json').resolve().parent
        graph=shared/'private_c0.npz'
        verify_directory(shared,('private_c0.npz',))
        if digest(graph)!=previous['reference_graph_sha256']:
            raise ValueError('Frozen reference data changed')
        for core,path in paths.items():
            archives[core]=dict(path=path,receipt_sha256=previous['fit_receipt_sha256'][core],
                                reference=str((shared/(core+'_reference_receipt.json')).resolve().parent))
        for b in set(part['region_by_core'].values()):
            footprints[b]=prior_manifest['region_footprints'][b]
        old[tag]=dict(groups=set(read_offsets(graph)),center=reference_frame(graph), specs=specs, entry=part)
    tags=[('new',w) for w in sorted(possible)]+[('old',name) for name in sorted(old)]
    groups=[set(possible[name]) if role=='new' else old[name]['groups'] for role,name in tags]
    assigned={}
    reused={}
    for selected in connected([dict(exposures=sorted(g)) for g in groups]):
        new=[tags[i][1] for i in selected if tags[i][0]=='new']
        previous=[tags[i][1] for i in selected if tags[i][0]=='old']
        if not new:
            if len(previous)!=1:
                raise ValueError('Frozen reference graphs share a physical group')
            reused[previous[0]]=old[previous[0]]['entry']
            continue
        label='v19_'+new[0]
        dest=work/label
        dest.mkdir(exist_ok=True)
        layout=dest/'layout'
        layout.mkdir(exist_ok=True)
        windows=defaultdict(dict)
        geometry=defaultdict(dict)
        for w in new:
            for core,spec in tomllib.loads((intake/f'batch-{w}.toml').read_text())['fields'].items():
                windows[w][core]=str(intake/'refs'/core)
                geometry[w][core]=spec
        for tag in previous:
            for core,w in old[tag]['entry']['region_by_core'].items():
                windows[w][core]=archives[core]['reference']
                geometry[w][core]=old[tag]['specs'][core]
        fields={core:spec for by_window in geometry.values() for core,spec in by_window.items()}
        center=projection(fields,[old[k]['center'] for k in previous])
        for w in sorted(windows):
            save_json(layout/f'references-{w}.json',windows[w])
            # Footprint windows and fitting cores share the identical signed boxes.
            minima=[min(v[k][0] for v in geometry[w].values()) for k in ('ra','dec')]
            maxima=[max(v[k][1] for v in geometry[w].values()) for k in ('ra','dec')]
            lines=[f'[boxes.{w}]',f'ra = {[minima[0],maxima[0]]}',f'dec = {[minima[1],maxima[1]]}']
            for c,s in geometry[w].items():
                lines.extend([f'[fields.{c}]',f'ra = {s["ra"]}',f'dec = {s["dec"]}'])
            (layout/f'batch-{w}.toml').write_text('\n'.join(lines)+'\n')
        assigned[label]=dict(center=center, windows=sorted(windows), old_domains=previous,new_windows=new,
                             new_cores=sum(c in tasks for c in fields),replayed_cores=sum(c in archives for c in fields),
                             region_by_core={c:w for w,entries in windows.items() for c in entries})
    all_cores=[c for r in assigned.values() for c in r['region_by_core']]
    all_cores.extend(c for r in reused.values() for c in r['region_by_core'])
    if len(all_cores)!=len(archives)+len(tasks) or set(all_cores)!=set(archives)|set(tasks):
        raise ValueError('Reference plan does not assign every prior/new core exactly once')
    old_actual=sum(a['replayed_cores'] for a in assigned.values())
    old_estimate=selection['estimated_resources']['estimated_old_replayed_cores']
    canopy=read(intake/'next_intake_canary_summary.json')
    cost=canopy['updated_cpu_node_hours']+(old_actual-old_estimate)*.0203
    manifest=dict(solve=assigned,reuse=reused,archives=archives,prior_footprints=footprints)
    save_json(work/'private_continuation_reference_plan.json',manifest)
    result=dict(new_cores=len(tasks),replayed_cores=old_actual,reused_cores=len(archives)-old_actual,
                forecast_cpu_node_hours=cost,budget_cpu_node_hours=selection['budget_nodes_hours'],
                physical_partitions={k:{f:v for f,v in p.items() if f!='region_by_core'} for k,p in assigned.items()},
                plan_sha256=digest(work/'private_continuation_reference_plan.json'),
                intake_sha256=digest(proof_path),slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result)
    save_json(work/'v19_reference_plan_summary.json',result)
    if cost>selection['budget_nodes_hours']:
        raise ValueError('Real shared physical groups require a larger replay budget than registered')
    print('Reference domains verified',len(assigned),'replayed cores',old_actual,'new',len(tasks),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('work','intake','registered','frozen'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use Slurm compute node')
    run(**vars(args))
