"""v2.2 spherical reference shards from signed immutable normalised inputs.

Private Parquet shards preserve native angles/errors. No old-plane coordinate is
merged across domains; projection is assigned only after the global graph closes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .astrometry import project
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION, SPATIAL, reference_frame, verify_directory
from .io import digest, save_json
from .pilot_production import quality_detections
from .precision_v03 import calibration_groups, reference_rows
from .public_summary import check_aggregate

BASELINE = '75a7abfa883a13d649719eaefe8d676e713bb3b3'


def register(current, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = current/'private_continuation_manifest.json'
    scope = read(manifest)
    entries = []
    graphs = {}
    for partition in scope['partitions']:
        fits = read(Path(partition['owned'])/'fits.json')
        specs = tomllib.loads(Path(partition['geometry']).read_text())['fields']
        for core, window in partition['region_by_core'].items():
            fit = Path(fits[core])
            graph = (fit/'reference_receipt.json').resolve().parent/'private_c0.npz'
            source = (fit/'source_receipt.json').resolve().parent
            if str(graph) not in graphs:
                graphs[str(graph)] = dict(sha256=digest(graph), center=reference_frame(graph), label=partition['label'])
            old_shard = (graph.parent/(core+'_reference_receipt.json')).resolve().parent/'reference.csv.gz'
            entries.append(dict(core=core, window=window, source=str(source), source_receipt_sha256=digest(source/'run_manifest.json'),
                fit=str(fit), old_graph=str(graph), old_shard=str(old_shard), spec=specs[core], **graphs[str(graph)]))
    if len({e['core'] for e in entries}) != len(entries):
        raise ValueError('A core occurs twice in frozen area')
    plan = dict(baseline=BASELINE, manifest_sha256=digest(manifest), cores=entries, graphs=graphs,
        method='spherical-reference-full2d-v22-nativeSE12', budget_cpu_node_hours=48, stellar_budget_cpu_node_hours=2,
        source_shape_minimum=0.8, native_position_floor_mas=50, target_reference_se_native_mas=12,
        bootstrap_draws=48, bootstrap='Bayesian by physical reference, final IRLS fixed',
        gauge='Per connected component, peeled/pre-SE NSC median point; each draw own-group centering then same NSC median')
    save_json(output/'private_full_coordinate_plan.json', plan)
    public = dict(baseline=BASELINE, cores=len(entries), old_graphs=len(graphs),
        plan_sha256=digest(output/'private_full_coordinate_plan.json'), budget_cpu_node_hours=48,
        source_shape_minimum=0.8, method=plan['method'], slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(public)
    save_json(output/'v22_registration_summary.json', public)
    return plan


def extract_one(task):
    entry, destination = task
    output = Path(destination)/entry['core']
    output.mkdir(parents=True, exist_ok=True)
    summary = output/'v22_reference_input_core_summary.json'
    source = Path(entry['source'])
    code = Path(__file__).parent
    ticket = dict(source_receipt_sha256=entry['source_receipt_sha256'],
        implementation={name:digest(code/name) for name in ('full_reference_input.py','precision_v03.py','coordinate_metric.py','__main__.py','pilot_production.py')},
        precision_sha256=digest(PRECISION), spatial_sha256=digest(SPATIAL), center=entry['center'])
    if digest(source/'run_manifest.json') != ticket['source_receipt_sha256']:
        raise ValueError('Frozen source receipt changed')
    if summary.exists():
        result = read(summary)
        if result['ticket'] != ticket or result['parquet_sha256'] != digest(output/'reference.parquet'):
            raise ValueError('Changed reference shard checkpoint')
        return result
    start = monotonic()
    verify_directory(source, ('detections.csv','external_labels.csv','spectra_veto.csv','validation/gaia.csv'))
    config = tomllib.loads(PRECISION.read_text())
    prod = tomllib.loads(SPATIAL.read_text())['production']
    center = entry['center']
    data, xy, _, sigma, _ = quality_detections(source, center, config, prod)
    labels = pd.read_csv(source/'external_labels.csv', dtype=str).set_index('detection_id').original_object_id
    veto = pd.concat([pd.read_csv(source/name, usecols=['ra','dec']) for name in ('spectra_veto.csv','validation/gaia.csv')],ignore_index=True).dropna()
    selected = reference_rows(data, xy, data.detection_id.map(labels), project(veto.ra,veto.dec,center),
        config['calibration'], config['seed'], sky_center=center)
    ix = selected.index.to_numpy()
    frame = selected[['reference','is_holdout','detection_id','exposure_id','ra_deg','dec_deg']].copy()
    frame['group'] = calibration_groups(data)[ix]
    # sigma already contains native 50mas floor; never add it again downstream.
    frame[['sigma_e','sigma_n']] = sigma[ix]
    frame['source_core'] = entry['core']
    frame.to_parquet(output/'reference.parquet', index=False, compression='zstd')
    old = pd.read_csv(entry['old_shard'],usecols=['detection_id'],dtype=str).detection_id
    original, current = set(old), set(frame.detection_id)
    result = dict(ticket=ticket, core=entry['core'], measurements=len(frame), reference_sources=frame.reference.nunique(),
        training_sources=frame.loc[~frame.is_holdout].reference.nunique(), holdout_sources=frame.loc[frame.is_holdout].reference.nunique(),
        groups=frame.group.nunique(), old_measurements=len(old), same_detection_count=len(original&current),
        added_detection_count=len(current-original), exited_detection_count=len(original-current),
        parquet_sha256=digest(output/'reference.parquet'), input_detections=len(data),
        elapsed_seconds=monotonic()-start, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Core halo shard; NOT globally unique reference count. Physical reference split hash unchanged. Native sigma includes floor once.')
    check_aggregate(result)
    save_json(summary,result)
    print('REFERENCE_INPUT',entry['core'],len(frame),round(result['elapsed_seconds'],2),flush=True)
    return result


def run(current, output, workers, pilot):
    if workers < 1 or workers > int(os.environ['SLURM_CPUS_PER_TASK']):
        raise ValueError('Workers exceed allocated CPUs')
    registration = output/'registration'
    plan_path = registration/'private_full_coordinate_plan.json'
    plan = read(plan_path) if plan_path.exists() else register(current, registration)
    selected = plan['cores']
    if pilot:
        # Fixed metadata-only first core per old graph; no fitted PM examined.
        selected = [next(e for e in selected if e['old_graph']==g) for g in plan['graphs']]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(extract_one,[(e,str(output/'input')) for e in selected],chunksize=1))
    report = dict(plan_sha256=digest(plan_path),pilot=pilot, completed_cores=len(results),
        shard_measurements=sum(r['measurements'] for r in results),
        input_detections=sum(r['input_detections'] for r in results),
        added_detection_count=sum(r['added_detection_count'] for r in results),
        exited_detection_count=sum(r['exited_detection_count'] for r in results),
        total_worker_seconds=sum(r['elapsed_seconds'] for r in results),
        maximum_worker_rss_mib=max(r['peak_rss_mib'] for r in results),
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(report)
    save_json(output/('v22_reference_input_pilot_summary.json' if pilot else 'v22_reference_input_summary.json'),report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('current',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--pilot',action='store_true')
    a=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Scientific work requires Slurm')
    run(**vars(a))
