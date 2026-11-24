"""Benchmark real, frozen catalogue cores in separate Slurm output directories."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic


PRODUCTS = ('private_standard_motion.json.gz', 'private_r_control.json.gz',
            'private_known_qso_blocks.json', 'private_blind_qso.json',
            'nsc_science_catalogue_summary.json')
TIMING_FIELDS = ('elapsed_seconds', 'peak_rss_mib', 'slurm_job', 'fit_cache')


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def benchmark_one(args):
    from deep_pm import allband_qso, nsc_science_catalogue as catalogue, precision_v03
    from deep_pm.continuous_reference import verify_directory
    from deep_pm.io import digest, save_json
    from deep_pm.public_summary import check_aggregate

    folder = args.original
    work = args.output
    original_receipt = verify_directory(folder, PRODUCTS)
    timings = {}

    def meter(target, name, label):
        original = getattr(target, name)

        @wraps(original)
        def timed(*a, **kw):
            tick = monotonic()
            try:
                return original(*a, **kw)
            finally:
                stat = timings.setdefault(label, dict(calls=0, seconds=0.0))
                stat['calls'] += 1
                stat['seconds'] += monotonic() - tick
        setattr(target, name, timed)

    for module in (precision_v03, allband_qso):
        if hasattr(module, 'fit_motion'):
            meter(module, 'fit_motion', 'robust_solver')
    meter(catalogue, 'quality_detections', 'standardized_input')
    meter(catalogue, 'make_tracks', 'track_construction')
    meter(catalogue, 'select_catalogue', 'selection_including_fits')
    meter(catalogue, 'evaluated_fit', 'final_state_including_any_fit')
    meter(catalogue, 'stationary_controls', 'stationary_controls')
    source = (folder/'source_receipt.json').resolve().parent
    reference = (folder/'reference_receipt.json').resolve().parent
    geometry = (folder/'geometry.toml').resolve()
    legacy = (folder/'legacy.csv').resolve()
    sys.argv = ['nsc_science_catalogue', str(work), str(source), str(reference),
                str(geometry), str(legacy), args.core]
    start = monotonic()
    catalogue.main()
    elapsed = monotonic() - start
    verify_directory(work, PRODUCTS)
    fit_summary = read(work/PRODUCTS[-1])
    expected = read(folder/PRODUCTS[-1])
    equality = {}
    for name in PRODUCTS:
        before, after = read(folder/name), read(work/name)
        if name == PRODUCTS[-1]:
            before, after = [{k: v for k, v in obj.items() if k not in TIMING_FIELDS}
                             for obj in (before, after)]
        equality[name] = before == after
    result = dict(input_receipt_sha256=digest(original_receipt),
                  reference_epoch=2015.5, all_products_equal=all(equality.values()),
                  equal_products=equality, stage_timings=timings,
                  elapsed_with_input_checks_seconds=elapsed,
                  elapsed_pipeline_seconds=fit_summary['elapsed_seconds'],
                  original_pipeline_seconds=expected['elapsed_seconds'],
                  peak_rss_mib=fit_summary['peak_rss_mib'],
                  input_detections=fit_summary['input_detections'],
                  research_fits=fit_summary['owned_all_nsc'],
                  affinity_cpus=len(os.sched_getaffinity(0)),
                  output_bytes=sum((work/name).stat().st_size for name in PRODUCTS),
                  cache=fit_summary.get('fit_cache'), slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result)
    save_json(work/'performance_summary.json', result)
    if not result['all_products_equal']:
        raise ValueError('Replay changed frozen catalogue content')


def benchmark(args):
    plan = read(args.frozen/'private_continuation_manifest.json')
    ranked = read(args.frozen/'continuation_production_science_summary.json')['cores']
    tasks = []
    for partition in plan['partitions']:
        owned = Path(partition['owned'])
        fits = read(owned/'fits.json')
        candidates = sorted((ranked[k]['research_fits'], k) for k in fits
                            if ranked[k]['research_fits'] >= 300)
        for fraction in (.25, .8):
            _, core = candidates[int(len(candidates)*fraction)]
            tasks.append((core, fits[core]))
    if len({core for core, _ in tasks}) != len(tasks):
        raise ValueError('Representative cores must be distinct')
    args.output.mkdir(parents=True, exist_ok=True)
    def run_task(task):
        j, (core, original) = task
        for tag, snapshot in (('baseline', args.baseline), ('trial', args.trial))[::1 if j % 2 == 0 else -1]:
            output = args.output/str(j)/tag
            logfile = output.parent/(tag+'.log')
            logfile.parent.mkdir(parents=True, exist_ok=True)
            if (output/'performance_summary.json').is_file():
                continue
            env = dict(os.environ, DPM_COMMIT=snapshot.name,
                       PYTHONPATH=str(snapshot/'src')+os.pathsep+os.environ['PYTHONPATH'])
            with logfile.open('a') as log:
                subprocess.run([sys.executable, __file__, '--one', str(output), str(original), core],
                               env=env, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_task, enumerate(tasks)))
    from deep_pm.io import save_json
    from deep_pm.public_summary import check_aggregate
    combined = dict(slurm_job=os.environ['SLURM_JOB_ID'], workers=args.workers,
                    cores=[dict(**{tag:read(args.output/str(j)/tag/'performance_summary.json')
                                   for tag in ('baseline','trial')}) for j in range(len(tasks))])
    check_aggregate(combined)
    save_json(args.output/'v17_catalogue_performance_summary.json', combined)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--one', action='store_true')
    parser.add_argument('output', type=Path)
    args, _ = parser.parse_known_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use a Slurm compute node')
    if args.one:
        parser.add_argument('original', type=Path)
        parser.add_argument('core')
        benchmark_one(parser.parse_args())
    else:
        for name in ('frozen', 'baseline', 'trial'):
            parser.add_argument(name, type=Path)
        parser.add_argument('--workers', type=int, default=6)
        benchmark(parser.parse_args())
