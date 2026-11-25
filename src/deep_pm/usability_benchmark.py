"""Verify draw-chunked statistics on signed real QSO inputs in Slurm."""
import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import monotonic

from .blind_catalogue_quality import read
from .io import digest, json_default, save_json
from .production_replay_readout import load_blind
from .public_summary import check_aggregate


def content(value):
    return sha256(json.dumps(value,default=json_default,sort_keys=True,allow_nan=False).encode()).hexdigest()


def one(sample, implementation, destination):
    value=read(sample)
    spec=importlib.util.spec_from_file_location('deep_pm.benchmark_comparison',implementation)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start=monotonic()
    result=module.comparison(value['values'],value['settings'],value['contrasts'])
    seconds=monotonic()-start
    save_json(destination,dict(result=result,elapsed_seconds=seconds,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        result_sha256=content(result),implementation_sha256=digest(implementation),
        input_sha256=digest(sample),affinity_cpus=len(os.sched_getaffinity(0))))


def run(frozen, blind, baseline, output):
    output.mkdir(parents=True,exist_ok=True)
    rows=load_blind(frozen,blind)
    required=('blind_all','known_all','physical_r','Xiao','NSC')
    chosen=[r for r in rows.values() if 21<=r['magnitude']<22 and set(required)<=r['values'].keys()][:40001]
    if len(chosen)<20000:raise ValueError('Real equivalence sample too small')
    sample=output/'private_comparison_sample.json'
    save_json(sample,dict(values=[{k:r['values'][k] for k in required} for r in chosen],
        settings=dict(seed=9137,bootstrap_draws=800,min_publish_count=5,tail_masyr=5,bootstrap_absolute_zero=True),
        contrasts=[('blind_all',k) for k in required[1:]]))
    current=Path(__file__).parent/'nsc_comparison.py'
    for name,path in [('baseline',baseline/'src/deep_pm/nsc_comparison.py'),('chunked',current)]:
        target=output/(name+'.json')
        subprocess.run([sys.executable,'-m','deep_pm.usability_benchmark','one',str(sample),str(path),str(target)],check=True)
    old,new=[read(output/(name+'.json')) for name in ('baseline','chunked')]
    if old['result']!=new['result']:raise ValueError('Chunked real statistics changed')
    cache=output/'cached_comparison.json'
    save_json(cache,dict(input_sha256=digest(sample),implementation_sha256=digest(current),result=new['result']))
    start=monotonic(); reused=read(cache)
    if reused['input_sha256']!=digest(sample) or reused['implementation_sha256']!=digest(current):raise ValueError('Stale real statistic cache')
    if reused['result']!=old['result']:raise ValueError('Cached real statistics changed')
    report=dict(n=len(chosen),bootstrap_draws=800,methods=list(required),contrasts=4,
        exact_result_equal=True,result_sha256=new['result_sha256'],
        baseline={k:v for k,v in old.items() if k!='result'},
        chunked={k:v for k,v in new.items() if k!='result'},
        cache_reuse_seconds=monotonic()-start,cache_bytes=cache.stat().st_size,
        slurm_job=os.environ['SLURM_JOB_ID'],scope='Same real spectral objects, frozen positions and formal errors, original object order, 800 paired draws and quantiles. Independent subprocess memory peaks; cache ticket includes input and implementation. No fitted science state changed.')
    check_aggregate(report);save_json(output/'v20_statistics_performance_summary.json',report)
    print(json.dumps(report,default=json_default),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('one')
    for n in ('sample','implementation','destination'):a.add_argument(n,type=Path)
    a=sub.add_parser('run')
    for n in ('frozen','blind','baseline','output'):a.add_argument(n,type=Path)
    args=vars(p.parse_args());mode=args.pop('mode')
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm compute resources')
    (one if mode=='one' else run)(**args)
