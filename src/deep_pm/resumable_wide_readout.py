"""Cache unchanged wide-summary calculations before rebuilding the owned index."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

from deep_pm import nsc_wide_readout as wide
from deep_pm.blind_catalogue_quality import read
from deep_pm.continuous_reference import verify_directory
from deep_pm.io import digest, json_default, save_json
from deep_pm.nsc_ownership_overlay import STANDARD
from deep_pm.nsc_science_summary import SETTINGS
from deep_pm.public_summary import check_aggregate


def content_hash(value):
    encoded = json.dumps(value, default=json_default, separators=(',', ':'), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


def cached_value(folder, key, ticket):
    path = folder/(key+'.json')
    if not path.is_file():
        return None
    value = read(path)
    if value['ticket'] != ticket or content_hash(value['summary']) != value['summary_sha256']:
        raise ValueError('The wide-statistic checkpoint changed')
    return value['summary']


def calculate(task):
    key, known, blind, folder, implementation = task
    ticket = dict(input_sha256=key, implementation_sha256=implementation)
    result = cached_value(folder,key,ticket)
    if result is None:
        result = wide.summarize(known,blind)
        check_aggregate(result)
        save_json(folder/(key+'.json'), dict(ticket=ticket,summary=result,summary_sha256=content_hash(result)))
    return key


def spectral_tasks(manifest):
    known, blind = defaultdict(list), defaultdict(list)
    config = read(manifest)
    regions = set()
    for part in config['partitions']:
        owned = Path(part['owned'])
        verify_directory(owned,('fits.json','private_owner_overlay.json','continuous_science_summary.json'))
        path = owned/'continuous_science_summary.json'
        if digest(path) != part['science_receipt_sha256']:
            raise ValueError('The owned scientific catalogue changed')
        summary, spectral = read(path), read(Path(part['private_pool']))
        if spectral['fit_receipt_sha256'] != summary['fit_receipt_sha256']:
            raise ValueError('The spectrum pool belongs to different fits')
        by_core = part['region_by_core']
        if set(by_core) != set(summary['spatial']):
            raise ValueError('Missing owned region assignment')
        regions.update(by_core.values())
        for row in spectral['known']:
            known[by_core[row['core']]].append(row)
        overlay = read(owned/'private_owner_overlay.json')
        moves = {(r['origin_core'],r['source_slot']):r['owner_core']
                 for r in overlay['transfers'] if r['filename']==STANDARD}
        for row in spectral['blind']:
            owner = moves.get((row['core'],row['source_slot']),row['core'])
            if owner is not None:
                blind[by_core[owner]].append(row)
    tasks = [(region,known[region],blind[region]) for region in sorted(regions)]
    # Preserve the original per-survey insertion order in the pooled calculation.
    tasks.append(('aggregate',[r for v in known.values() for r in v],
                  [r for v in blind.values() for r in v]))
    return tasks


def run(manifest, output, index, cache, workers, probe=False):
    cache.mkdir(parents=True,exist_ok=True)
    sources = Path(wide.__file__).parent
    binding = {path.name:digest(path) for path in sorted(sources.glob('*.py'))}
    implementation = content_hash(dict(source_files=binding,settings=SETTINGS,
                                       recovery_source_sha256=digest(Path(__file__))))
    tasks = spectral_tasks(manifest)
    if probe:
        label,known,blind = next(task for task in tasks if len(task[1])>=25)
        known,blind = known[:25],blind[:25]
        expected = content_hash(wide.summarize(known,blind))
        key = content_hash((known,blind))
        task = (key,known,blind,cache,implementation)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            assert list(pool.map(calculate,[task])) == [key]
        actual = cached_value(cache,key,dict(input_sha256=key,implementation_sha256=implementation))
        if content_hash(actual) != expected:
            raise ValueError('Parallel/cached statistics differ from frozen serial statistics')
        print('EXACT_SERIAL_PARALLEL_CACHE_PROBE',label,len(known),len(blind),expected,flush=True)
        return
    unique = {}
    labels = {}
    for label,known,blind in tasks:
        key = content_hash((known,blind))
        labels.setdefault(key,[]).append(label)
        unique[key] = (key,known,blind,cache,implementation)
    pending = sorted(unique.values(),key=lambda task:-(len(task[1])+len(task[2])))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for key in pool.map(calculate,pending,chunksize=1):
            print('STATISTICS_READY',','.join(labels[key]),flush=True)
    original = wide.summarize

    def frozen_statistics(known,blind):
        key = content_hash((known,blind))
        ticket = dict(input_sha256=key,implementation_sha256=implementation)
        result = cached_value(cache,key,ticket)
        if result is None:
            raise ValueError('The index readout encountered an unregistered statistic sample')
        return result

    wide.summarize = frozen_statistics
    try:
        wide.run(manifest,output,index)
    finally:
        wide.summarize = original
    save_json(cache/'resumable_readout_summary.json',dict(
        source_manifest_sha256=digest(manifest),implementation_sha256=implementation,
        scientific_summary_sha256=digest(output),index_sha256=digest(index),
        workers=workers,statistic_samples=len(unique),slurm_job=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','output','index','cache'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--probe',action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or args.workers<1:
        parser.error('Use a Slurm allocation and positive worker count')
    run(**vars(args))
