"""Compare fixed-reference galaxy differences for equal filter pairs on/off night."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .continuous_reference import bind, read_offsets, verify_directory
from .io import digest, save_json
from .nsc_information import night_labels
from .provenance import run_stage
from .public_summary import check_aggregate


def robust_statistics(y, native, reference):
    median = np.median(y,axis=0)
    diag = (1.4826*np.median(abs(y-median),axis=0))**2
    return dict(median_mas=median, nmad_mas=np.sqrt(diag),
                median_native_difference_variance_mas2=np.median(native,axis=0),
                median_reference_difference_variance_mas2=np.median(reference,axis=0),
                w0_standardized_nmad=1.4826*np.median(abs((y-median)/np.sqrt(native+reference+2*50**2)),axis=0))


def choose_pairs(block):
    """Adjacent visits limit long-term changes; compare identical band combinations."""
    by_class = {}
    for a in range(len(block)):
        for b in range(a+1,len(block)):
            left,right = block[a],block[b]
            day_gap = right.night-left.night
            if day_gap>30:
                break
            if left.instrument != right.instrument or left.group == right.group:
                continue
            bands = tuple(sorted((left.band,right.band)))
            family = (left.instrument, *bands)
            route = 'same_night' if day_gap==0 else 'different_night'
            key = family,route
            gap = right.observation_time-left.observation_time
            if key not in by_class or gap<by_class[key][0]:
                by_class[key] = gap,left,right
    return by_class


def paired_night_contrast(rows):
    same, different = (np.asarray([r[j][:3] for r in rows]) for j in (0,1))
    nights = np.asarray([list(r[0][3])+list(r[1][3]) for r in rows])
    same_nights = len(np.unique(nights[:,0]))
    result = dict(comparisons=len(rows),same_arm_nights=same_nights,
                  different_arm_nights=len(np.unique(nights[:,2:])))
    if len(rows)<50 or same_nights<5:
        return result

    def variance(data):
        y = data[:,0]
        scatter = (1.4826*np.median(abs(y-np.median(y,axis=0)),axis=0))**2
        return scatter - np.median(data[:,1]+data[:,2],axis=0)

    # Native and the synchronized-reference difference enter each arm explicitly.
    # Variance difference over two estimates a common variance only if the
    # remaining marginals are equal in the two arms.
    delta = (variance(different)-variance(same))/2
    groups = [np.flatnonzero(nights[:,0]==day) for day in np.unique(nights[:,0])]
    rng = np.random.default_rng(9319)
    draws = []
    for draw in rng.integers(len(groups),size=(320,len(groups))):
        index = np.concatenate([groups[i] for i in draw])
        draws.append((variance(different[index])-variance(same[index]))/2)
    return dict(result,common_variance_proxy_mas2=delta,
                same_arm_block_sensitivity95_mas2=np.quantile(draws,[.025,.975],axis=0).T,
                same_night=robust_statistics(*same.transpose(1,0,2)),
                different_night=robust_statistics(*different.transpose(1,0,2)))


def run(work, state, plan_path):
    plan = json.loads(plan_path.read_text())
    graph = (Path(plan['original_state'])/'shared/private_c0.npz')
    if digest(graph) != plan['reference_sha256']:
        raise ValueError('Frozen C0 graph hash changed')
    solution,gauges = read_offsets(graph,with_gauge=True)
    parts = json.loads((Path(plan['original_state'])/'references.json').read_text())
    sources = json.loads((state/'sources.json').read_text())
    # Geometrically dispersed controls and both previously named C pilot cores.
    cores = ('repeat_rich','repeat_sparse','c_01_04','c_03_07','c_06_04','c_08_09')
    reference_files = {}
    taken = set()
    samples = defaultdict(list)
    paired = defaultdict(list)
    paired_seen = set()
    exposure_pairs = defaultdict(set)
    night_sets = defaultdict(set)
    for core in cores:
        folder = Path(parts[core])
        receipt = verify_directory(folder,('reference.csv.gz',))
        reference_files[core] = digest(receipt)
        ref = pd.read_csv(folder/'reference.csv.gz',dtype={'reference':str})
        ref = ref.loc[ref.is_holdout & ref.group.str.startswith('NSC:') & ref.group.isin(solution)].copy()
        ref = ref.drop_duplicates('detection_id').drop_duplicates(['reference','group'],keep=False)
        source = Path(sources[core])
        receipt = verify_directory(source,('detections.csv',))
        source_ids = set(ref.detection_id)
        meta = pd.concat([chunk.loc[chunk.detection_id.isin(source_ids)] for chunk in pd.read_csv(
            source/'detections.csv',chunksize=250000,usecols=['detection_id','instrument','band','observation_time',
                'sigma_east_mas','sigma_north_mas','magnitude','magnitude_error','quality_flags'])],ignore_index=True)
        ref = ref.merge(meta,on='detection_id',validate='one_to_one')
        flags = pd.to_numeric(ref.quality_flags,errors='coerce').fillna(255).astype(int)
        ref = ref.loc[ref.instrument.notna() & ref.band.notna()
                      & np.isfinite(ref.observation_time) & flags.between(0,255) & ((flags&252)==0)
                      & ref.magnitude_error.between(0,2.5/np.log(10)/5,inclusive='right')].copy()
        ref['night'] = night_labels(ref)
        ref['gauge'] = ref.group.map(gauges)
        ref.sort_values(['reference','gauge','observation_time','detection_id'],inplace=True)
        added = 0
        for (identity,gauge),frame in ref.groupby(['reference','gauge'],sort=False):
            # Keep the target independent of the C0 graph fit. Repeated references
            # in overlapping halos can contribute a given comparison only once.
            split = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:2],'big') % 3
            role = 'held' if split==0 else 'development'
            pairs = choose_pairs(list(frame.itertuples(index=False)))
            current = defaultdict(dict)
            for (family,route),(_,a,b) in pairs.items():
                va,vb = solution[a.group],solution[b.group]
                y = (np.array([b.east,b.north])-vb[0]) - (np.array([a.east,a.north])-va[0])
                native = np.square([a.sigma_east_mas,a.sigma_north_mas])+np.square([b.sigma_east_mas,b.sigma_north_mas])
                refvar = np.var(vb[1]-va[1],axis=0,ddof=1)
                current[family][route] = y,native,refvar,(a.night,b.night)
                guard = identity,family,route
                if guard in taken:
                    continue
                taken.add(guard)
                key = '/'.join((*family,role,route))
                samples[key].append((y,native,refvar))
                exposure_pairs[key].add((a.group,b.group))
                night_sets[key].update([a.night,b.night])
                added += 1
            for family,arms in current.items():
                guard = identity,family
                if len(arms)==2 and guard not in paired_seen:
                    paired_seen.add(guard)
                    paired['/'.join((*family,role))].append((arms['same_night'],arms['different_night']))
        print(core, len(ref), added, flush=True)
    report = {}
    for key, rows in samples.items():
        stats = dict(comparisons=len(rows), physical_exposure_pairs=len(exposure_pairs[key]), utc_nights=len(night_sets[key]))
        if len(rows)>=20:
            stats.update(robust_statistics(*(np.asarray(v) for v in zip(*rows))))
        report[key] = stats
    result = dict(cores=cores, reference_shard_receipts=reference_files, comparison_cells=report,
                  same_galaxy_contrasts={k:paired_night_contrast(v) for k,v in paired.items()},
                  graph_sha256=digest(graph), slurm_job=os.environ['SLURM_JOB_ID'],
                  scope='C0 holdout galaxies only. Same instrument and unordered filter pair, distinct exposures, adjacent pairs within 30 days. Closest eligible pair per galaxy/route/family, first registered core in overlaps. Development and held galaxy identities disjoint. Same and different night comparison populations can differ. Pair bootstrap uses the variance of the synchronized reference difference; observed variance also includes galaxy centroid/seeing variation and unmodeled exposure effects. Paired contrasts use identical galaxies in both arms. The same-arm night bootstrap describes sensitivity to that side of the sampling and cannot rule out dependence through shared between-night exposures. Instrument-wide night covariance requires positive, consistent proxy across supported filter combinations in both splits, with shared-marginal assumptions and target verification.')
    check_aggregate(result)
    save_json(work/'night_pair_summary.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('work','state','geometry'):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use Slurm')
    args.work.mkdir(parents=True,exist_ok=True)
    bind(args.work,'frozen_receipt.json',verify_directory(args.state,('sources.json','private_frozen_plan.json')))
    with run_stage(args.work,args.geometry,'C','nsc-night-pairs-v16',('frozen_receipt.json',),('night_pair_summary.json',)) as go:
        if go:
            run(args.work,args.state,args.state/'private_frozen_plan.json')


if __name__=='__main__':
    main()
