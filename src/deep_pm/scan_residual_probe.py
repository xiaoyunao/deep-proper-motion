"""Assess one centered SDSS field term using same-galaxy repeat detections."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .astrometry import project
from .blind_catalogue_quality import read
from .continuous_reference import deduplicate, read_offsets, reference_frame, verify_directory
from .exposure_catalogues import inside
from .io import digest, save_json
from .public_summary import check_aggregate

DEVELOPMENT = ('w150_110', 'w020_110')
CONFIRMATION = ('w340_110', 'w170_080')
SEED = 13714
MIN_REFERENCE = 30
N_CORES = 4


def extract(task):
    core, fit, box = task
    fit = Path(fit)
    source = (fit/'source_receipt.json').resolve().parent
    shared = (fit/'reference_receipt.json').resolve().parent
    reference = (shared/(core+'_reference_receipt.json')).resolve().parent
    stamp = verify_directory(reference, ('reference.csv.gz',))
    frames = pd.read_csv(reference/'reference.csv.gz', dtype={'reference':str})
    headers = set(pd.read_csv(source/'detections.csv', nrows=0))
    if 'field_id' not in headers:
        raise ValueError('Released SDSS field identity missing')
    verify_directory(source, ('detections.csv',))
    columns = ['detection_id','field_id','band']
    detections = pd.read_csv(source/'detections.csv', usecols=columns)
    frames = frames.loc[inside(frames.ra_deg,frames.dec_deg,box)].copy()
    frames = frames.merge(detections,on='detection_id',how='left',validate='many_to_one')
    frames = frames.loc[frames.field_id.notna() | ~frames.group.str.startswith('SDSS:')]
    return frames, shared/'private_c0.npz', digest(stamp)


def pairs(frame):
    """Order each pair deterministically, count each galaxy once in reweighting."""
    part = frame.sort_values(['reference','group','detection_id'])
    first = part.drop_duplicates('reference')
    other = part.merge(first[['reference','group','field_id','x','y','sigma_e','sigma_n']],
                       on='reference', suffixes=('', '_first'), validate='many_to_one')
    return other.loc[other.group!=other.group_first].copy()


def bootstrap_galaxies(values, ids):
    codes, names = pd.factorize(ids)
    n = len(names)
    if n<5:
        return dict(galaxies=n)
    rng = np.random.default_rng(SEED)
    draws = []
    order = np.argsort(values,axis=0)
    for _ in range(192):
        counts = rng.multinomial(n, np.full(n,1/n))[codes]
        threshold = .68*counts.sum()
        draws.append([values[ix[np.searchsorted(counts[ix].cumsum(),threshold)],j]
                      for j,ix in enumerate(order.T)])
    return dict(galaxies=n, p68_ci95=np.quantile(draws,[.025,.975],axis=0).T)


def analyze(chunks):
    graphs = {r[1] for r in chunks}
    if len(graphs)!=1:
        raise ValueError('A local diagnostic requires one original reference graph')
    graph = graphs.pop()
    offsets, gauges = read_offsets(graph, with_gauge=True)
    data, counts = deduplicate([r[0] for r in chunks])
    data = data.loc[data.group.isin(offsets)].copy()
    data['gauge'] = data.group.map(gauges)
    coords = project(data.ra_deg,data.dec_deg,reference_frame(graph))
    data[['x','y']] = coords - np.array([offsets[k][0] for k in data.group])
    sdss = data.loc[data.group.str.startswith('SDSS:') & data.band.eq('r') & data.field_id.notna()].copy()
    late = data.loc[~data.group.str.startswith('SDSS:') & data.is_holdout].dropna(subset=['band'])
    late['ref_band'] = late.reference.astype(str)+':'+late.band.astype(str)+':'+late.gauge.astype(str)
    repeated_nsc = pairs(late.assign(reference=late.ref_band))
    comp = repeated_nsc[['x','y']].to_numpy()-repeated_nsc[['x_first','y_first']].to_numpy()
    out = dict(reference_sha256=digest(graph),input_reference_sha256=[r[2] for r in chunks],
               reference_counts=counts, nsc_same_band_heldout=dict(pairs=len(comp),
                   **bootstrap_galaxies(np.linalg.norm(comp,axis=1)[:,None], repeated_nsc.reference)),
               components={})
    for gauge, frame in sdss.groupby('gauge'):
        visits = frame.groupby('reference').group.nunique()
        support = frame.loc[frame.reference.isin(visits.index[visits>1])]
        repeat = pairs(support)
        train = repeat.loc[~repeat.is_holdout]
        held = repeat.loc[repeat.is_holdout]
        unique = support.loc[~support.is_holdout].groupby('group')
        eligible = {g for g, rows in unique if rows.reference.nunique()>=MIN_REFERENCE and rows.field_id.nunique()>=4}
        entry = dict(train_repeat_galaxies=train.reference.nunique(), held_repeat_galaxies=held.reference.nunique(),
                     train_pairs=len(train),held_pairs=len(held),candidate_sdss_groups=len(eligible))
        out['components'][str(gauge)] = entry
        if not eligible or train.reference.nunique()<2*len(eligible) or held.reference.nunique()<20:
            entry['status'] = 'insufficient_independent_repeat_support'
            continue
        centers = {}
        for key, rows in unique:
            if key not in eligible:
                continue
            w = (rows.sigma_e**2+rows.sigma_n**2)**-1
            u = rows.field_id.to_numpy(float)
            middle = np.average(u,weights=w)
            centers[key] = middle, np.sqrt(np.average((u-middle)**2,weights=w))
        keys = sorted(centers)
        def design(table):
            matrix = np.zeros((len(table),len(keys)))
            for column,key in enumerate(keys):
                middle, width = centers[key]
                for suffix,sign in (('',1),('_first',-1)):
                    choose = table['group'+suffix].eq(key).to_numpy()
                    matrix[choose,column] += sign*(table['field_id'+suffix].to_numpy(float)[choose]-middle)/width
            return matrix
        def observed(table):
            return table[['x','y']].to_numpy()-table[['x_first','y_first']].to_numpy()
        matrix = design(train)
        errors = np.sqrt(train[['sigma_e','sigma_n']].to_numpy()**2+
                         train[['sigma_e_first','sigma_n_first']].to_numpy()**2)
        singular = np.linalg.svd(matrix / errors.mean(axis=1,keepdims=True),compute_uv=False)
        rank = int(np.count_nonzero(singular>singular[0]*1.e-8))
        condition = float(singular[0]/singular[-1]) if singular[-1]>0 else None
        entry.update(design_rank=rank, parameters=len(keys),condition_number=condition)
        if rank<len(keys) or condition is None or condition>200:
            entry['status'] = 'unstable_relative_field_gauge'
            continue
        coefficients = np.stack([np.linalg.lstsq(matrix/errors[:,i,None],observed(train)[:,i]/errors[:,i],rcond=None)[0] for i in range(2)],axis=1)
        names = np.count_nonzero(design(held),axis=1)>0
        observed_held = observed(held)[names]
        if np.count_nonzero(names)<20:
            entry['status'] = 'insufficient_held_pairs_for_candidate'
            continue
        conditional = observed_held - design(held)[names]@coefficients
        vectors = np.column_stack((np.linalg.norm(observed_held,axis=1),np.linalg.norm(conditional,axis=1)))
        original, adjusted = np.quantile(vectors,.68,axis=0)
        entry.update(status='conditional_test',held_test_pairs=int(names.sum()),
                     held_original_p68_mas=original,held_adjusted_p68_mas=adjusted,
                     fraction_improved=1-adjusted/original,
                     held_repeat_p68_intervals=bootstrap_galaxies(vectors,held.reference.to_numpy()[names]),
                     scope='Window-local conditional test with original C0 fixed. Neither fitted slopes nor their covariance are transferred to a catalogue.')
    return out


def run(frozen, output):
    from .__main__ import tomllib
    summary = frozen/'continuation_production_science_summary.json'
    receipt = read(frozen/'continuation_science_receipt.json')
    if digest(summary)!=receipt['combined_science_sha256']:
        raise ValueError('Changed frozen science')
    scope = read(summary)
    plan = read(frozen/'private_continuation_manifest.json')
    candidates = defaultdict(list)
    for part in plan['partitions']:
        owned = Path(part['owned'])
        verify_directory(owned, ('fits.json','continuous_science_summary.json'))
        if digest(owned/'continuous_science_summary.json')!=receipt['graphs'][part['label']]['science_receipt_sha256']:
            raise ValueError('Changed archived partition')
        fits = read(owned/'fits.json')
        specs = tomllib.loads(Path(part['geometry']).read_text())['fields']
        for core, window in part['region_by_core'].items():
            if window in DEVELOPMENT+CONFIRMATION and scope['cores'][core]['common_r21']['n']>=10:
                candidates[window].append((core, fits[core],specs[core]))
    rng = np.random.default_rng(SEED)
    # Draws in a stable order with the same seed as the v1.8 development probe.
    choices = {}
    for window in DEVELOPMENT+CONFIRMATION:
        possible=sorted(candidates[window])
        if len(possible)<N_CORES:
            raise ValueError('Too few eligible preregistered cores')
        choices[window]=[possible[i] for i in rng.choice(len(possible),N_CORES,replace=False)]
    output.mkdir(parents=True,exist_ok=True)
    evidence={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for w in DEVELOPMENT:
            evidence[w]=analyze(list(pool.map(extract,choices[w])))
            print(w,'complete',flush=True)
        supported = all(any(v.get('fraction_improved',0)>.03 for v in evidence[w]['components'].values())
                        for w in DEVELOPMENT)
        if supported:
            # Region-dependent slopes are only a diagnostic. Transfer requires
            # a shared identifiable parameterization and independent QSO checks.
            for w in CONFIRMATION:
                evidence[w]=analyze(list(pool.map(extract,choices[w])))
                print(w,'complete',flush=True)
    result=dict(windows=evidence,development_repeat_gate=supported, production_adopted=False,
                thresholds=dict(minimum_reference_per_group=MIN_REFERENCE,minimum_conditional_held_p68_gain=.03,
                                zero_reduction_target=.30, maximum_qso_p68_loss=.05,
                                cores_per_window=N_CORES, seed=SEED),
                slurm_job=os.environ['SLURM_JOB_ID'],
                scope='Same physical galaxies and distinct SDSS run/camcol r measurements. Field coordinate is FIELD_ID between fields within a physical scan. Candidate is one training-weight-centered linear field term per supported run/camcol, tested conditionally with the archived C0. A negative development repeat gate closes the method trial before evaluating a QSO correction.')
    check_aggregate(result)
    save_json(output/'v19_scan_residual_summary.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frozen',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use a Slurm compute node')
    run(**vars(args))
