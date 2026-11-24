"""Read the frozen blind catalogue and known-position controls on common spectra."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import tomllib
from .astrometry import project
from .continuous_reference import verify_directory
from .continuous_summary import estimate
from .exposure_catalogues import inside
from .io import digest, save_json
from .nsc_ownership_overlay import STANDARD, record_hash
from .precision_v03 import astrometric_metrics
from .public_summary import check_aggregate

METHODS = ('blind_all', 'known_all', 'physical_r', 'Xiao', 'NSC')


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def source_pool(task):
    core, folder, source_spec, window, graph, moves, receipt_hash, center = task
    folder = Path(folder)
    source = (folder/'source_receipt.json').resolve().parent
    receipt = verify_directory(folder, (STANDARD, 'private_known_qso_blocks.json', 'private_blind_qso.json'))
    if digest(receipt) != receipt_hash:
        raise ValueError('Unexpected fit receipt')
    verify_directory(source, ('spectra_union.csv',))
    spectra = pd.read_csv(source/'spectra_union.csv')
    spectra = spectra.loc[inside(spectra.ra, spectra.dec, source_spec)].reset_index(drop=True)
    blind = read(folder/'private_blind_qso.json')
    known = read(folder/'private_known_qso_blocks.json')
    axes = {name:{r['validation_local_index']:r for r in selection} for name, selection in known.items()}
    catalogue = read(folder/STANDARD)
    exclusions = Counter(core_spectra=len(spectra), local_unique_blind=len(blind), output_sources=len(catalogue))
    if len(spectra) and catalogue:
        coordinates = np.asarray([r['position_deg'] for r in catalogue])
        targets = project(spectra.ra, spectra.dec, center)
        positions = project(*coordinates.T, center)
        tree = cKDTree(targets)
        distance, neighbors = tree.query(positions, k=2)
        nearest = neighbors[:,0]
        in_radius = distance[:,0] <= 500
        good = in_radius & (distance[:,1]-distance[:,0] > 300)
        claims = Counter(map(int, nearest[good]))
        replay = {(int(j), int(nearest[j])) for j in np.flatnonzero(good) if claims[int(nearest[j])] == 1}
        if replay != {(r['source_slot'], r['qso_local_index']) for r in blind}:
            raise ValueError('Blind spectral matching does not replay on the frozen rows')
        exclusions.update(blind_ambiguous_target_sources=sum(n for n in claims.values() if n>1),
                          blind_candidates_within_radius=int(in_radius.sum()),
                          blind_candidates_rejected_by_margin=int((in_radius & ~good).sum()))
    results = []
    for match in blind:
        slot, local = match['source_slot'], match['qso_local_index']
        row = catalogue[slot]
        owner, owner_window = (core, window)
        if slot in moves:
            move, owner_window = moves[slot]
            if record_hash(row) != move['unchanged_record_sha256']:
                raise ValueError('Rehomed state changed')
            owner = move['owner_core']
            if owner is None:
                exclusions['outside_final_owner'] += 1
                continue
        if row['reference_model_sha256'] != graph:
            raise ValueError('Unexpected blind source reference graph')
        qso = spectra.iloc[local]
        magnitude = match['magnitude']
        if magnitude is None or not 21 <= magnitude < 23:
            exclusions['missing_or_outside_spectral_r_range'] += 1
            continue
        if (qso.reference_r != magnitude or match['kind'] != row['kind']
                or match['values']['blind_all'] != [row['pm_icrs_masyr'], row['pm_error_masyr']]):
            raise ValueError('Blind match no longer identifies the original physical fit')
        values = {name:match['values'][name] for name in ('blind_all','Xiao','NSC')
                  if name in match['values']}
        all_block = axes['all_nsc'].get(local)
        physical = axes['r_physical_quality'].get(local)
        legacy = axes['r_legacy_quality'].get(local)
        shape = legacy['median_nsc_r_class_star'] if legacy else None
        if all_block:
            values['known_all'] = [all_block['motion_icrs']['C0'], all_block['error_icrs']['C0']]
        if physical:
            values['physical_r'] = [physical['motion_icrs']['C0'], physical['error_icrs']['C0']]
        # The archived union keeps DESI identity when available and the released
        # DR16Q position for unmatched SDSS spectra. No proper motion selects identity.
        identity = ('DESI', int(qso.target)) if pd.notna(qso.target) else ('DR16Q', float(qso.ra), float(qso.dec))
        info = row['information']
        results.append(dict(identity=identity, core=owner, window=owner_window,
                            kind=row['kind'], magnitude=magnitude, shape=shape,
                            same_visits=(all_block['exposures']==match['exposures']) if all_block else None,
                            values=values, exposures=match['exposures'], nsc_epoch_days=match['nsc_epoch_days'],
                            nsc_seasons=info['nsc_seasons'], nsc_nights=info['nsc_nights'],
                            sdss_runs=info['sdss_physical_runs'], nsc_span=info['nsc_span_years'],
                            joint_span=info['joint_span_years'], instruments='+'.join(row['instruments']),
                            leverage=max(info['largest_information_fraction']['physical_group'])))
    return results, exclusions


def merge_strata(rows, methods, sensitive=False):
    selected = [r for r in rows if set(methods)<=r['values'].keys()]
    result = dict(n=len(selected))
    if len(selected) < 5:
        return result
    result['methods'] = {}
    for name in methods:
        motions, errors = np.asarray([r['values'][name] for r in selected]).transpose(1,0,2)
        center = np.median(motions, axis=0)
        metrics = astrometric_metrics(motions, errors, 5.)
        radial = np.linalg.norm(motions, axis=1)
        metrics.update(radial_abs68=np.quantile(radial,.68), radial_tail_count=int(np.sum(radial>5)),
                       centered_fraction_within_1sigma=np.mean(np.abs(motions-center)<=errors,axis=0),
                       centered_fraction_within_2sigma=np.mean(np.abs(motions-center)<=2*errors,axis=0))
        result['methods'][name] = metrics
    p68 = result['methods']['blind_all']['radial_abs68']
    result['relative_p68_gain'] = {name:1-p68/metrics['radial_abs68'] if metrics['radial_abs68']>0 else None
                                   for name,metrics in result['methods'].items() if name!='blind_all'}
    result['sdss_runs_observed'] = len({e for r in selected for e in r['exposures'] if e.startswith('SDSS:')})
    result['nsc_nights_observed'] = len({n for r in selected for n in r['nsc_epoch_days']})
    if sensitive:
        samples = [dict(r,motion={n:r['values'][n][0] for n in methods}) for r in selected]
        result['visit_weighted_sensitivity'] = estimate(samples,methods,9137,candidate='blind_all')
        result['without_each_window'] = {window:merge_strata([r for r in selected if r['window']!=window],methods)
                                        for window in sorted({r['window'] for r in selected})}
    return result


def summarize(rows):
    cohorts = {}
    identities = Counter(tuple(r['identity']) for r in rows)
    unique = [r for r in rows if identities[tuple(r['identity'])]==1]
    for lower in (21,22):
        selected = [r for r in unique if lower<=r['magnitude']<lower+1]
        strata = defaultdict(list)
        for r in selected:
            strata['all'].append(r)
            strata['window_'+r['window']].append(r)
            strata['class_'+r['kind']].append(r)
            strata['instrument_'+r['instruments']].append(r)
            for key, good in [('multiple_sdss_runs',r['sdss_runs']>1),
                              ('three_nsc_seasons',r['nsc_seasons']>=3),
                              ('five_nsc_nights',r['nsc_nights']>=5),
                              ('nsc_span_ge4',r['nsc_span']>=4),
                              ('joint_span_ge12',r['joint_span']>=12),
                              ('high_leverage',r['leverage']>=.8),
                              ('pointlike',r['shape'] is not None and r['shape']>=.8)]:
                strata[key if good else key+'_complement'].append(r)
            if r['same_visits'] is not None:
                strata['same_exposures' if r['same_visits'] else 'changed_exposures'].append(r)
        cohorts[str(lower)] = {}
        for key, part in sorted(strata.items()):
            cohorts[str(lower)][key] = dict(
                all_blind=merge_strata(part, ('blind_all',)),
                common_published=merge_strata(part, ('blind_all','Xiao','NSC'), key=='all'),
                common_known=merge_strata(part, ('blind_all','known_all')),
                common_physical_r=merge_strata(part, ('blind_all','known_all','physical_r')),
                all_five=merge_strata(part, METHODS),
                missing_xiao=sum('Xiao' not in r['values'] for r in part),
                missing_nsc=sum('NSC' not in r['values'] for r in part),
                missing_known=sum('known_all' not in r['values'] for r in part))
    return dict(cohorts=cohorts, duplicate_spectral_identities=sum(n>1 for n in identities.values()),
                rows_excluded_for_duplicate_spectral_identity=len(rows)-len(unique))


def run(frozen, output, workers):
    config = frozen/'private_continuation_manifest.json'
    plan = read(config)
    receipt = read(frozen/'continuation_science_receipt.json')
    if digest(frozen/'continuation_production_science_summary.json') != receipt['combined_science_sha256']:
        raise ValueError('Frozen combined summary changed')
    output.mkdir(parents=True, exist_ok=True)
    source_hash = digest(config)
    implementation_hash = digest(Path(__file__))
    rows, counts = [], Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in plan['partitions']:
            owned = Path(part['owned'])
            files = ('fits.json','private_owner_overlay.json','continuous_science_summary.json')
            stage = verify_directory(owned, files)
            frozen_summary = owned/'continuous_science_summary.json'
            if digest(frozen_summary) != receipt['graphs'][part['label']]['science_receipt_sha256']:
                raise ValueError('Changed final ownership result')
            specs = tomllib.loads(Path(part['geometry']).read_text())['fields']
            fits, overlay, science = [read(owned/name) for name in files]
            if digest(owned/'fits.json') != overlay['fits_manifest_sha256']:
                raise ValueError('Ownership overlay has a different partition')
            moves = defaultdict(dict)
            for move in overlay['transfers']:
                if move['filename']==STANDARD:
                    owner = move['owner_core']
                    moves[move['origin_core']][move['source_slot']] = move,part['region_by_core'].get(owner)
            from .nsc_science_summary import plan_center
            center = plan_center(Path(next(iter(fits.values()))))
            for chunk, start in enumerate(range(0,len(fits),64)):
                path = output/f"private_{part['label']}_{chunk:03}.json"
                ticket = dict(manifest_sha256=source_hash, receipt_sha256=digest(stage), implementation_sha256=implementation_hash)
                if path.is_file():
                    cached = read(path)
                    if cached['ticket']!=ticket:
                        raise ValueError('Stale blind readout checkpoint')
                    new, counts_chunk = cached['spectra'], cached['counts']
                else:
                    subset = list(sorted(fits))[start:start+64]
                    tasks = ((core, fits[core], specs[core], part['region_by_core'][core], science['reference_graph_sha256'],
                              moves[core], science['fit_receipt_sha256'][core], center) for core in subset)
                    new, counts_chunk = [], Counter()
                    for selected, exits in pool.map(source_pool, tasks, chunksize=2):
                        new.extend(selected)
                        counts_chunk.update(exits)
                    pending = path.with_suffix('.pending')
                    save_json(pending,dict(ticket=ticket,spectra=new,counts=counts_chunk))
                    pending.replace(path)
                rows.extend(new)
                counts.update(counts_chunk)
                print(part['label'], chunk, 'number of faint matches', len(new),flush=True)
    result = dict(**summarize(rows), selection_denominators=dict(counts), slurm_job=os.environ['SLURM_JOB_ID'],
                  manifest_sha256=source_hash, implementation_sha256=implementation_hash,
                  scope='Frozen blind spectral matching and final owner overlay. Equal exposure lists are a diagnostic; exact detection identity requires recovery from the original measurements. Visit reweighting and leave-window-out are sensitivity analyses with shared observations, not independent spatial confidence intervals. External values are the frozen public methods for these same spectral identities.')
    check_aggregate(result)
    save_json(output/'v17_blind_catalogue_quality_summary.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frozen', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use a Slurm compute node')
    run(**vars(args))
