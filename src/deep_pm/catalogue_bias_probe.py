"""Replay frozen QSO membership and diagnose spatial/temporal C0 residuals."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .allband_qso import evaluated_fit, physical_candidates
from .astrometry import fit_motion, icrs_motion, project
from .blind_catalogue_quality import read
from .continuous_reference import PRECISION, SPATIAL, read_offsets, reference_frame, verify_directory
from .exposure_catalogues import inside
from .io import digest, save_json
from .nsc_comparison import qso_matches
from .nsc_ownership_overlay import STANDARD
from .pilot_production import quality_detections, valid_epochs
from .precision_v03 import calibration_groups, calibration_support
from .public_summary import check_aggregate

WINDOWS = ('w150_110','w020_110','w130_090','north_three')
GATES = dict(seed=13714, cores_per_window=4, minimum_archived_common_r21=10,
             minimum_held_galaxies_per_sdss_group=12, minimum_held_galaxies_per_field_half=8,
             target_zero_reduction=.30, maximum_p68_degradation=.05)


def summarize(values):
    result = dict(n=len(values))
    if len(values)>=5:
        v = np.asarray(values)
        median = np.median(v,axis=0)
        result.update(median=median,nmad=1.4826*np.median(np.abs(v-median),axis=0),
                      absolute_p68=np.quantile(np.linalg.norm(v,axis=1),.68))
    return result


def galaxy_controls(reference, center, offsets, spec, data):
    verify_directory(reference, ('reference.csv.gz',))
    frames = pd.read_csv(reference/'reference.csv.gz', dtype={'reference':str})
    frames = frames.loc[frames.is_holdout & frames.group.isin(offsets)].drop_duplicates('detection_id')
    p = project(frames.ra_deg, frames.dec_deg, center)
    corrected = p-np.asarray([offsets[k][0] for k in frames.group]).reshape(-1,2)
    frames[['east','north']] = corrected
    sdss = frames.group.str.startswith('SDSS:')
    axes = ['east','north']
    later = frames.loc[~sdss].groupby('reference')[axes].median().rename(columns={x:x+'_nsc' for x in axes})
    early = frames.loc[sdss & inside(frames.ra_deg,frames.dec_deg,spec)].join(later,on='reference').dropna(subset=['east_nsc'])
    early = early.assign(dx=early.east-early.east_nsc,dy=early.north-early.north_nsc)
    delta = ['dx','dy']
    field = data.set_index('detection_id')['field_id'] if 'field_id' in data else None
    if field is not None:
        early['field'] = early.detection_id.map(field)
    group_location = early.groupby(['group','reference'])[delta].median().reset_index()
    prediction, gradients = {}, []
    for group, part in group_location.groupby('group'):
        if len(part) >= GATES['minimum_held_galaxies_per_sdss_group']:
            prediction[group] = part[delta].median().to_numpy()
        if field is None:
            continue
        measurements = early.loc[(early.group==group)&early.field.notna()]
        boundary = measurements.field.median()
        first = measurements.loc[measurements.field<=boundary].groupby('reference')[delta].median()
        second = measurements.loc[measurements.field>boundary].groupby('reference')[delta].median()
        if min(len(first),len(second)) >= GATES['minimum_held_galaxies_per_field_half']:
            gradients.append((second.median()-first.median()).to_numpy())
    return prediction, gradients, early.groupby('reference')[delta].median().to_numpy()


def one_core(task):
    folder, core, spec = task
    folder = Path(folder)
    receipt = verify_directory(folder, (STANDARD,'private_blind_qso.json','private_known_qso_blocks.json'))
    source = (folder/'source_receipt.json').resolve().parent
    verify_directory(source, ('detections.csv','spectra_union.csv'))
    shared = (folder/'reference_receipt.json').resolve().parent
    graph = shared/'private_c0.npz'
    config = tomllib.loads(PRECISION.read_text())
    prod = tomllib.loads(SPATIAL.read_text())['production']
    epoch = config['reference_epoch']
    center = reference_frame(graph)
    all_offsets, gauges = read_offsets(graph,with_gauge=True)
    gauge = read(folder/'nsc_science_catalogue_summary.json')['reference_gauge']
    offsets = {g:v for g,v in all_offsets.items() if gauges[g]==gauge}
    data, xy, time, sigma, keep = quality_detections(source, center, config, prod)
    groups = calibration_groups(data)
    support = calibration_support(groups,offsets)
    nsc = data.survey.eq('NSC').to_numpy()
    physical = physical_candidates(data)
    full = (keep&(~nsc|physical)) | (nsc&physical&support)
    qso = pd.read_csv(source/'spectra_union.csv')
    qso = qso.loc[inside(qso.ra,qso.dec,spec)].reset_index(drop=True)
    _, truth = qso_matches(data,xy,full,project(qso.ra,qso.dec,center))
    known = read(folder/'private_known_qso_blocks.json')['all_nsc']
    blind = {r['qso_local_index']:r for r in read(folder/'private_blind_qso.json')}
    catalogue = read(folder/STANDARD)
    reference = (shared/(core+'_reference_receipt.json')).resolve().parent
    local, gradients, galaxies = galaxy_controls(reference,center,offsets,spec,data)
    areas = np.degrees(np.ptp(np.sin(np.deg2rad(spec['dec']))))*np.ptp(spec['ra'])
    rows = []
    for block in known:
        j = block['validation_local_index']
        members = truth[j]
        ix = members[support[members]&full[members]]
        state = evaluated_fit(ix,data,xy,time,sigma,groups,offsets,center,epoch)
        np.testing.assert_allclose(state['pm_icrs_masyr'],block['motion_icrs']['C0'],atol=1e-8,rtol=1e-10)
        ids = data.detection_id.iloc[ix].tolist()
        t = time[ix]
        w = state['robust_weight'][:,None]/sigma[ix]**2
        dt = t[:,None]-(t[:,None]*w).sum(axis=0)/w.sum(axis=0)
        h = w*dt/(w*dt**2).sum(axis=0)
        held_prediction = None
        early = ~nsc[ix]
        linked = [local.get(g) for g in groups[ix][early]]
        if all(v is not None for v in linked):
            plane_shift = (h[early]*np.asarray(linked)).sum(axis=0)
            state_position = state['tangent_state'][[0,2]]
            held_prediction = icrs_motion(state_position,plane_shift,center)['pm_icrs_masyr']
        row = dict(magnitude=block['magnitude'], known=state['pm_icrs_masyr'], predicted=held_prediction,
                   span=np.ptp(t), runs=len(set(data.exposure_id.iloc[ix[early]])))
        if j in blind:
            found = catalogue[blind[j]['source_slot']]
            saved = found['detection_ids']
            row['membership'] = ('same_order' if ids==saved else 'same_set' if set(ids)==set(saved) else 'changed')
            row['overlap'] = len(set(ids)&set(saved))/len(set(ids)|set(saved))
            row['blind_minus_known'] = np.asarray(found['pm_icrs_masyr'])-state['pm_icrs_masyr']
        visits = data.exposure_id.iloc[ix].to_numpy()
        remove = visits[np.argmin(t)]
        remain = ix[visits!=remove]
        row['full_without_first_run'] = bool(valid_epochs(remain,nsc,time,prod))
        row['remaining_span'] = np.ptp(time[remain])
        if len(remain)>=3 and np.ptp(time[remain])>=1:
            # NSC-only diagnostic when the sole SDSS run is removed.
            correction = np.asarray([offsets[g][0] for g in groups[remain]])
            alternate = fit_motion(xy[remain]-correction,time[remain],sigma[remain],epoch)
            row['without_first'] = icrs_motion(*alternate.coeff,center)['pm_icrs_masyr']
        rows.append(row)
    return rows, galaxies, gradients, areas, digest(receipt), len(local)


def run(frozen,output,workers):
    survey = read(frozen/'continuation_production_science_summary.json')
    receipt = read(frozen/'continuation_science_receipt.json')
    if digest(frozen/'continuation_production_science_summary.json')!=receipt['combined_science_sha256']:
        raise ValueError('Frozen survey summary changed')
    plan = read(frozen/'private_continuation_manifest.json')
    options = defaultdict(list)
    for part in plan['partitions']:
        owned = Path(part['owned'])
        verify_directory(owned, ('fits.json','continuous_science_summary.json'))
        if digest(owned/'continuous_science_summary.json')!=receipt['graphs'][part['label']]['science_receipt_sha256']:
            raise ValueError('Changed partition science')
        fits = read(owned/'fits.json')
        specs = tomllib.loads(Path(part['geometry']).read_text())['fields']
        for core, window in part['region_by_core'].items():
            if window in WINDOWS and survey['cores'][core]['common_r21']['n']>=GATES['minimum_archived_common_r21']:
                options[window].append((fits[core],core,specs[core]))
    rng = np.random.default_rng(GATES['seed'])
    result = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for window in WINDOWS:
            candidates = sorted(options[window],key=lambda t:t[1])
            chosen = [candidates[j] for j in rng.choice(len(candidates),GATES['cores_per_window'],replace=False)]
            chunks = list(pool.map(one_core,chosen))
            rows = [r for chunk,*_ in chunks for r in chunk]
            galaxies, gradients = ([v for chunk in chunks for v in chunk[j]] for j in (1,2))
            result[window] = dict(nominal_sqdeg=sum(c[3] for c in chunks), eligible_cores=len(candidates),
                                  cores=len(chunks), source_receipt_hashes=[c[4] for c in chunks],
                                  locally_supported_sdss_group_core_pairs=sum(c[5] for c in chunks),
                                  independent_galaxy_cross_survey_mas=summarize(galaxies),
                                  field_half_change_mas=summarize(gradients),magnitude_bins={})
            for low in (21,22):
                part = [r for r in rows if low<=r['magnitude']<low+1]
                matched = [r for r in part if r['predicted'] is not None]
                deletion = [r for r in part if 'without_first' in r]
                group = dict(known_all=summarize([r['known'] for r in part]),
                             membership_counts=dict(Counter(r.get('membership','no_blind') for r in part)),
                             observed_on_galaxy_supported_subset=summarize([r['known'] for r in matched]),
                             held_galaxy_implied_motion=summarize([r['predicted'] for r in matched]),
                             after_subtracting_conditional_prediction=summarize([np.asarray(r['known'])-r['predicted'] for r in matched]),
                             eligible_after_remove_first_sdss_run=sum(r['full_without_first_run'] for r in part),
                             deletion_original=summarize([r['known'] for r in deletion]),
                             deletion_fit=summarize([r['without_first'] for r in deletion]),
                             deletion_remaining_span_median_years=np.median([r['remaining_span'] for r in deletion]) if deletion else None)
                for key in ('same_order','same_set','changed'):
                    group[key+'_blind_difference']=summarize([r['blind_minus_known'] for r in part if r.get('membership')==key])
                result[window]['magnitude_bins'][str(low)] = group
            print(window, len(rows),flush=True)
    report = dict(windows=result, preregistered=GATES, slurm_job=os.environ['SLURM_JOB_ID'],
                  scope='Previously inspected windows. Held galaxy local SDSS minus NSC residuals propagate through fixed QSO slope influence; this conditional diagnostic does not recalibrate C0. Deleting a first SDSS run can remove the cross-survey fit criterion even when remaining positions support an NSC-only slope.')
    check_aggregate(report)
    save_json(output/'v17_catalogue_bias_probe_summary.json',report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frozen',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use a Slurm compute node')
    run(**vars(args))
