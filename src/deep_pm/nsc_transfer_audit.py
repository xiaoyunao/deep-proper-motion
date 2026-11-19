"""Verify frozen catalogue members and audit blind-QSO selection without refitting."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path

import numpy as np

from .__main__ import tomllib
from .continuous_reference import bind, verify_directory
from .exposure_catalogues import inside
from .io import digest, save_json
from .nsc_comparison import comparison
from .nsc_ownership_overlay import CONTROL, STANDARD, record_hash
from .nsc_science_summary import SETTINGS
from .provenance import run_stage
from .public_summary import check_aggregate


def read_rows(folder, name):
    with gzip.open(folder/name, 'rt') as stream:
        return json.load(stream)


def control_match(members, owners, control):
    votes = Counter(owners[item] for item in members if item in owners)
    if not votes:
        return None, 0, False
    index, overlap = votes.most_common(1)[0]
    return index, overlap, len(votes) > 1


def inspect(state, geometry):
    expected = json.loads((state/'continuous_science_summary.json').read_text())
    paths = json.loads((state/'fits.json').read_text())
    overlay = json.loads((state/'private_owner_overlay.json').read_text())
    specs = tomllib.loads(geometry.read_text())['fields']
    if digest(state/'fits.json') != overlay['fits_manifest_sha256']:
        raise ValueError('Frozen source index changed')
    moves = {(m['origin_core'],m['filename'],m['source_slot']):m for m in overlay['transfers']}
    classes = Counter()
    owned = Counter()
    seen_moves = set()
    unmatched = []
    blind = []
    receipt_hashes = {}
    counts = Counter()
    comparison_members = Counter()
    for core, path in paths.items():
        print(core, flush=True)
        folder = Path(path)
        files = (STANDARD, CONTROL, 'private_known_qso_blocks.json', 'private_blind_qso.json', 'nsc_science_catalogue_summary.json')
        receipt = verify_directory(folder, files)
        if digest(receipt) != expected['fit_receipt_sha256'][core]:
            raise ValueError('Frozen fit receipt changed')
        receipt_hashes[core] = digest(receipt)
        ref = (folder/'reference_receipt.json').resolve().parent
        if digest(ref/'private_c0.npz') != expected['reference_graph_sha256']:
            raise ValueError('Frozen reference graph changed')
        full, control = [read_rows(folder, name) for name in (STANDARD, CONTROL)]
        for name, data in ((STANDARD,full),(CONTROL,control)):
            for slot, row in enumerate(data):
                key = core, name, slot
                move = moves.get(key)
                owner = move['owner_core'] if move else core
                if move:
                    seen_moves.add(key)
                    if record_hash(row) != move['unchanged_record_sha256']:
                        raise ValueError('Transferred state changed')
                if not inside(*row['position_deg'], specs[owner]):
                    raise ValueError('Wrong common-epoch owner')
                owned[owner,name] += 1
            counts[name] += len(data)
            cov = np.asarray([row['covariance'] for row in data]).reshape(-1,4,4)
            errs = np.asarray([row['pm_error_masyr'] for row in data]).reshape(-1,2)
            if not np.isfinite(cov).all():
                raise ValueError('Nonfinite covariance')
            np.testing.assert_allclose(cov, cov.transpose(0,2,1), atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(np.sqrt(cov[:,[1,3],[1,3]]),errs, rtol=1e-9)
        all_ids = [item for row in full for item in row['detection_ids']]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError('Repeated physical measurement within one science shard')
        for r in full:
            if r['reference_model_sha256'] != expected['reference_graph_sha256']:
                raise ValueError('Source graph hash changed')
            if len({len(r[k]) for k in ('detection_ids','physical_groups','exposures','nights_utc18','robust_weight')}) != 1:
                raise ValueError('Broken measurement metadata alignment')
        owners = {item:i for i,r in enumerate(control) for item in r['detection_ids']}
        claims = Counter()
        touches = defaultdict(list)
        for i,r in enumerate(full):
            classes[r['kind']] += 1
            member = r['detection_ids']
            for j, overlap in Counter(owners[item] for item in member if item in owners).items():
                touches[j].append((i,overlap))
            winner, overlap, contested = control_match(member, owners, control)
            if r['kind'] == 'r_existing':
                if winner is None or overlap < .6*len(control[winner]['detection_ids']):
                    raise ValueError('Published r_existing rule cannot be replayed')
                claims[winner] += 1
                comparison_members['contested_r_existing' if contested else 'unique_r_existing'] += 1
        comparison_members['multiple_full_claimants_of_old_identity'] += sum(v>1 for v in claims.values())
        for j,r in enumerate(control):
            if j not in claims:
                unmatched.append(dict(core=core, slot=j, detection_ids=r['detection_ids'],
                    position_deg=r['position_deg'], touched_full_slots=touches[j]))
        for q in json.loads((folder/'private_blind_qso.json').read_text()):
            row = full[q['source_slot']]
            if row['kind'] != q['kind']:
                raise ValueError('Blind QSO source class changed')
            np.testing.assert_array_equal(row['pm_icrs_masyr'],q['values']['blind_all'][0])
            blind.append(dict(q, origin_core=core, record=row))
    if classes != expected['all_magnitude_source_classes'] or seen_moves != moves.keys():
        raise ValueError('Frozen counts changed')
    for core in paths:
        assert owned[core,STANDARD] == expected['spatial'][core]['owned_all_nsc']
        assert owned[core,CONTROL] == expected['spatial'][core]['owned_r_control']
    assert counts[STANDARD] == 481945 and len(paths) == 91
    # Distinguish logical ownership from a different input origin that shares members.
    unmatched_ids = {item for r in unmatched for item in r['detection_ids']}
    by_member = defaultdict(list)
    for core,path in paths.items():
        for i,row in enumerate(read_rows(Path(path),STANDARD)):
            for item in unmatched_ids.intersection(row['detection_ids']):
                by_member[item].append((core,i))
    overlap_classes = Counter()
    for r in unmatched:
        hits = Counter(hit for item in r['detection_ids'] for hit in by_member[item])
        r['member_overlap'] = [dict(origin_core=c,slot=i,count=n) for (c,i),n in hits.items()]
        if not hits:
            key = 'no_shared_member'
        elif len(hits)>1:
            key = 'split_among_multiple_full_detections'
        elif next(iter(hits))[0]!=r['core']:
            key = 'shared_in_another_input_core'
        else:
            key = 'shared_in_one_same_core_full_fit'
        overlap_classes[key] += 1
    cohort = {}
    for kind in classes:
        for low in (21,22):
            sample = [q for q in blind if q['kind']==kind and q['magnitude'] is not None and low<=q['magnitude']<low+1]
            key = f'{kind}__r{low}'
            control = expected['blind_spectral'][key]['all_spectral']
            metrics = comparison([{'blind_all':q['values']['blind_all']} for q in sample], SETTINGS)
            assert metrics['n'] == control['n']
            if 'methods' in control:
                for name in ('median','radial_abs68','nmad','fraction_within_1sigma','fraction_within_2sigma'):
                    np.testing.assert_allclose(metrics['methods']['blind_all'][name],control['methods']['blind_all'][name],atol=0,rtol=0)
            cohort[key] = metrics
    result = dict(cores=len(paths),source_counts=dict(counts),classes=dict(classes),
                  owner_transfers_verified=len(seen_moves),reference_sha256=expected['reference_graph_sha256'],
                  fit_receipt_sha256=receipt_hashes,distinct_old_controls_without_r_existing=len(unmatched),
                  old_control_overlap_categories=dict(overlap_classes),identity_audit=dict(comparison_members),
                  frozen_blind_qso=cohort,slurm_job=os.environ['SLURM_JOB_ID'],
                  scope='All immutable source files and both QSO block hashes verified. Old controls are counted by unique physical-member claims. Overlap categories alone do not prove the reason for quality exits.')
    check_aggregate(result)
    return result, blind, unmatched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work','state','geometry'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use Slurm')
    args.work.mkdir(parents=True,exist_ok=True)
    receipt = verify_directory(args.state, ('fits.json','private_owner_overlay.json','continuous_science_summary.json'))
    bind(args.work,'frozen_receipt.json',receipt)
    outputs = ('transfer_audit_summary.json','private_blind_audit.json','private_unmatched_controls.json')
    with run_stage(args.work,args.geometry,'C','nsc-transfer-audit-v16',('frozen_receipt.json',),outputs) as go:
        if go:
            result, blind, unmatched = inspect(args.state,args.geometry)
            for name, payload in zip(outputs,(result,blind,unmatched)):
                save_json(args.work/name,payload)


if __name__ == '__main__':
    main()
