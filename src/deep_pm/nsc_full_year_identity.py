"""Trace full-period source classes into independent held-year associations."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path

import numpy as np

from .continuous_reference import PRECISION, bind, verify_directory
from .io import save_json
from .provenance import run_stage
from .public_summary import check_aggregate
from .validation import ratio


def sdss_members(source):
    return tuple(sorted(d for d in source['detection_ids'] if d.startswith('SDSS:')))


def summarize(full, train, held):
    train_by_key = defaultdict(list)
    full_counts = Counter(sdss_members(r) for r in full)
    for slot, source in enumerate(train):
        train_by_key[sdss_members(source)].append(slot)
    matches = defaultdict(list)
    for r in held:
        matches[r['slot']].append(r)
    result = {}
    for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
        for low in (21,22):
            population = [r for r in full if r['kind']==kind and r['magnitude'] is not None
                          and low<=r['magnitude']<low+1]
            counts = Counter()
            values = []
            for r in population:
                key = sdss_members(r)
                possible = train_by_key[key]
                if not key or full_counts[key]>1 or len(possible)>1:
                    counts['ambiguous_exact_sdss_members']+=1
                    continue
                if not possible:
                    counts['no_exact_sdss_members_in_training']+=1
                    continue
                slot = possible[0]
                shared = set(r['detection_ids']) & set(train[slot]['detection_ids'])
                if not any(k.startswith('NSC:') for k in shared):
                    counts['no_shared_training_nsc_member']+=1
                    continue
                counts['exact_sdss_plus_shared_nsc']+=1
                counts['trained_as_'+train[slot]['kind']]+=1
                if matches[slot]:
                    counts['unique_held_source_recovered']+=1
                    counts['recovered_with_full_period_member']+=any(x['held_detection_id'] in r['detection_ids'] for x in matches[slot])
                    values.extend(matches[slot])
            record=dict(full_period_sources=len(population),counts=dict(counts),
                        training_association=ratio(counts['exact_sdss_plus_shared_nsc'],len(population)),
                        held_position_recovery_given_link=ratio(counts['unique_held_source_recovered'],counts['exact_sdss_plus_shared_nsc']))
            if len(values)>=5:
                offsets = np.asarray([r['residual_mas'] for r in values])
                errors = np.asarray([r['error_mas'] for r in values])
                record.update(position_p68_mas=float(np.quantile(np.linalg.norm(offsets,axis=1),.68)),
                    within_1sigma=[ratio(int(x),len(values)) for x in (abs(offsets)<=errors).sum(axis=0)],
                    within_2sigma=[ratio(int(x),len(values)) for x in (abs(offsets)<=2*errors).sum(axis=0)])
            result[kind+'__r'+str(low)] = record
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','catalogue','year'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use Slurm')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'full_receipt.json',verify_directory(a.catalogue,('private_standard_motion.json.gz',)))
    bind(a.work,'year_receipt.json',verify_directory(a.year,('private_training.json','private_held.json','allband_blind_year_summary.json')))
    with run_stage(a.work,PRECISION,a.catalogue.name,'nsc-full-year-identity-v16',('full_receipt.json','year_receipt.json'),('full_year_identity_summary.json',)) as go:
        if go:
            with gzip.open(a.catalogue/'private_standard_motion.json.gz','rt') as stream:
                full=json.load(stream)
            year=json.loads((a.year/'allband_blind_year_summary.json').read_text())
            out=dict(status=year['status'],year=year.get('year'),slurm_job=os.environ['SLURM_JOB_ID'],
                classes=summarize(full,json.loads((a.year/'private_training.json').read_text()),json.loads((a.year/'private_held.json').read_text())),
                scope='Post hoc cross-identification by exact unique SDSS physical members plus shared NSC training members. The original year was excluded before training reconstruction. Full-period class uses all data and is a retrospective condition. The independent candidate, footprint and displaced-position denominators remain in the blind-year archive. This comparison is not an unconditional star-purity estimate.')
            check_aggregate(out)
            save_json(a.work/'full_year_identity_summary.json',out)


if __name__ == '__main__':
    main()
