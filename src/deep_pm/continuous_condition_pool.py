"""Verify frozen C0 reproduction and pool only de-identified QSO statistics."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .continuous_conditions import METHODS, summaries
from .continuous_summary import estimate
from .io import digest, save_json
from .public_summary import check_aggregate


def reproduce(records, frozen):
    current = [r for r in records if 21 <= r['magnitude'] < 23]
    if len(current) != len(frozen):
        raise ValueError('Frozen C0 sample count changed')
    difference = 0.
    for new, old in zip(current, frozen):
        if new['magnitude'] != old['magnitude'] or new['exposures'] != old['exposures']:
            raise ValueError('Frozen C0 object or detection support changed')
        delta = np.max(np.abs(np.asarray(new['motion']['all_SDSS'])-old['motion']['C0']))
        difference = max(difference, float(delta))
    if difference > 1e-7:
        raise ValueError('Frozen C0 proper motion failed reproduction')
    return dict(n=len(current), maximum_absolute_motion_difference_masyr=difference)


def pool(frozen, inputs, output):
    plan = [r.split('\t')[0] for r in (frozen/'run-plan.tsv').read_text().splitlines()]
    records, receipts, missing = [], {}, []
    for name in plan:
        work = inputs/name
        if not (work/'conditions_summary.json').is_file():
            missing.append(name)
            continue
        manifest = json.loads((work/'run_manifest.json').read_text())
        stage = manifest['stages']['continuous-conditions-v13'][-1]
        if stage['state'] != 'completed':
            missing.append(name)
            continue
        for filename, checksum in stage['outputs'].items():
            if digest(work/filename) != checksum:
                raise ValueError('Conditions output changed')
        current = json.loads((work/'private_conditions.json').read_text())
        old = json.loads((frozen/'outputs'/name/'private_qso_blocks.json').read_text())
        verified = reproduce(current, old)
        receipts[name] = dict(verified, commit=manifest['commit'], slurm_job=stage['slurm_job'],
                              summary_sha256=digest(work/'conditions_summary.json'))
        records.extend(current)
    stats = summaries(records, 9137)
    shared = {}
    for scan in ('all','ordinary','transition','repeat'):
        for low in (20,21,22):
            selected = [r for r in records if low <= r['magnitude'] < low+1 and r['published'] is not None
                        and r['legacy'] is not None and (scan == 'all' or
                        scan == 'ordinary' and abs(r['latitude_deg']) >= 3 or
                        scan == 'transition' and 1 < abs(r['latitude_deg']) < 3 or
                        scan == 'repeat' and abs(r['latitude_deg']) <= 1)]
            converted = [dict(magnitude=r['magnitude'],exposures=r['exposures'],nsc_epoch_days=r['nsc_epoch_days'],
                             motion=dict(r['motion'],C0=r['motion']['all_SDSS'],
                                         NSC=r['published'][0],Xiao=r['legacy'][0],all_raw=r['raw_motion']))
                         for r in selected]
            shared[f'{scan}__r{low}_{low+1}'] = estimate(converted,
                ('NSC','Xiao','C0','NSC_only','one_SDSS','all_raw'),9137)
    result = dict(frozen_baseline='13d4e29',completed_cores=len(receipts),expected_cores=len(plan),
                  missing_cores=missing,eligible_count=len(records),reproduction=receipts,
                  cohorts=stats,physical_group_sensitivity=shared,
                  input_scope='Stationary known-position spectroscopic QSO; source detections remain on NERSC.')
    check_aggregate(result)
    save_json(output,result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('frozen','inputs','output'):
        p.add_argument(name,type=Path)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use a Slurm compute node')
    pool(a.frozen,a.inputs,a.output)


if __name__ == '__main__':
    main()
