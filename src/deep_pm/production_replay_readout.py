"""Compare old, replayed, and new blind QSO cohorts from signed readout caches."""

import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np

from .blind_catalogue_quality import merge_strata, read
from .io import digest, save_json
from .public_summary import check_aggregate


def load_blind(science, blind):
    manifest = science/'private_continuation_manifest.json'
    manifest_sha = digest(manifest)
    proof = read(science/'continuation_science_receipt.json')
    if proof['combined_science_sha256'] != digest(science/'continuation_production_science_summary.json'):
        raise ValueError('Combined science summary digest changed')
    status = read(blind/'v17_blind_catalogue_quality_summary.json')
    if status['manifest_sha256'] != manifest_sha:
        raise ValueError('Blind cohort and final catalogue refer to different scopes')
    parts = read(manifest)['partitions']
    rows = []
    for part in parts:
        owned = Path(part['owned'])
        expected_chunks = (part['cores'] + 63)//64
        for i in range(expected_chunks):
            chunk = read(blind/f"private_{part['label']}_{i:03}.json")
            ticket = dict(manifest_sha256=manifest_sha, receipt_sha256=digest(owned/'run_manifest.json'),
                          implementation_sha256=status['implementation_sha256'])
            if chunk['ticket'] != ticket:
                raise ValueError('A blind checkpoint belongs to different signed catalogue inputs')
            rows.extend(chunk['spectra'])
    identities = Counter(tuple(row['identity']) for row in rows)
    return {tuple(row['identity']):row for row in rows if identities[tuple(row['identity'])]==1}


def shape_class(row):
    value = row['shape']
    if value is None or not np.isfinite(value) or not 0 <= value <= 1:
        return 'unknown_r_shape'
    return 'pointlike_r' if value >= .8 else 'below_r_pointlike_threshold'


def summarize(rows):
    strata = {}
    for low in (21,22):
        view = [row for row in rows if low <= row['magnitude'] < low+1]
        subsamples = {
            'all': view,
            'single_sdss_run': [r for r in view if r['sdss_runs']==1],
            'multiple_sdss_runs': [r for r in view if r['sdss_runs']>1],
            'short_nsc_span': [r for r in view if r['nsc_span']<4],
            'long_nsc_span': [r for r in view if r['nsc_span']>=4],
        }
        for row in view:
            subsamples.setdefault(shape_class(row), []).append(row)
        strata[str(low)] = {key:dict(blind=merge_strata(part,('blind_all',)),
                                    strict_published=merge_strata(part,('blind_all','Xiao','NSC')))
                            for key,part in subsamples.items()}
    return strata


def run(previous, previous_blind, batch, output):
    current = batch/'science'
    plan = read(batch/'reference/private_continuation_reference_plan.json')
    trace = read(batch/'reference/v19_reference_plan_summary.json')
    if trace['plan_sha256'] != digest(batch/'reference/private_continuation_reference_plan.json'):
        raise ValueError('Affected reference graph manifest changed')
    # Final owner keys determine all cohorts, including sources moved in halo overlap.
    affected = {core for part in plan['solve'].values() for core in part['region_by_core']}
    new = {line.split()[0] for line in (batch/'intake/tasks.tsv').read_text().splitlines()}
    replayed = affected & plan['archives'].keys()
    if affected != replayed|new:
        raise ValueError('Some affected cores have no recorded intake or predecessor')
    old = load_blind(previous,previous_blind)
    after = load_blind(current,batch/'blind_quality')
    before_replay = {k:r for k,r in old.items() if r['core'] in replayed}
    after_replay = {k:r for k,r in after.items() if r['core'] in replayed}
    paired = set(before_replay)&set(after_replay)
    before_rows = [before_replay[k] for k in sorted(paired)]
    after_rows = [after_replay[k] for k in sorted(paired)]
    scope_rows = dict(new=[r for r in after.values() if r['core'] in new],
                      before_replay=list(before_replay.values()), after_replay=list(after_replay.values()),
                      same_qso_before=before_rows,same_qso_after=after_rows,cumulative=list(after.values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(scope='Blind optical identities after final ownership. Same QSO before/after requires unique spectral identity in both signed cohorts. R morphology is the saved legacy r shape statistic and may be unavailable.',
                  source_catalogue_sha256=digest(current/'private_continuation_manifest.json'),
                  previous_catalogue_sha256=digest(previous/'private_continuation_manifest.json'),
                  replayed_core_count=len(replayed),new_core_count=len(new),
                  shared_replayed_spectra=len(paired),
                  exited_replayed_spectra=len(before_replay.keys()-after_replay.keys()),
                  entered_replayed_spectra=len(after_replay.keys()-before_replay.keys()),
                  samples={k:summarize(rows) for k,rows in scope_rows.items()},
                  slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(result)
    save_json(output,result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous','previous_blind','batch','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use a Slurm compute node')
    run(**vars(args))
