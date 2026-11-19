"""Read coverage on the same QSO, separating observed scatter and formal errors."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from .continuous_reference import PRECISION, bind, verify_directory
from .independent_summary import standard_block
from .io import digest, save_json
from .nsc_science_summary import plan_center
from .provenance import run_stage
from .public_summary import check_aggregate
from .validation import ratio


def coverage(rows):
    out = dict(n=len(rows))
    if len(rows)<5:
        return out
    for method in ('r_physical_quality','all_nsc'):
        pm,error = np.asarray([r[method] for r in rows]).transpose(1,0,2)
        center = np.median(pm,axis=0)
        norm = pm / error
        out[method] = dict(median_formal_error_masyr=np.median(error,axis=0),
            median_pm_masyr=center, nmad_masyr=1.4826*np.median(abs(pm-center),axis=0),
            nmad_normalized=1.4826*np.median(abs(norm-np.median(norm,axis=0)),axis=0),
            p68_masyr=float(np.quantile(np.linalg.norm(pm,axis=1),.68)),
            within_1sigma=[ratio(int(v),len(rows)) for v in (abs(norm)<=1).sum(axis=0)],
            within_2sigma=[ratio(int(v),len(rows)) for v in (abs(norm)<=2).sum(axis=0)])
    out['nsc_nights_p16_p50_p84'] = np.quantile([r['nights'] for r in rows],[.16,.5,.84])
    out['joint_span_p16_p50_p84'] = np.quantile([r['span'] for r in rows],[.16,.5,.84])
    out['physical_sdss_runs'] = len({k for r in rows for k in r['sdss']})
    out['contributing_cores'] = len({r['core'] for r in rows})
    return out


def run(work, state, baseline):
    fits=json.loads((state/'fits.json').read_text())
    expected=json.loads(baseline.read_text())
    groups=defaultdict(list)
    for core,folder in fits.items():
        folder=Path(folder)
        receipt=verify_directory(folder,('private_known_qso_blocks.json',))
        if digest(receipt)!=expected['fit_receipt_sha256'][core]:
            raise ValueError('Changed current QSO file provenance')
        data=json.loads((folder/'private_known_qso_blocks.json').read_text())
        center=plan_center(folder)
        table={name:{r['validation_local_index']:r for r in rows} for name,rows in data.items()}
        for k in sorted(set.intersection(*(set(v) for v in table.values()))):
            r=table['r_legacy_quality'][k]
            if not {'Xiao','NSC'}<=r['motion'].keys():
                continue
            full=table['all_nsc'][k]
            visits=[e for e in full['exposures'] if e.startswith('SDSS:')]
            nights=len(full['nsc_nights_utc18'])
            if not 21<=r['magnitude']<23 or not visits:
                continue
            values=dict(core=core,sdss=visits,nights=nights,span=full['joint_span_years'])
            for name in ('r_physical_quality','all_nsc'):
                q=standard_block(table[name][k],center)
                values[name]=q['motion']['C0'],q['error_icrs']['C0']
            key='r'+str(int(r['magnitude']))
            runs='one_run' if len(visits)==1 else 'multiple_runs'
            groups[key+'__'+runs].append(values)
            coast='short_joint_lt12' if values['span']<12 else 'long_joint_ge12'
            groups['__'.join([key,runs,coast])].append(values)
            season='nsc_nights_le5' if nights<=5 else 'nsc_nights_gt5'
            groups['__'.join([key,runs,season])].append(values)
    result=dict(cohorts={k:coverage(v) for k,v in sorted(groups.items())},slurm_job=os.environ['SLURM_JOB_ID'],
                base_summary_sha256=digest(baseline),
                scope='Current spectral common objects, same data quality masks as frozen main summary. Distinguish formal error size and observed normalized scatter on identical source selections. Joint-span and actual-night partitions expose overlap between run strata. Wilson intervals are conditional on the object sample. Different physical exposures, calibration and sky distributions preclude causal attribution from these strata alone.')
    check_aggregate(result)
    save_json(work/'coverage_conditions_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','state','baseline'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use Slurm')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'fits.json',a.state/'fits.json')
    bind(a.work,'science_summary.json',a.baseline)
    with run_stage(a.work,PRECISION,'frozen_qso','nsc-coverage-conditions-v16',('fits.json','science_summary.json'),('coverage_conditions_summary.json',)) as go:
        if go:run(**vars(a))


if __name__=='__main__':
    main()
