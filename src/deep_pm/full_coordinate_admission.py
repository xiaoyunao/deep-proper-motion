"""Numerical and fixed-control admission of the first complete component."""
import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np

from .blind_catalogue_quality import read
from .io import digest,save_json
from .nsc_comparison import comparison
from .public_summary import check_aggregate


def admission(root):
    first=read(root/'graph/private_first_complete_component.json')['selected'];number=first['component']
    folder=root/'reference'/f'component-{number:04d}'
    mathematical=read(folder/'v22_reference_component_summary.json')
    hold=read(root/'admission/held-galaxies/v22_complete_reference_validation_summary.json')
    assert mathematical['convergence_history']['irls'][-1]['iteration']==6
    assert len(mathematical['convergence_history']['draws'])==48
    assert max(d['relative_normal_residual'] for d in mathematical['convergence_history']['draws'])<1e-8
    with np.load(folder/'private_c0_unfiltered.npz') as a:
        nsc=np.char.startswith(a['names'],'NSC:')
        np.testing.assert_allclose(np.median(a['position'][nsc],axis=0),0,atol=1e-8)
        np.testing.assert_allclose(np.median(a['draws'][nsc],axis=0),0,atol=1e-8)
        assert np.isfinite(a['position']).all() and np.isfinite(a['draws']).all()
    meta=read(root/'admission/private_metadata_source_plan.json');fixed=[];checks=[]
    for e in meta['selected']:
        core=e['entry']['core'];dest=root/'admission/source/sources'/core
        report=read(dest/'v22_source_core_summary.json')
        science=read(dest/'private_standard_motion.json.gz')
        for r in science:
            cov=np.array(r['covariance']);np.testing.assert_allclose(cov,cov.T,atol=1e-7)
            if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min()<=0:raise ValueError('Nonpositive full state covariance')
            np.testing.assert_allclose(r['pm_error_masyr'],np.sqrt(np.diag(cov)[[1,3]]),rtol=1e-10)
            assert r['reference_epoch']==2015.5 and len(set(r['detection_ids']))==len(r['detection_ids'])
        rows=read(dest/'private_fixed_blind_qso.json.gz');fixed.extend(rows)
        checks.append(dict(core=core,chart_angle_deg=e['angle_deg'],science_solutions=len(science),fixed_QSO_denominator=len(rows),
            fixed_QSO_status=report['fixed_qso_status'],elapsed_seconds=report['elapsed_seconds'],peak_rss_mib=report['peak_rss_mib'],
            source_receipt_sha256=digest(dest/'v22_source_core_summary.json')))
    settings=dict(min_publish_count=5,tail_masyr=5,bootstrap_draws=256,bootstrap_absolute_zero=True,seed=222)
    samples={}
    for low in (20,21,22):
        selected=[r for r in fixed if r['magnitude'] is not None and low<=r['magnitude']<low+1]
        computed=[r for r in selected if r['status']=='computed' and 'blind_all' in r['old_values']]
        pairs=[dict(old=r['old_values']['blind_all'],new=r['new_values'],B1=r['B1_values']) for r in computed]
        samples[str(low)]=dict(frozen_denominator=len(selected),computed_same_members=len(computed),
            new_local_support_exit=sum(not r['local_support_pass'] for r in computed),new_formal_error_exit=sum(not r['formal_error_pass'] for r in computed),
            new_residual_exit=sum(not r['residual_pass'] for r in computed),
            comparison=comparison(pairs,settings,(('new','old'),('B1','old'))))
    remaining=sum(c['measurements'] for c in read(root/'graph/v22_global_graph_summary.json')['components'] if c['component']!=number)
    result=dict(component=number,numerical_admission_passed=True,reference_mathematics_sha256=digest(folder/'v22_reference_component_summary.json'),
        held_reference_sha256=digest(root/'admission/held-galaxies/v22_complete_reference_validation_summary.json'),
        held_same_source_n=hold['strict_same_physical_detections_and_objects']['n'],
        held_position_p68_new_over_old=hold['strict_same_physical_detections_and_objects']['radial_p68_new_over_old'],
        real_source_checks=checks,fixed_QSO_samples=samples,fixed_status=dict(Counter(r['status'] for r in fixed)),
        reference_remaining_seconds_linear_times3=float(3*mathematical['elapsed_seconds']*remaining/first['measurements']),
        source_cost_scope='Three preselected low-input-size cores are source correctness pilots, not an unbiased all-core resource forecast. Final memory scheduling also uses the maximum measured 25GiB input worker.',
        decision='Continue remaining complete components and all registered cores under the frozen 48 billing-node-hour ceiling. QSO shifts are scientific output, not a zero-point retuning gate.',
        slurm_job=os.environ['SLURM_JOB_ID'],scope='First COMPLETE reference component, including object-heldout reference sample and fixed old-blind QSO before new quality cuts. Source pilots retain origin-footprint denominator, not final global catalogue. No science catalogue outside registered coverage adopted here.')
    check_aggregate(result);save_json(root/'admission/v22_complete_component_admission_summary.json',result)
    print('COMPLETE_COMPONENT_ADMISSION_PASSED',number,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm')
    admission(a.root)
