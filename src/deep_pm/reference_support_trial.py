"""Research-only support trial retaining full pre-cut shared reference covariance."""
import argparse
import os
from pathlib import Path
import shutil

import numpy as np

from .continuous_reference import PRECISION,bind,verify_directory
from .io import save_json,digest
from .provenance import run_stage
from .public_summary import check_aggregate


def prepare(work,original):
    old=original/'private_c0.npz';uncut=original/'private_c0_unfiltered.npz'
    with np.load(old,allow_pickle=False) as a,np.load(uncut,allow_pickle=False) as b:
        index={k:j for j,k in enumerate(b['names'])};ix=[index[k] for k in a['names']]
        for key in ('position','draws','counts','gauge'):np.testing.assert_array_equal(a[key],b[key][ix])
        np.testing.assert_array_equal(a['projection_center_deg'],b['projection_center_deg'])
        old_count=len(a['names']);all_count=len(b['names'])
    shutil.copyfile(uncut,work/'private_c0.npz')
    result=dict(frozen_group_count=old_count,pre_cut_group_count=all_count,added_high_se_groups=all_count-old_count,
        shared_offset_and_bootstrap_values_changed=False,full_position_and_motion_reference_covariance_required=True,
        original_uncut_sha256=digest(uncut),trial_sha256=digest(work/'private_c0.npz'),slurm_job=os.environ['SLURM_JOB_ID'],
        adoption='research_only_not_adopted',
        scope='No recalibration, no coefficient or draw discarded, no arbitrary recentering of a weak subgraph. High marginal group-SE measurements admitted only to this support sensitivity experiment; full correlated reference covariance remains. No independent-zero certification implied.')
    check_aggregate(result);save_json(work/'reference_support_trial_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('work',type=Path);p.add_argument('original',type=Path);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    bind(a.work,'original_receipt.json',verify_directory(a.original,('private_c0.npz','private_c0_unfiltered.npz')))
    with run_stage(a.work,PRECISION,'a','reference-support-sensitivity-v14',inputs=('original_receipt.json',),outputs=('private_c0.npz','reference_support_trial_summary.json')) as go:
        if go:prepare(**vars(a))


if __name__=='__main__':main()
