"""Re-evaluate frozen v20 scales with identical training-centre information."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import numpy as np

from .blind_catalogue_quality import read
from .io import digest,save_json
from .public_summary import check_aggregate
from .usability_quality import load_blind,condition,split_identity,vectors
from .coordinate_readout import stats


def run(registration,old_summary,output):
    plan=read(registration);manifest=Path(plan['frozen_manifest'])
    if digest(manifest)!=plan['frozen_manifest_sha256']:raise ValueError('Frozen quality scope changed')
    frozen=manifest.parent;rows=list(load_blind(frozen,frozen.parent/'blind_quality').values())
    old=read(old_summary);models=old['models'];new_windows={r['window'] for r in plan['selected']};result={};counts=defaultdict(int)
    for key,model in models.items():
        if not model['supported']:
            counts['unsupported_training']+=1;result[key]=dict(training=model['n'],supported=False);continue
        counts['trainable']+=1;center=np.asarray(model['median_zero']);scale=np.asarray(model['marginal_total_error_multiplier']);tests={}
        subset=[r for r in rows if condition(r)==key];passed=[]
        for name,predicate in [('old_validation',lambda r:r['window'] not in new_windows and split_identity(r['identity'])==3),
                               ('old_confirmation',lambda r:r['window'] not in new_windows and split_identity(r['identity'])==4),
                               ('new_window_transfer',lambda r:r['window'] in new_windows)]:
            selected=[r for r in subset if predicate(r)];motion,error=vectors(selected) if selected else (np.empty((0,2)),np.empty((0,2)))
            baseline=stats(motion-center,error);revised=stats(motion-center,error*scale)
            gate=False
            if len(selected)>=20:
                before=np.abs(np.asarray(baseline['raw_coverage_1sigma']['fraction'])-.682689492);after=np.abs(np.asarray(revised['raw_coverage_1sigma']['fraction'])-.682689492)
                before2=np.abs(np.asarray(baseline['raw_coverage_2sigma']['fraction'])-.954499736);after2=np.abs(np.asarray(revised['raw_coverage_2sigma']['fraction'])-.954499736)
                gate=bool(np.all((after<=before)&(after2<=before2+.02)&(after<=.05)&(after2<=.03)))
            tests[name]=dict(n=len(selected),same_training_center_unscaled=baseline,same_training_center_frozen_scale=revised,registered_coverage_gate=gate)
            passed.append(gate)
        if all(passed[:2]):counts['symmetric_old_validation_and_confirmation_pass']+=1
        if passed[2]:counts['symmetric_new_transfer_pass']+=1
        original=old['validation'][key]
        legacy_old=all(v.get('support_sufficient') and all(v.get('calibrated_marginal_coverage_by_axis',[False])) and all(v.get('coverage_improvement_by_axis',[False])) for name,v in original.items() if name!='new_window_transfer')
        if legacy_old:counts['legacy_old_validation_and_confirmation_pass']+=1
        result[key]=dict(training=model['n'],supported=True,frozen_training_center=center,frozen_scale=scale,tests=tests)
    report=dict(input_sha256=digest(old_summary),fixed_blind_QSO=len(rows),counts=dict(counts),models=result,slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Frozen v20 models, no new training or catalogue changes. Both baseline and scaled comparison use the same training identity centre, evaluation identities and formal errors; only frozen scale differs. Raw/centred/tail statistics separate. Original4unsupported/12trainable/5oldpass/0newtransfer interpretation retained; symmetric comparison explicitly reported. No faint Gaia or star error calibration.')
    check_aggregate(report);save_json(output,report);print('SYMMETRIC_SCALE_AUDIT',dict(counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registration','old_summary','output'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
