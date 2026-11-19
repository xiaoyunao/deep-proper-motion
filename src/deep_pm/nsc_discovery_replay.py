"""Trace frozen whole-year identities after completing the NSC retrieval universe."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from .continuous_reference import PRECISION,bind,verify_directory
from .io import save_json,digest
from .provenance import run_stage
from .public_summary import check_aggregate


def compare(old,new):
    def sdss_key(row):return tuple(sorted(i for i in row['detection_ids'] if i.startswith('SDSS:')))
    groups={};counts=Counter(sdss_key(r) for r in new)
    for row in new:
        k=sdss_key(row)
        if k and counts[k]==1:groups[k]=row
    out={}
    for kind in ('r_existing','extra_r_anchor','non_r_anchor'):
        for low in (20,21,22):
            part=[r for r in old if r['kind']==kind and low<=r['magnitude']<low+1];status=Counter();motion=[];error=[];changed_class=0
            for row in part:
                candidate=groups.get(sdss_key(row))
                if candidate is None:status['no_unique_same_sdss_member_identity']+=1;continue
                changed_class+=int(candidate['kind']!=row['kind'])
                if set(candidate['detection_ids'])!=set(row['detection_ids']):status['same_sdss_changed_nsc_members']+=1;continue
                status['identical_physical_members']+=1;motion.append(np.asarray(candidate['pm_icrs_masyr'])-row['pm_icrs_masyr']);error.append(np.asarray(candidate['pm_error_masyr'])-row['pm_error_masyr'])
            out[kind+'__r'+str(low)]=dict(frozen_sources=len(part),identity_status=dict(status),class_changed_at_same_sdss_identity=changed_class,
                maximum_absolute_pm_change_masyr=float(np.max(abs(np.asarray(motion)))) if motion else None,
                maximum_absolute_error_change_masyr=float(np.max(abs(np.asarray(error)))) if error else None)
    return dict(frozen_sources=len(old),new_sources=len(new),by_frozen_class=out,scope='All frozen old training objects retained in denominators. Unique exact SDSS physical-member identities and full detection-member sets, never PM proximity, define replay. Changed NSC sets are separate; same-member PM/error differences diagnose implementation/reference changes, not scientific gains or truth.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','old','new'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    for label,folder in [('old',a.old),('new',a.new)]:bind(a.work,label+'_receipt.json',verify_directory(folder,('private_training.json','allband_blind_year_summary.json')))
    with run_stage(a.work,PRECISION,'frozen_members','nsc-discovery-replay-v15',inputs=('old_receipt.json','new_receipt.json'),outputs=('nsc_discovery_replay_summary.json',)) as go:
        if go:
            oldmeta=json.loads((a.old/'allband_blind_year_summary.json').read_text());newmeta=json.loads((a.new/'allband_blind_year_summary.json').read_text())
            if oldmeta['year']!=newmeta['year']:raise ValueError('Held year changed')
            result=compare(json.loads((a.old/'private_training.json').read_text()),json.loads((a.new/'private_training.json').read_text()));result['year']=oldmeta['year'];result['slurm_job']=os.environ['SLURM_JOB_ID']
            graphs=[digest((folder/'reference_receipt.json').resolve().parent/'private_c0.npz') for folder in (a.old,a.new)]
            result.update(old_reference_sha256=graphs[0],new_reference_sha256=graphs[1],same_reference_graph=graphs[0]==graphs[1]);check_aggregate(result);save_json(a.work/'nsc_discovery_replay_summary.json',result)


if __name__=='__main__':main()
