"""Strict astrometric replay; only effective-group sum roundoff is admissible."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path

from .continuous_reference import PRECISION,bind,verify_directory
from .io import digest,save_json
from .provenance import run_stage
from .public_summary import check_aggregate


def compare_rows(old,new):
    if len(old)!=len(new):raise ValueError('Source count changed during metadata recovery')
    maximum=0.;changed=0
    for a,b in zip(old,new):
        if set(a)!=set(b):raise ValueError('Source schema changed during metadata recovery')
        if {k:v for k,v in a.items() if k!='information'}!={k:v for k,v in b.items() if k!='information'}:
            raise ValueError('Astrometry, covariance, membership or source metadata changed')
        x,y=a['information'],b['information']
        if {k:v for k,v in x.items() if k!='effective_information_groups'}!={k:v for k,v in y.items() if k!='effective_information_groups'}:
            raise ValueError('Non-roundoff information diagnostics changed')
        x,y=x['effective_information_groups'],y['effective_information_groups']
        if set(x)!=set(y):raise ValueError('Effective-group diagnostic schema changed')
        for family in x:
            if family not in ('physical_group','observing_night') or len(x[family])!=len(y[family]):raise ValueError('Unexpected diagnostic family')
            for p,q in zip(x[family],y[family]):
                if not math.isfinite(p+q) or not math.isclose(p,q,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Effective information changed beyond float summation roundoff')
                maximum=max(maximum,abs(p-q));changed+=int(p!=q)
    return dict(source_count=len(old),all_source_fields_and_covariances_exact=True,all_other_information_exact=True,effective_group_float_elements_changed=changed,maximum_effective_group_roundoff=maximum)


def run(work,old,new):
    def read(folder,name):
        opener=gzip.open if name.endswith('.gz') else open
        with opener(folder/name,'rt') as stream:return json.load(stream)
    result=compare_rows(read(old,'private_standard_motion.json.gz'),read(new,'private_standard_motion.json.gz'))
    for key,name in [('r_control_exact','private_r_control.json.gz'),('known_qso_exact','private_known_qso_blocks.json')]:
        result[key]=read(old,name)==read(new,name)
        if not result[key]:raise ValueError('Frozen r or QSO control changed')
    manifest=json.loads((old/'run_manifest.json').read_text());attempt=manifest['stages']['nsc-science-catalogue-v15'][-1]
    for name,sha in attempt['inputs'].items():
        if digest(old/name)!=sha:raise ValueError('Original failed-stage input changed')
    result.update(old_failed_receipt_sha256=digest(old/'run_manifest.json'),new_completed_receipt_sha256=digest(new/'run_manifest.json'),slurm_job=os.environ['SLURM_JOB_ID'],
        scope='Every physical member, source order, position, proper motion, all full/independent/reference covariances, robust weights and quality metadata are exactly equal as parsed JSON; r and known-QSO controls exactly equal. Only two effective-information-group arrays permit relative/absolute 1e-12 summation roundoff. Original failed outputs are newly hashed as audit inputs, never retroactively certified as completed. Original failure and strict byte-check failure remain preserved.')
    check_aggregate(result);save_json(work/'science_replay_summary.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('work','old','new'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True);bind(a.work,'old_failed_receipt.json',a.old/'run_manifest.json');bind(a.work,'new_receipt.json',verify_directory(a.new,('private_standard_motion.json.gz','private_r_control.json.gz','private_known_qso_blocks.json','nsc_science_catalogue_summary.json')))
    inputs=['old_failed_receipt.json','new_receipt.json']
    for name in ('private_standard_motion.json.gz','private_r_control.json.gz','private_known_qso_blocks.json'):
        alias='old_'+name;bind(a.work,alias,a.old/name);inputs.append(alias)
    with run_stage(a.work,PRECISION,'strict_science_replay','science-replay-v15',inputs=tuple(inputs),outputs=('science_replay_summary.json',)) as go:
        if go:run(a.work,a.old,a.new)


if __name__=='__main__':main()
