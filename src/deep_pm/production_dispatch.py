"""Submit the dependency-linked registered production stages within one budget."""
import argparse
import os
from pathlib import Path
import subprocess

from .blind_catalogue_quality import read
from .io import digest, save_json


def run(work, code, account, qos, python, runtime, window):
    proof_path = work/'reference/v19_reference_plan_summary.json'
    pol = read(proof_path)
    plan_hash = digest(work/'reference/private_continuation_reference_plan.json')
    if pol['plan_sha256']!=plan_hash or pol['forecast_cpu_node_hours']>pol['budget_cpu_node_hours']:
        raise ValueError('Changed or over-budget reference plan')
    stages = pol['physical_partitions']
    if not stages:
        raise ValueError('Missing registered reference tasks')
    stamp=work/'private_production_jobs.json'
    expected=dict(plan_sha256=plan_hash, proof_sha256=digest(proof_path))
    state=read(stamp) if stamp.is_file() else dict(**expected,jobs={})
    if any(state[k]!=v for k,v in expected.items()):
        raise ValueError('Existing Slurm job ledger belongs to a different scope')
    jobs=state['jobs']

    def submit(alias, stage, nodes, cpus, memory, minutes, extra=(), dependencies=()):
        if alias in jobs:
            return jobs[alias]['job']
        limits = f'{minutes//60:02d}:{minutes%60:02d}:00'
        command=['sbatch','--parsable','-A',account,'-q',qos,'-C','cpu',
                 f'--nodes={nodes}','--ntasks-per-node=1',f'--cpus-per-task={cpus}',
                 f'--mem={memory}',f'--time={limits}',f'--job-name=dpm-v19-{alias}',
                 f'--output={work}/logs/{alias}-%j.out',f'--chdir={code}']
        if dependencies:
            command.extend(['--kill-on-invalid-dep=yes','--dependency=afterok:'+':'.join(dependencies)])
        command.extend([code/'scripts/wide-v19-production.sh',stage,work,code,account,qos,python,runtime,*extra])
        number=subprocess.check_output(list(map(str,command)),text=True).strip().split(';')[0]
        if not number.isdigit():
            raise RuntimeError('Unexpected Slurm job receipt')
        jobs[alias]=dict(job=number,dependencies=list(dependencies),nodes=nodes,requested_wall_minutes=minutes)
        save_json(stamp,state)
        print('SUBMITTED',alias,number,flush=True)
        return number

    # 12 intake + up to 12 reference + 20 fit + 3 final closure.
    reference_budget=720
    min_minutes=min(45,reference_budget//len(stages))
    sizes={k:v['new_cores']+v['replayed_cores'] for k,v in stages.items()}
    ref_jobs=[submit(label,'reference',1,48,'0',
               min_minutes+(reference_budget-min_minutes*len(stages))*n//sum(sizes.values()),[label])
              for label,n in sizes.items()]
    fit=submit('fit','fit',4,96,'320G',300,dependencies=ref_jobs)
    if 'science' in jobs:
        print('RETAIN_LEGACY_SCIENCE_JOB',jobs['science']['job'],flush=True)
        return
    # Same three node-hour closure ceiling; no reserved second node during
    # serial accumulation or the independently parallel blind quality readout.
    partitions=submit('science-partitions','science-partitions',2,8,'180G',60,[window],dependencies=[fit])
    merge=submit('science-merge','science-merge',1,8,'96G',30,[window],dependencies=[partitions])
    submit('science-quality','science-quality',1,16,'96G',30,[window],dependencies=[merge])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('work','code'):
        parser.add_argument(name,type=Path)
    for name in ('account','qos','python','runtime','window'):
        parser.add_argument(name)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use the successful intake allocation')
    run(**vars(args))
