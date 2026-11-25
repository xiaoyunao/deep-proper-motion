"""Continue signed scientific readouts without repeating completed wide statistics."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

from deep_pm.blind_catalogue_quality import read
from deep_pm.continuous_reference import verify_directory
from deep_pm.io import digest, save_json


def existing_readout(manifest, summary, index):
    if not summary.is_file() or not index.is_file():
        return False
    report = read(summary)
    with gzip.open(index,'rt') as stream:
        header = stream.read(4096)
    field = re.search(r'"partition_manifest_sha256"\s*:\s*"([a-f0-9]{64})"',header)
    expected = digest(manifest)
    if field is None or field.group(1)!=expected or digest(index)!=report['healpix_index_sha256']:
        raise ValueError('The completed wide readout belongs to different signed inputs')
    cores = {c for part in read(manifest)['partitions'] for c in part['region_by_core']}
    if cores!=set(report['cores']):
        raise ValueError('The completed wide readout has a different set of cores')
    return True


def complete_science(work):
    folder = work/'science'
    path = folder/'continuation_science_receipt.json'
    if not path.is_file():
        return False
    proof = read(path)
    files = {'combined_science_sha256':folder/'continuation_production_science_summary.json',
             'new_spectral_sha256':folder/'continuation_new_spectral_summary.json',
             'fit_summary_sha256':work/'fit/next_fit_summary.json',
             'reference_plan_sha256':work/'reference/private_continuation_reference_plan.json'}
    if any(proof[key]!=digest(file) for key,file in files.items()):
        raise ValueError('Scientific completion receipt no longer verifies')
    manifest = folder/'private_continuation_manifest.json'
    for part in read(manifest)['partitions']:
        owned = Path(part['owned'])
        verify_directory(owned,('continuous_science_summary.json','private_owner_overlay.json','fits.json'))
        if digest(owned/'continuous_science_summary.json')!=proof['graphs'][part['label']]['science_receipt_sha256']:
            raise ValueError('The completed scientific partition changed')
    if not existing_readout(manifest,files['combined_science_sha256'],folder/'private_nside128_index.json.gz'):
        raise ValueError('Completed scientific receipt lacks its verified wide index')
    return True


def run(work, code, helper, window, workers):
    cache = work/'resumable-statistics'
    cache.mkdir(parents=True,exist_ok=True)
    original_run = subprocess.run
    decisions = []

    def resume_command(command, *args, **kwargs):
        if isinstance(command,list) and 'deep_pm.nsc_wide_readout' in command:
            module = command.index('deep_pm.nsc_wide_readout')
            manifest,summary,index = map(Path,command[module+1:])
            if existing_readout(manifest,summary,index):
                decisions.append('reuse_completed_wide_summary')
                print('REUSE_VERIFIED_WIDE_STATISTICS',digest(summary),flush=True)
                return subprocess.CompletedProcess(command,0)
            command = command[:module-1]+[str(helper)]+command[module+1:]+[str(cache),'--workers',str(workers)]
            command = ['--mem=400G' if item.startswith('--mem=') else '-c12' if item=='-c4' else item for item in command]
            decisions.append('parallel_checkpointed_frozen_statistics')
        return original_run(command,*args,**kwargs)

    if not complete_science(work):
        script = code/'src/deep_pm/production_science.py'
        sys.argv = [str(script),str(work/'science'),str(work/'reference'),str(work/'fit'),str(work/'intake'),str(window)]
        subprocess.run = resume_command
        try:
            runpy.run_path(str(script),run_name='__main__')
        finally:
            subprocess.run = original_run
    else:
        decisions.append('reuse_completed_scientific_receipt')
    if not complete_science(work):
        raise ValueError('Scientific recovery did not produce a valid completed receipt')
    manifest = work/'science/private_continuation_manifest.json'
    blind_file = work/'blind_quality/v17_blind_catalogue_quality_summary.json'
    if blind_file.is_file():
        if read(blind_file)['manifest_sha256']!=digest(manifest):
            raise ValueError('Blind quality belongs to different output ownership')
        decisions.append('reuse_completed_blind_readout')
    else:
        original_run([sys.executable,'-um','deep_pm.blind_catalogue_quality',str(work/'science'),
                      str(work/'blind_quality'),'--workers','16'],check=True)
    save_json(work/'scientific_recovery_summary.json',dict(decisions=decisions,
        frozen_scientific_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip(),
        recovery_source_sha256=digest(Path(__file__)),statistics_helper_sha256=digest(helper),
        scientific_receipt_sha256=digest(work/'science/continuation_science_receipt.json'),
        blind_readout_sha256=digest(blind_file),slurm_job=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('work','code','helper','window'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('Use a Slurm CPU allocation')
    run(**vars(args))
