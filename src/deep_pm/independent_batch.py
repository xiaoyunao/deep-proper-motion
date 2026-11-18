"""Prepare small resumable acquisition shards for a predeclared sky batch."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .__main__ import tomllib
from .io import digest,save_json
from .provenance import run_stage


def split_cell(row):
    a,b=row['ra_bounds_deg'];c,d=row['dec_bounds_deg']
    return {f"{row['name']}_{i}{j}":dict(ra=[a+i*(b-a)/2,a+(i+1)*(b-a)/2],
        dec=[c+j*(d-c)/2,c+(j+1)*(d-c)/2]) for i in range(2) for j in range(2)}


def prepare(work,selection,protocol):
    selected=json.loads(selection.read_text())['first_batch']
    config=tomllib.loads(protocol.read_text())
    plan=[]
    configs={}
    for row in selected:
        box=row['box']
        center=[float(np.mean(config['boxes'][box][k])) for k in ('ra','dec')]
        lines=[f'projection_center_deg = {center}', 'halo_deg = 0.06', 'seed = 9137',
               'reference_epoch = 2015.5','nside = 128','native_join = true','native_count_window = true',
               'native_row_cap = 80000','download_workers = 1']
        for core,bounds in split_cell(row).items():
            lines.extend([f'[fields.{core}]',f"ra = {bounds['ra']}",f"dec = {bounds['dec']}"])
            plan.append((core,box))
        path=work/f'batch-{box}.toml'
        path.write_text('\n'.join(lines)+'\n')
        save_json(work/f'references-{box}.json',{core:str((work/'refs'/core).resolve()) for core in split_cell(row)})
        configs[box]=dict(path=path.name,sha256=digest(path))
    (work/'tasks.tsv').write_text(''.join(f'{name}\t{box}\n' for name,box in plan))
    save_json(work/'batch_preparation_summary.json',dict(selection_sha256=digest(selection),
        selected_parent_count=len(selected),acquisition_core_count=len(plan),configs=configs,
        selection_uses_new_motion=False,area_sqdeg=sum(row['area_sqdeg'] for row in selected)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','selection','protocol'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    link=a.work/'selection.json'
    if not link.exists():link.symlink_to(a.selection.resolve(strict=True))
    if link.resolve()!=a.selection.resolve(strict=True):raise ValueError('Selection changed')
    # Configuration files are deterministic outputs, bound by the input selection.
    names=json.loads(a.selection.read_text())['first_batch']
    outputs=['tasks.tsv','batch_preparation_summary.json']+[f"batch-{r['box']}.toml" for r in names]+[f"references-{r['box']}.json" for r in names]
    with run_stage(a.work,a.protocol,'batch_01','independent-batch-prepare-v13',inputs=('selection.json',),outputs=outputs) as go:
        if go:prepare(a.work,a.selection,a.protocol)


if __name__=='__main__':
    main()
