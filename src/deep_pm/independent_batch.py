"""Prepare small resumable acquisition shards for a predeclared sky batch."""
import argparse
import json
import os
import sys
from importlib import metadata
from collections import defaultdict
from pathlib import Path

import numpy as np

from .__main__ import tomllib
from .io import digest,save_json
from .provenance import run_stage
from .continuous_reference import FILES, verify_directory


def split_cell(row):
    a,b=row['ra_bounds_deg'];c,d=row['dec_bounds_deg']
    return {f"{row['name']}_{i}{j}":dict(ra=[a+i*(b-a)/2,a+(i+1)*(b-a)/2],
        dec=[c+j*(d-c)/2,c+(j+1)*(d-c)/2]) for i in range(2) for j in range(2)}


def prepare(work,selection,protocol,previous=None):
    selected=json.loads(selection.read_text())['first_batch']
    config=tomllib.loads(protocol.read_text())
    plan=[]
    configs={}
    boxes=defaultdict(list)
    for row in selected:boxes[row['box']].append(row)
    if len({r['name'] for r in selected})!=len(selected):raise ValueError('Duplicate selected parents')
    for box,rows in boxes.items():
        center=[float(np.mean(config['boxes'][box][k])) for k in ('ra','dec')]
        lines=[f'projection_center_deg = {center}', 'halo_deg = 0.06', 'seed = 9137',
               'reference_epoch = 2015.5','nside = 128','native_join = true','native_count_window = true',
               'native_row_cap = 80000','download_workers = 1']
        fields={core:bounds for row in rows for core,bounds in split_cell(row).items()}
        for core,bounds in fields.items():
            lines.extend([f'[fields.{core}]',f"ra = {bounds['ra']}",f"dec = {bounds['dec']}"])
            plan.append((core,box))
        path=work/f'batch-{box}.toml'
        path.write_text('\n'.join(lines)+'\n')
        save_json(work/f'references-{box}.json',{core:str((work/'refs'/core).absolute()) for core in fields})
        configs[box]=dict(path=path.name,sha256=digest(path))
        if previous is not None and (previous/f'batch-{box}.toml').exists():
            old=tomllib.loads((previous/f'batch-{box}.toml').read_text())
            current=tomllib.loads(path.read_text())
            if {k:v for k,v in old.items() if k!='fields'}!={k:v for k,v in current.items() if k!='fields'}:
                raise ValueError('Reused acquisition parameters or projection changed')
            for core,bounds in fields.items():
                if core not in old['fields']:continue
                if old['fields'][core]!=bounds:raise ValueError('Reused core geometry changed')
                for category,outputs in [('inputs',FILES),('refs',('reference.csv.gz','reference_summary.json'))]:
                    target=previous/category/core
                    verify_directory(target,outputs)
                    link=work/category/core
                    link.parent.mkdir(parents=True,exist_ok=True)
                    if not link.exists():link.symlink_to(target.resolve())
                    if link.resolve()!=target.resolve():raise ValueError('Reused cache target changed')
    if previous is not None:
        target=previous/'catalogues'
        verify_directory(target,tuple(f'xiao_{box}.csv' for box in boxes))
        link=work/'catalogues'
        if not link.exists():link.symlink_to(target.resolve())
        if link.resolve()!=target.resolve():raise ValueError('Legacy cache target changed')
    (work/'tasks.tsv').write_text(''.join(f'{name}\t{box}\n' for name,box in plan))
    versions={}
    for name in ("numpy","scipy","pandas","astropy","matplotlib","scikit-learn","requests","healpy","tomli"):
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]=None
    save_json(work/'batch_preparation_summary.json',dict(runtime_versions=dict(python=sys.version.split()[0],packages=versions),selection_sha256=digest(selection),
        selected_parent_count=len(selected),acquisition_core_count=len(plan),configs=configs,
        final_test_box=config.get("final_test_box"),
        selection_uses_new_motion=False,area_sqdeg=sum(row['area_sqdeg'] for row in selected)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','selection','protocol'):
        p.add_argument(name,type=Path)
    p.add_argument('--previous',type=Path)
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
    inputs=['selection.json']
    if a.previous is not None:
        receipt=a.previous/'run_manifest.json'
        link=a.work/'previous_preparation_receipt.json'
        if not link.exists():link.symlink_to(receipt.resolve(strict=True))
        if link.resolve()!=receipt.resolve():raise ValueError('Previous batch changed')
        inputs.append(link.name)
    with run_stage(a.work,a.protocol,'independent_batch','independent-batch-prepare-v13',inputs=inputs,outputs=outputs) as go:
        if go:prepare(a.work,a.selection,a.protocol,a.previous)


if __name__=='__main__':
    main()
