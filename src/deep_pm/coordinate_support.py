"""Inventory existing validation, CCD and photometry metadata before new queries."""
import argparse
import os
from pathlib import Path

import pandas as pd

from .blind_catalogue_quality import read
from .continuous_instruments import chip_receipts
from .io import save_json
from .public_summary import check_aggregate


def run(registration,root,output):
    plan=read(registration);output.mkdir(parents=True,exist_ok=True);report=dict(cores={},schemas={})
    selected=[r for r in plan['selected'] if r['ngc']][:1]+[r for r in plan['selected'] if r['core'] in plan['pilot']]
    for row in selected:
        source=Path(row['source']);chips,complete=chip_receipts(source/'run_manifest.json')
        columns={}
        for name in ('sdss.csv','parent_sdss.csv','validation/gaia.csv','chips.csv'):
            file=source/name
            if file.exists():columns[name]=list(pd.read_csv(file,nrows=0))
        report['cores'][row['core']]=dict(input_headers=columns,verified_chip_parts=len(chips),chip_parts_complete=complete,
            chip_columns=list(pd.read_csv(chips[0],nrows=0)) if chips else [],
            source_file_names=sorted(p.name for p in source.iterdir()))
    for name in ('nsc_schema.csv','native_schema.csv'):
        file=root/'cache'/name
        if not file.exists():continue
        table=pd.read_csv(file)
        # This is schema text only: column names/units/descriptions, no source rows.
        report['schemas'][name]=dict(columns=list(table),schema_records=table.astype(str).to_dict('records'))
    report['slurm_job']=os.environ['SLURM_JOB_ID'];check_aggregate(report)
    save_json(output/'v21_support_metadata_summary.json',report)
    print('SUPPORT_METADATA_DONE',len(selected),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registration','root','output'):p.add_argument(n,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use Slurm resources')
    run(**vars(a))
