"""Read the actual Xiao2025 FITS once for all preregistered new windows."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd

from .__main__ import tomllib
from .exposure_catalogues import inside
from .io import digest,save_json
from .provenance import run_stage


def extract(work,source,boxes):
    before=source.stat()
    requested=('objID','RA','DEC','PM_SOURCE','pmRA','pmDEC','pmRA_err','pmDEC_err')
    counts=Counter()
    flags={name:Counter() for name in boxes}
    for name in boxes:
        (work/f'xiao_{name}.part').unlink(missing_ok=True)
    with fits.open(source,memmap=True,character_as_bytes=True) as handle:
        table=handle[1].data
        lookup={name.casefold():name for name in table.names}
        if not all(k.casefold() in lookup for k in requested):
            raise ValueError('Legacy catalogue schema is incomplete')
        columns=[lookup[k.casefold()] for k in requested]
        total=len(table)
        schema={c.name:dict(format=c.format,unit=c.unit) for c in handle[1].columns}
        for start in range(0,total,1000000):
            stop=min(total,start+1000000)
            ra=table[lookup['ra']][start:stop]
            dec=table[lookup['dec']][start:stop]
            for name,box in boxes.items():
                idx=np.flatnonzero(inside(ra,dec,box,.06))+start
                if not len(idx):continue
                values={key:table[key][idx] for key in columns}
                frame=pd.DataFrame({key:np.char.strip(v.astype(str)) if key.casefold()=='pm_source'
                    else v.astype(v.dtype.newbyteorder('='),copy=False) for key,v in values.items()})
                path=work/f'xiao_{name}.part'
                frame.to_csv(path,index=False,mode='a',header=not path.exists())
                counts[name]+=len(frame)
                flags[name].update(frame[lookup['pm_source']].value_counts().to_dict())
            if start%30000000==0:print(f'Legacy rows inspected {stop}/{total}',flush=True)
    after=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Legacy FITS changed during extraction')
    outputs={}
    for name in boxes:
        part=work/f'xiao_{name}.part'
        if not part.exists():pd.DataFrame(columns=columns).to_csv(part,index=False)
        output=work/f'xiao_{name}.csv'
        part.replace(output)
        outputs[name]=dict(catalogue_count=counts[name],pm_source_counts=flags[name],sha256=digest(output))
    checksum=digest(source)
    final=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(final.st_size,final.st_mtime_ns):
        raise ValueError('Legacy FITS changed during checksum verification')
    save_json(work/'legacy_region_summary.json',dict(parent_count=total,parent_bytes=before.st_size,
        parent_sha256=checksum,schema=schema,regions=outputs,slurm_job=os.environ['SLURM_JOB_ID']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('work','source','config'):p.add_argument(name,type=Path)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):p.error('Use a Slurm compute node')
    a.work.mkdir(parents=True,exist_ok=True)
    config=tomllib.loads(a.config.read_text())
    stat=a.source.stat()
    current=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino,device=stat.st_dev)
    receipt=a.work/'legacy_input_receipt.json'
    if receipt.exists():
        if json.loads(receipt.read_text())!=current:p.error('Legacy source changed; use a new run directory')
    else:save_json(receipt,current)
    outputs=['legacy_region_summary.json']+[f'xiao_{name}.csv' for name in config['boxes']]
    with run_stage(a.work,a.config,'new_windows','legacy-regions-v13',inputs=('legacy_input_receipt.json',),outputs=outputs) as go:
        if go:extract(a.work,a.source,config['boxes'])


if __name__=='__main__':main()
