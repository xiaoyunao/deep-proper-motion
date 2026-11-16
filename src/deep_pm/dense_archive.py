"""Measure count-checked acquisition and restart on a COSMOS degree-scale subregion."""
import argparse
import json
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np
import pandas as pd

from . import access
from .io import digest, save_json
from .provenance import run_stage

CENTER = (150.12, 2.20)
AREA = .12
SHARDS = 12


def fetch(work, shard, radius, bounds):
    folder=work/f"shard_{shard:02d}"
    folder.mkdir(exist_ok=True)
    lo,hi=bounds
    where=(access.q3c(*CENTER,radius)
           +f" AND dec >= {lo:.13f} AND dec < {hi:.13f}"
           +" AND ndetr>0 AND (rmag<24 OR nphotr=0 OR rmag IS NULL)")
    table=folder/"measurements.csv.gz"
    receipt=folder/"receipt.json"
    index_file=folder/"objects.csv"
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if (saved["predicate"]!=where or saved["measurement_sha256"]!=digest(table)
                or saved["object_sha256"]!=digest(index_file)):
            raise ValueError("Interrupted shard does not match its frozen receipt.")
        return saved,True
    cache=access.Cache(folder/"cache")
    start=monotonic()
    index=access.checked(cache,"nsc","id AS objectid","nsc_dr2.object",where)
    ids=index.objectid.astype(str).sort_values()
    if ids.duplicated().any():
        raise ValueError("Nonunique NSC archive index.")
    data=access.fetch_measurements_for_objects(cache,ids,lambda _:None,batch_size=200)
    if set(data.objectid.astype(str))!=set(ids):
        raise ValueError("Every selected r object must have returned measurements.")
    index.to_csv(index_file,index=False)
    temporary=table.with_name(table.name+".part")
    data.to_csv(temporary,index=False,compression="gzip")
    temporary.replace(table)
    saved=dict(predicate=where,objects=len(ids),measurements=len(data),bytes=table.stat().st_size,
               elapsed_seconds=monotonic()-start,measurement_sha256=digest(table),
               object_sha256=digest(index_file))
    save_json(receipt,saved)
    return saved,False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("--checkpoint",action="store_true")
    p.add_argument("--config",type=Path,default=Path("configs/regional-v07.toml"))
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Use a Slurm compute node.")
    args.work.mkdir(parents=True,exist_ok=True)
    radius=np.rad2deg(np.arccos(1-AREA/(2*np.pi)*(np.pi/180)**2))
    edges=CENTER[1]+np.linspace(-radius,radius,SHARDS+1)
    edges[0]-=1e-9
    edges[-1]+=1e-9
    outputs=["dense_archive_summary.json"]
    inputs=["shard_00/receipt.json"] if not args.checkpoint else []
    stage="dense-checkpoint-v07" if args.checkpoint else "dense-archive-v07"
    if args.checkpoint:
        outputs=["shard_00/receipt.json"]
    with run_stage(args.work,args.config,"cosmos_dense",stage,inputs=inputs,outputs=outputs) as go:
        if not go:
            return
        start=monotonic()
        rows=[]
        resumed=0
        for i in range(1 if args.checkpoint else SHARDS):
            saved,hit=fetch(args.work,i,radius,edges[i:i+2])
            resumed+=hit
            item={k:saved[k] for k in ("objects","measurements","bytes","elapsed_seconds")}
            rows.append(item)
            print(f"COSMOS shard {i+1}/{SHARDS} {item['objects']} objects {item['measurements']} measurements resumed={hit}",flush=True)
        if args.checkpoint:
            return
        identifiers=set()
        for i in range(SHARDS):
            block=pd.read_csv(args.work/f"shard_{i:02d}"/"objects.csv",dtype=str).objectid
            if identifiers.intersection(block):
                raise ValueError("Overlapping COSMOS archive shards.")
            identifiers.update(block)
        save_json(args.work/outputs[0],dict(area_sqdeg=AREA,shard_count=SHARDS,
                  objects=len(identifiers),measurements=sum(x["measurements"] for x in rows),
                  compressed_bytes=sum(x["bytes"] for x in rows),shards=rows,
                  resumed_shards=resumed,run_elapsed_seconds=monotonic()-start,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))


if __name__=="__main__":
    main()
