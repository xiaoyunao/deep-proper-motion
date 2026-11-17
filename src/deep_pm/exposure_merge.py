"""Merge stable detection and catalogue identities across acquisition halos."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .__main__ import tomllib
from .exposure_catalogues import inside
from .io import digest, save_json
from .provenance import run_stage

FILES = {
    "detections.csv": ["detection_id"],
    "external_labels.csv": ["detection_id"],
    "official_nsc.csv": ["id"],
    "spectra_union.csv": ["ra","dec","origin"],
    "spectra_veto.csv": ["ra","dec","origin"],
    "validation/gaia.csv": ["source_id"],
}


def merge(work,parts,box):
    prior=[]
    for part in parts:
        rec=json.loads((part/"run_manifest.json").read_text())["stages"]["exposure-inputs-v09"][-1]
        if rec["state"]!="completed":
            raise ValueError(f"Part {part.name} has incomplete acquisition")
        for name in (*FILES,"acquisition_summary.json"):
            if digest(part/name)!=rec["outputs"].get(name):
                raise ValueError(f"Checksum differs for {part.name}/{name}")
        prior.append(json.loads((part/"acquisition_summary.json").read_text()))
    sizes={}
    for name,keys in FILES.items():
        out=work/name
        out.parent.mkdir(exist_ok=True)
        dtypes={k:str for k in ("detection_id","id","original_object_id") if k in keys or name=="external_labels.csv"}
        frame=pd.concat([pd.read_csv(p/name,low_memory=False,dtype=dtypes) for p in parts],ignore_index=True)
        merged=frame.drop_duplicates(keys).copy()
        if name.startswith("spectra"):
            merged["in_core"]=inside(merged.ra,merged.dec,box)
        if name=="detections.csv":
            merged["in_core"]=inside(merged.ra_deg,merged.dec_deg,box)
            merged.sort_values(["observation_time","detection_id"],inplace=True)
        merged.to_csv(out,index=False)
        sizes[name.split("/")[-1]]={"source_records":len(frame),"unique_records":len(merged)}
    save_json(work/"acquisition_summary.json",dict(shard_count=len(parts),counts=sizes,shard_resources=prior))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("region")
    p.add_argument("parts",type=Path)
    p.add_argument("config",type=Path)
    a=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run on a Slurm compute node")
    box=tomllib.loads(a.config.read_text())["fields"][a.region]
    work=a.root/a.region
    work.mkdir(parents=True,exist_ok=True)
    inputs=[]
    for name in box["parts"]:
        pth=(a.parts/name/"run_manifest.json").resolve(strict=True)
        link=work/f"{name}_manifest.json"
        if not link.exists():
            link.symlink_to(pth)
        if link.resolve()!=pth:
            raise ValueError("Pilot is bound to different acquisition parts")
        inputs.append(link.name)
    with run_stage(work,a.config,a.region,"exposure-merge-v09",inputs=inputs,
                   outputs=(*FILES,"acquisition_summary.json")) as go:
        if go:
            merge(work,[a.parts/k for k in box["parts"]],box)


if __name__=="__main__":
    main()
