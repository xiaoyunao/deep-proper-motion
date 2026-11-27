"""Locate old receipts affected by true, noncanonical archive metadata."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .exposure_catalogues import inside
from .exposure_preflight import where_box
from .access import Cache,checked_spatial
from .io import digest,save_json


def read(path):return json.loads(path.read_text())


def spatial_groups(df):
    ix=(np.floor(df.ra_normalized).astype(int)%360+360*np.floor(df.dec+90).astype(int)).to_numpy()
    return {int(key):group for key,group in df.groupby(ix,sort=False)}


def nearby(by_cell,spec,padding):
    west,east=spec["ra"];south,north=spec["dec"]
    labels={j%360+360*d for j in range(int(np.floor(west-padding)),int(np.ceil(east+padding))+1)
            for d in range(max(0,int(np.floor(south+90-padding))),min(180,int(np.ceil(north+90+padding)))+1)}
    groups=[by_cell[k] for k in labels if k in by_cell]
    if not groups:return None
    local=pd.concat(groups,ignore_index=True)
    return local.loc[inside(local.ra_normalized,local.dec,spec,padding)]


def run(base):
    root=base/"v2.4";out=root/"meridian"
    indices=read(out/"v24_archive_longitude_summary.json")
    path=out/"private_noncanonical_object.parquet"
    if digest(path)!=indices["input_sha256"]:raise ValueError("Changed raw-longitude census")
    raw=pd.read_parquet(path,columns=["id","ra","dec","ra_normalized"])
    raw=raw[raw.ra.between(-360,720)&raw.dec.between(-90,90)]
    objects=spatial_groups(raw)
    chips_dir=root/"chips";metadata=read(chips_dir/"private_chip_manifest.json")
    chip_path=chips_dir/"private_noncanonical_centres.parquet"
    if metadata["noncanonical_sha256"]!=digest(chip_path):raise ValueError("Changed noncanonical CCD census")
    chip=pd.read_parquet(chip_path,columns=["exposure","ccdnum","ra","dec"])
    chip=chip[chip.ra.between(-360,720)&chip.dec.between(-90,90)].copy()
    chip["ra_normalized"]=np.mod(chip.ra,360.)
    ccds=spatial_groups(chip)
    old=read(base/"v2.2/registration/private_full_coordinate_plan.json")
    manifest=read(root/"registration/private_full_footprint_plan.json")
    entries=[(e["core"],e["spec"],Path(e["source"]),True) for e in old["cores"]]
    entries.extend((label,spec,root/"intake"/label,False) for label,spec in manifest["new"].items())
    cache=Cache(out/"canonical-object-check")
    suspect={};done=affected=0
    for label,spec,source,oldcore in entries:
        ids=nearby(objects,spec,.06)
        c=nearby(ccds,spec,.71)
        # Old, pre-v2.4 archive SQL did not wrap even canonical east/west
        # coordinates in halo queries. Compare the full rectangle on replay.
        crosses=oldcore and (spec["ra"][0]<.71 or spec["ra"][1]>359.29)
        if not crosses and (ids is None or not len(ids)) and (c is None or not len(c)):continue
        record=dict(old=oldcore,noncanonical_targets=0 if ids is None else len(ids),
                    noncanonical_ccds=0 if c is None else len(c),historical_meridian_query=crosses)
        pointer=root/"resolved"/(label+".json")
        if pointer.is_file():source=Path(read(pointer)["source"])
        if (source/"acquisition_summary.json").is_file():
            missing=set()
            surplus=set()
            available=set(pd.read_csv(source/"official_nsc.csv",usecols=["id"],dtype=str).id)
            if ids is not None and len(ids):
                missing=set(ids.id.astype(str))-available
            if crosses:
                south,north=spec["dec"]
                all_ids=checked_spatial(cache,"nsc","id","nsc_dr2.object",where_box(spec,.06),south-.0600001,north+.0600001)
                if all_ids.id.duplicated().any():raise ValueError("Nonunique canonical object census")
                expected=set(all_ids.id.astype(str));missing=expected-available;surplus=available-expected
            chip_missing=set()
            if c is not None and len(c):
                known=pd.read_csv(source/"chips.csv",usecols=["exposure","ccdnum"])
                chip_missing=set(zip(c.exposure,c.ccdnum))-set(zip(known.exposure,known.ccdnum))
            # Absent incidental chips matter only for actual science detections.
            used=set()
            if chip_missing:
                for sub in pd.read_csv(source/"nsc_object.csv",usecols=["exposure","ccdnum"],chunksize=150000):
                    used.update(set(zip(sub.exposure,sub.ccdnum))&chip_missing)
            record.update(missing_archive_ids=len(missing),missing_used_chip_metadata=len(used),
                          surplus_archive_ids=len(surplus),requires_input_replay=bool(missing or used or surplus))
            done+=1;affected+=int(record["requires_input_replay"])
        suspect[label]=record
    private=out/"private_archive_audit.json";save_json(private,suspect)
    save_json(out/"v24_archive_input_audit_summary.json",dict(registered=len(entries),possible=len(suspect),
              signed_inputs_checked=done,inputs_for_replay=affected,private_audit_sha256=digest(private),
              scope="Physical raw-coordinate anomalies and historic unwrapped meridian rectangles. Missing source data remains pending."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use a Slurm compute node")
    run(a.base)
