"""Bind the immutable full intake to one physical calibration graph on PScratch."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from .io import digest,save_json


def read(path):
    return json.loads(path.read_text())


def link(source,dest):
    source=source.resolve(strict=True)
    if dest.is_symlink():
        if dest.resolve(strict=True)!=source:raise ValueError("Inconsistent prior shard link")
    elif dest.exists():
        raise ValueError("Would replace an existing reference shard")
    else:dest.symlink_to(source,target_is_directory=source.is_dir())


def assemble(base):
    root=base/"v2.4"
    output=root/"registration"
    manifest=output/"private_full_footprint_plan.json"
    frozen=read(manifest)
    expected=read(output/"v24_footprint_inventory_summary.json")["signed_manifest_sha256"]
    if digest(manifest)!=expected:raise ValueError("Changed survey footprint")
    check=root/"meridian/v24_archive_input_audit_summary.json"
    audit=read(root/"meridian/private_archive_audit.json")
    if read(check)["private_audit_sha256"]!=digest(root/"meridian/private_archive_audit.json"):
        raise ValueError("Changed archive audit")
    unresolved=[name for name,r in audit.items() if r.get("requires_input_replay") or "requires_input_replay" not in r]
    if unresolved:
        raise RuntimeError(f"{len(unresolved)} archive checks or verified differences remain; inspect per-core receipts")
    prior=base/"v2.2";prior_plan=prior/"registration/private_full_coordinate_plan.json"
    if digest(prior_plan)!=frozen["baseline_sha256"]:raise ValueError("Changed old core registration")
    graphs={}
    for graph in sorted((prior/"reference").glob("component-*/private_c0_unfiltered.npz")):
        summary=read(graph.parent/"v22_reference_component_summary.json")
        if digest(graph)!=summary["graph_sha256"]:raise ValueError("Changed old physical reference")
        graphs[str(graph)]=dict(sha256=summary["graph_sha256"],center=summary["center_deg"])
    old=read(prior_plan)
    registered=[]
    def inputs(label,source,shard):
        pointer=root/"resolved"/(label+".json")
        if not pointer.is_file():return source,shard
        record=read(pointer);source,shard=Path(record["source"]),Path(record["shard"])
        if record["source_sha256"]!=digest(source/"run_manifest.json") or record["reference_sha256"]!=digest(shard/"v22_reference_input_core_summary.json"):
            raise ValueError("Changed revision of source or reference")
        return source,shard
    layout=root/"joint"
    (layout/"input").mkdir(exist_ok=True,parents=True)

    for entry in old["cores"]:
        label=entry["core"]
        source,oldshard=inputs(label,Path(entry["source"]),prior/"input"/label)
        proof=read(oldshard/"v22_reference_input_core_summary.json")
        if proof["parquet_sha256"]!=digest(oldshard/"reference.parquet"):
            raise ValueError("Changed frozen old input")
        link(oldshard,layout/"input"/label)
        registered.append(dict(core=label,spec=entry["spec"],window=entry["window"],source=str(source),
                               center=entry["center"],source_receipt_sha256=digest(source/"run_manifest.json"),
                               predecessor=str(prior/"sources"/label),production_blind=True))
    missing=[]
    for label,spec in frozen["new"].items():
        source,shard=inputs(label,root/"intake"/label,root/"input"/label)
        proof=shard/"v22_reference_input_core_summary.json"
        if not proof.exists():
            missing.append(label)
            continue
        r=read(proof)
        if r["parquet_sha256"]!=digest(shard/"reference.parquet") or r["ticket"]["source_receipt_sha256"]!=digest(source/"run_manifest.json"):
            raise ValueError("Changed completed acquisition")
        link(shard,layout/"input"/label)
        registered.append(dict(core=label,spec=spec,window=label,source=str(source),
                               center=[sum(spec[k])/2 for k in ("ra","dec")],
                               source_receipt_sha256=r["ticket"]["source_receipt_sha256"],production_blind=True))
    if missing:
        save_json(output/"private_outstanding_intake.json",dict(manifest_sha256=expected,missing=missing))
        raise RuntimeError(f"{len(missing)} registered intake shards still pending; graph has not been frozen")
    from .footprint_intake_v24 import select_pilots
    pilots=set(select_pilots(frozen["new"]))
    for entry in registered:
        tag=sha256(("physical-r-before-quality-v24:"+entry["core"]).encode()).digest()
        entry["save_r_comparison"]=entry["core"] in pilots or int.from_bytes(tag[:8],"big")%64==0
    result=dict(cores=registered,graphs=graphs,production_blind=True,footprint_sha256=expected,
                frozen_science="0320f59b972755454252a383ba1007c665c928e8",
                predecessor_index=str(prior/"catalogue/catalogue.duckdb"))
    path=layout/"registration/private_full_coordinate_plan.json"
    if path.is_file() and read(path)!=result:raise ValueError("Changed graph binding")
    save_json(path,result)
    save_json(output/"v24_graph_registration_summary.json",dict(cores=len(registered),old=len(old["cores"]),
              new=len(frozen["new"]),manifest_sha256=expected,graph_registration_sha256=digest(path)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Verify real signed shards on Slurm")
    assemble(a.base)
