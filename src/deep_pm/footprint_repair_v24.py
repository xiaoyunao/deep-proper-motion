"""Replay only confirmed archive differences and bind their revised references."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from .footprint_intake_v24 import one
from .io import digest,save_json


def fix(task):
    return one(task,revision="verified-archive")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("base","catalogues","sas"):p.add_argument(n,type=Path)
    p.add_argument("--workers",type=int,required=True);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):p.error("Use assigned Slurm CPUs")
    root=a.base/"v2.4";reg=root/"registration";directory=root/"meridian"
    audit=directory/"private_archive_audit.json"
    logged=json.loads((directory/"v24_archive_input_audit_summary.json").read_text())
    if digest(audit)!=logged["private_audit_sha256"]:raise ValueError("Changed private audit")
    manifest=reg/"private_full_footprint_plan.json";sig=digest(manifest)
    cells=json.loads(manifest.read_text())["new"]
    old=json.loads((a.base/"v2.2/registration/private_full_coordinate_plan.json").read_text())
    spec={**cells,**{r["core"]:r["spec"] for r in old["cores"]}}
    names=[k for k,v in json.loads(audit.read_text()).items() if v.get("requires_input_replay")]
    jobs=[(k,spec[k],a.base,a.catalogues,a.sas,sig) for k in names]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(fix,jobs))
    save_json(reg/"v24_longitude_replay_summary.json",dict(signed_footprint=sig,cores=len(rows),
              input_rows=sum(r["rows"] for r in rows),reference_rows=sum(r["measurements"] for r in rows),
              private_audit_sha256=digest(audit)))


if __name__=="__main__":main()
