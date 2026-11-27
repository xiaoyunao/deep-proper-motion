"""Finish one warmed dense core in its immutable original science checkout."""
import argparse
import json
import os
from pathlib import Path
from deep_pm.footprint_batch_v24 import worker
from deep_pm.io import digest


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("base","catalogues","sas"):
        p.add_argument(n,type=Path)
    p.add_argument("label")
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        p.error("Use a Slurm allocation")
    work=a.base/"v2.4"
    plan=work/"registration/private_full_footprint_plan.json"
    sha=digest(plan)
    warm=json.loads((work/"dense-pages"/(a.label+".json")).read_text())
    if warm["footprint_sha256"]!=sha or warm["core"]!=a.label:
        raise ValueError("Changed staged query footprint")
    names=json.loads(plan.read_text())["new"]
    control=work/"batch-receipts"
    task=(a.label,names[a.label],a.base,a.catalogues,a.sas,sha)
    result=worker(([task],control))
    proof=control/(a.label+".json")
    if not proof.is_file() or json.loads(proof.read_text())["manifest_sha256"]!=sha:
        raise ValueError(f"Dense source or reference did not close: {result}")
