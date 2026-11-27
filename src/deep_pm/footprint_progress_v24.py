"""A batch-area ledger from signed completed acquisitions; never infer sky from sources."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from .continuous_summary import area
from .io import digest,save_json


def read(path):
    return json.loads(path.read_text())


def run(base,out):
    root=base/"v2.4";reg=root/"registration"
    manifest=reg/"private_full_footprint_plan.json"
    sha=digest(manifest);cells=read(manifest)["new"]
    completed={}
    for p in sorted(reg.glob("v24_intake_*_summary.json")):
        pilot=read(p)
        if pilot["footprint_sha256"]!=sha:raise ValueError("Changed pilot footprint")
        completed.update({r["core"]:r for r in pilot["rows"] if "error" not in r})
    failures=set()
    for p in (root/"batch-receipts").glob("*.json"):
        if p.name.endswith(".failed.json"):
            failures.add(p.name.split(".")[0]);continue
        rec=read(p)
        if rec["manifest_sha256"]!=sha or rec["core"] not in cells:raise ValueError("Unregistered intake result")
        completed[rec["core"]]=rec
    failures-=set(completed)
    if not completed:raise ValueError("No completed input receipts")
    values=list(completed.values());r=np.array([row["rows"] for row in values])
    result=dict(signed_footprint=sha,registered_new_core_count=len(cells),completed_new_inputs=len(values),
                remaining_new_inputs=len(cells)-len(values),failed_unrepaired_inputs=len(failures),
                input_rectangles_nominal_sqdeg=sum(area(cells[k]) for k in completed),
                time_opportunity_in_completed_new_inputs_sqdeg=sum(cells[k]["time_area"] for k in completed),
                input_rows_including_halo=int(r.sum()),input_bytes=sum(v["raw_bytes"] for v in values),
                reference_rows_including_halo=sum(v["measurements"] for v in values),
                cores_with_local_reference_rows=sum(v["measurements"]>0 for v in values),
                core_rows_p10_p50_p90=np.quantile(r,[.1,.5,.9]),
                worker_seconds_sum=sum(v["wall_s"] for v in values),
                peak_individual_rss_mib=max(v["rss_mib"] for v in values),
                slurm_job=os.environ["SLURM_JOB_ID"],
                interpretation="Completed intake with signed source and reference extraction. Current graph/fit results are accounted separately. Shared detections across core halos occur in these row counts.")
    save_json(out,result)
    print("INPUTS",result["completed_new_inputs"],"ROWS",int(r.sum()),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use a Slurm node")
    run(a.base,a.output)
