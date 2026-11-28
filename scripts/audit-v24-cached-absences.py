"""Inventory archive objects absent from already-signed measurement pages without issuing a query."""
import argparse
import json
import os
from pathlib import Path

from deep_pm.footprint_dense_v24 import stage
from deep_pm.io import digest, save_json


def audit(work, core, output):
    plan = work / "registration/private_full_footprint_plan.json"
    footprint = digest(plan)
    registered = json.loads((work / "registration/v24_footprint_inventory_summary.json").read_text())
    if footprint != registered["signed_manifest_sha256"]:
        raise ValueError("Changed registered footprint")
    spec = json.loads(plan.read_text())["new"][core]
    receipt = stage((work, core, spec, footprint), prepare_absences_only=True)
    if receipt is None:
        raise ValueError("Core already has its complete science receipt")
    path = work / "dense-absence" / (core + ".json")
    missing = json.loads(path.read_text())
    count = len(missing["missing"])
    save_json(output, dict(footprint=footprint, slurm_job=os.environ["SLURM_JOB_ID"],
                           checked_pages=receipt["checked_pages"], archive_objects=receipt["archive_objects"],
                           measurements=receipt["delivered_measurements"], absent_objects=count,
                           canonical_direct_query_pairs=(count+249)//250, network_queries=0,
                           page_receipt_sha256=missing["page_receipt_sha256"], absence_sha256=digest(path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("core")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Use a Slurm computation step")
    audit(args.work, args.core, args.output)
