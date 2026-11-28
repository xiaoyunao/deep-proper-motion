"""Locate registered halos touched by the repeated exposure number, using real positions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .access import Cache, literal
from .io import digest, save_json
from .nsc_measurement_identity import load_exposures, measurement_ids


def intersects(box, vertices, halo):
    west, east = box["ra"]
    south, north = box["dec"]
    lo, hi, bottom, top = vertices
    if top < south - halo or bottom > north + halo:
        return False
    # Shift the interval in units of complete turns, including RA=0 and noncanonical RA.
    return np.ceil((lo - east - halo) / 360) <= np.floor((hi - west + halo) / 360)


def run(work, workers):
    public = work / "registration"
    registry = os.environ["DPM_NSC_IDENTITY_RULE"]
    rule_sha = os.environ["DPM_NSC_IDENTITY_SHA256"]
    rule = load_exposures(registry, rule_sha)
    manifest = public / "private_full_footprint_plan.json"
    fp = digest(manifest)
    if fp != json.loads((public / "v24_footprint_inventory_summary.json").read_text())["signed_manifest_sha256"]:
        raise ValueError("Changed footprint")
    plan = json.loads(manifest.read_text())
    old = work.parent / "v2.2/registration/private_full_coordinate_plan.json"
    if digest(old) != plan["baseline_sha256"]:
        raise ValueError("Changed predecessor footprint")
    old_boxes = {r["core"]: r["spec"] for r in json.loads(old.read_text())["cores"]}
    boxes = dict(old_boxes, **plan["new"])
    cache = Cache(work / "assets/exposures/identity-position-query")
    columns = "m.measid,m.exposure,m.ccdnum,m.mjd,m.ra,m.dec,o.ra AS object_ra,o.dec AS object_dec"
    join = " FROM nsc_dr2.meas m LEFT OUTER JOIN nsc_dr2.object o ON o.id=m.objectid WHERE "

    def fetch(exposure):
        condition = "m.exposure=" + literal(exposure)
        count = int(cache.query("nsc", "SELECT COUNT(*) AS n FROM nsc_dr2.meas m WHERE " + condition, expected=1).n.iat[0])
        if count <= 80000:
            table = cache.query("nsc", "SELECT " + columns + join + condition, expected=count)
        else:
            parts = cache.query("nsc", "SELECT m.ccdnum,COUNT(*) AS n FROM nsc_dr2.meas m WHERE " + condition + " GROUP BY m.ccdnum")
            if int(parts.n.sum()) != count or parts.n.max() > 80000 or parts.ccdnum.duplicated().any():
                raise ValueError("Unresolved high-density exposure partition")
            table = pd.concat([cache.query("nsc", "SELECT " + columns + join + condition + f" AND m.ccdnum={int(r.ccdnum)}",
                                           expected=int(r.n)) for r in parts.itertuples()], ignore_index=True)
        if not table.exposure.eq(exposure).all() or measurement_ids(table).duplicated().any():
            raise ValueError("Repeated physical measurement within the same published exposure")
        bounds = []
        for a, d in [("ra", "dec"), ("object_ra", "object_dec")]:
            matched = table[[a, d]].dropna()
            if len(matched):
                bounds.append([matched[a].min(), matched[a].max(), matched[d].min(), matched[d].max()])
        # Rectangles conservatively include holes; verify against actual inputs before replay.
        candidates = [name for name, box in boxes.items() if any(intersects(box, rect, .060001) for rect in bounds)]
        return dict(exposure=exposure, measurements=count, unmatched_archive_object=int(table.object_ra.isna().sum()),
                    cores=candidates)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(fetch, sorted(rule)))
    selected = sorted({n for row in rows for n in row["cores"]})
    dest = public / "private_exposure_identity_scope.json"
    save_json(dest, dict(footprint=fp, rule_sha256=rule_sha, rows=rows, cores=selected))
    save_json(public / "v24_exposure_identity_scope_summary.json",
              dict(footprint=fp, rule_sha256=rule_sha, scope_sha256=digest(dest),
                   exposures=len(rows), measurements=sum(v["measurements"] for v in rows),
                   unmatched_archive_object=sum(v["unmatched_archive_object"] for v in rows),
                   candidate_old=len(set(old_boxes).intersection(selected)),
                   candidate_new=len(set(plan["new"]).intersection(selected)), slurm_job=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1 <= args.workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        parser.error("Use the assigned Slurm CPUs")
    run(args.work, args.workers)
