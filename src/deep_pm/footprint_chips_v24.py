"""Disjoint all-sky NSC CCD metadata shards, including genuinely empty cells."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from time import monotonic

import pandas as pd

from . import access
from .io import digest, save_json

COLUMNS = ("exposure,ccdnum,ra,dec,mjd,instrument,filter," +
           ",".join(f"vertex_{axis}{i}" for i in range(1, 5) for axis in ("ra", "dec")))


def part(root, box, expected):
    a, b, d, e = box
    key = "_".join(f"{x:g}" for x in box)
    path = root / (key + ".parquet")
    receipt = path.with_suffix(".json")
    if receipt.exists():
        q = json.loads(receipt.read_text())
        if q["rows"] != expected or not path.exists() or digest(path) != q["sha256"]:
            raise ValueError(f"Changed completed CCD cell {key}")
        return [q]
    if not expected:
        return []
    region = (f"ra>={a} AND ra<{b} AND dec>={d} AND dec"
              f"{'<=' if e == 90 else '<'}{e}")
    # Triangle inequality in lon/lat paths gives an enclosing spherical cap.
    radius = .5 * (b-a+e-d) + .001
    where = (f"q3c_radial_query(ra,dec,{(a+b)/2},{(d+e)/2},{radius}) AND " + region)
    if expected > 60000:
        mid = (a+b)/2
        n, _ = access.request("nsc", f"SELECT count(*) AS n FROM nsc_dr2.chip WHERE {where} AND ra<{mid}")
        left = int(n.n.iat[0])
        if b-a < 1.e-7 or not 0 <= left <= expected:
            raise ValueError("Invalid metadata bisection")
        return part(root, (a,mid,d,e), left) + part(root, (mid,b,d,e), expected-left)
    t = monotonic()
    df, raw = access.request("nsc", f"SELECT {COLUMNS} FROM nsc_dr2.chip WHERE {where}")
    if len(df) != expected or df.duplicated(["exposure","ccdnum"]).any():
        raise ValueError(f"CCD partition count or physical-key mismatch {key}, {len(df)}/{expected}")
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    q = dict(cell=list(box), rows=len(df), bytes_http=len(raw), sha256=digest(path),
             elapsed_seconds=monotonic()-t)
    save_json(receipt, q)
    return [q]


def run(root, workers):
    root.mkdir(exist_ok=True, parents=True)
    cache = access.Cache(root / "counts")
    count = cache.query("nsc", "SELECT count(*) AS n FROM nsc_dr2.chip", expected=1)
    valid = "ra >= 0 AND ra < 360 AND dec >= -90 AND dec <= 90"
    hist = cache.query("nsc", f"SELECT CASE WHEN {valid} THEN floor(ra/15) ELSE -1 END AS a,"
                       f"CASE WHEN {valid} THEN least(17,floor((dec+90)/10)) ELSE -1 END AS d,"
                       "count(*) AS n FROM nsc_dr2.chip GROUP BY 1,2")
    total = int(count.n.iat[0])
    if hist.isna().any().any() or int(hist.n.sum()) != total:
        raise ValueError("Unaccounted all-sky CCD coordinates")
    cells = {(int(v.a),int(v.d)):int(v.n) for v in hist.itertuples()}
    exception_count = cells.pop((-1,-1), 0)
    exceptions = root / "private_noncanonical_centres.parquet"
    if exception_count:
        # Retain real vertices/keys for boundary geometry and invalid-sentinel audit.
        df = cache.query("nsc", f"SELECT {COLUMNS} FROM nsc_dr2.chip WHERE ({valid}) IS NOT TRUE",
                         expected=exception_count)
        if df.duplicated(["exposure","ccdnum"]).any():
            raise ValueError("Noncanonical CCD keys are duplicated")
        df.to_parquet(exceptions, index=False, compression="zstd")
    if any(not (0<=a<24 and 0<=d<18) for a,d in cells):
        raise ValueError("Unexpected catalogue sky coordinate outside declared domain")
    jobs = [((15*a,15*(a+1),10*d-90,10*d-80),cells.get((a,d),0))
            for a in range(24) for d in range(18)]
    # Larger cells first; only metadata is opened, no motion results.
    jobs.sort(key=lambda item:-item[1])
    def fetch(job):
        return part(root,*job)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = [p for result in pool.map(fetch,jobs) for p in result]
    if sum(p["rows"] for p in parts) + exception_count != total:
        raise ValueError("All-sky CCD counts do not close")
    save_json(root / "private_chip_manifest.json", dict(parts=parts, counted=jobs, total=total,
              noncanonical_centres=exception_count, noncanonical_sha256=digest(exceptions) if exception_count else None))
    print("Verified CCD rows", total, "shards", len(parts), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--workers",type=int,default=4)
    a = p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):
        p.error("Run inside assigned Slurm CPUs")
    run(a.root,a.workers)
