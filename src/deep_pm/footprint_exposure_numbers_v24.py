"""Freeze ambiguous NSC exposure-number namespaces against the full DR2 inventory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import pandas as pd

from .access import Cache
from .io import digest, save_json


def freeze(work, workers):
    root = work / "assets/exposures"
    public = work / "registration"
    old_path = root / "private_exposures.parquet"
    signed = json.loads((root / "v24_exposure_inventory_summary.json").read_text())
    if digest(old_path) != signed["sha256"]:
        raise ValueError("Frozen physical exposure inventory changed")
    old = pd.read_parquet(old_path, columns=["ra", "exposure", "instrument"])
    cache = Cache(root / "number-query")

    def fetch(low):
        expected = old.loc[old.ra.ge(low) & old.ra.lt(low + 10), ["exposure", "instrument"]]
        sql = f"SELECT exposure,instrument,expnum FROM nsc_dr2.exposure WHERE ra>={low} AND ra<{low+10}"
        part = cache.query("nsc", sql, expected=len(expected))
        match = expected.merge(part, on=["instrument", "exposure"], validate="one_to_one")
        if len(match) != len(expected):
            raise ValueError("Physical exposure set changed in the archive")
        return match

    with ThreadPoolExecutor(max_workers=workers) as pool:
        table = pd.concat(pool.map(fetch, range(0, 360, 10)), ignore_index=True)
    if len(table) != signed["rows"] or table.exposure.hasnans or not table.exposure.is_unique:
        raise ValueError("Exposure numbers do not cover the signed all-sky inventory")
    if table.expnum.hasnans or not table.expnum.eq(table.expnum.astype("int64")).all():
        raise ValueError("Cannot verify an integral exposure number for every exposure")
    ambiguous = table.loc[table.duplicated(["instrument", "expnum"], keep=False)]
    namespace = ambiguous.instrument + "." + ambiguous.expnum.astype("int64").astype(str)
    mapping = dict(sorted(zip(ambiguous.exposure, namespace)))
    rule = dict(method="qualify-reused-exposure-number-v1", inventory_sha256=signed["sha256"],
                exposures=mapping)
    dest = root / "private_exposure_identity_rule.json"
    if dest.exists() and json.loads(dest.read_text()) != rule:
        raise ValueError("Existing frozen exposure identity rule changed")
    if not dest.exists():
        save_json(dest, rule)
    save_json(public / "v24_exposure_identity_rule_summary.json",
              dict(method=rule["method"], exposures=len(table), instrument_counts=table.instrument.value_counts().to_dict(),
                   ambiguous_exposures=len(mapping), ambiguous_namespaces=namespace.nunique(),
                   affected_by_instrument=ambiguous.instrument.value_counts().to_dict(),
                   inventory_sha256=signed["sha256"], rule_sha256=digest(dest),
                   slurm_job=os.environ["SLURM_JOB_ID"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1 <= args.workers <= int(os.environ["SLURM_CPUS_PER_TASK"]):
        parser.error("Use the assigned Slurm CPUs")
    freeze(args.work, args.workers)
