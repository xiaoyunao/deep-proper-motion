"""Verify the counted object-index route on a signed complete all-band join page."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import pandas as pd

from .access import Cache, MEAS, fetch_measurements_for_objects
from .nsc_measurement_identity import measurement_ids
from .io import digest, save_json


def compare(work, core, output):
    cache = Cache(work / "intake" / core / "cache")
    columns = MEAS.split(",")
    for meta in sorted(cache.path.glob("*.json")):
        page = json.loads(meta.read_text())
        if (" FROM nsc_dr2.meas m JOIN nsc_dr2.object o " not in page["query"]
                or not page["query"].startswith("SELECT m.measid,") or page["rows"] < 32):
            continue
        raw = meta.with_suffix(".csv")
        if digest(raw) != page["sha256"]:
            raise ValueError("Changed reference page")
        reference = pd.read_csv(raw, usecols=columns)
        if len(reference) != page["rows"] or measurement_ids(reference).duplicated().any():
            raise ValueError("Invalid complete reference page")
        population = reference.objectid.astype(str).unique()
        ids = pd.Series(sorted(population, key=lambda x: sha256(x.encode()).digest())[:32])
        expected = reference.loc[reference.objectid.astype(str).isin(ids), columns].sort_values(["measid", "exposure", "ccdnum"])
        actual = fetch_measurements_for_objects(cache, ids, batch_size=120)[columns].sort_values(["measid", "exposure", "ccdnum"])
        pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
        save_json(output, dict(slurm_job=os.environ["SLURM_JOB_ID"], page_sha256=page["sha256"],
                               parent_rows=len(reference), tested_objects=len(ids), tested_measurements=len(actual),
                               compared_columns=columns, all_values_equal=True,
                               scope="Fixed hash sample of complete object identities from one signed join page."))
        return
    raise ValueError("No complete signed measurement page for independent testing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("core")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or os.getenv("DPM_NSC_TRANSPORT") != "counted-tap":
        parser.error("Run with counted TAP in a Slurm allocation")
    compare(args.work, args.core, args.output)
