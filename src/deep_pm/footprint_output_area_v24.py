"""Integrate released support and productive ownership against actual field/CCD polygons."""

import argparse
import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from .continuous_reference import verify_directory
from .footprint_manifest_v24 import layers
from .io import digest, save_json


def owner_grid(boxes, bins):
    result = np.zeros((180 * bins, 360 * bins), dtype=np.uint8)
    for spec, state in boxes:
        w, e = spec["ra"]
        south, north = spec["dec"]
        # Explicit half-open, sampled at the same centres as native WCS integration.
        ra_idx = np.arange(int(np.floor(w * bins)), int(np.ceil(e * bins)))
        dec_idx = np.arange(
            max(0, int(np.floor((south + 90) * bins))),
            min(180 * bins, int(np.ceil((north + 90) * bins))),
        )
        lon = (ra_idx + 0.5) / bins
        lat = (dec_idx + 0.5) / bins - 90
        ra_idx = ra_idx[(lon >= w) & (lon < e)] % (360 * bins)
        dec_idx = dec_idx[(lat >= south) & (lat < north)]
        ix = np.ix_(dec_idx, ra_idx)
        if np.any(result[ix]):
            raise ValueError("Overlapping common-epoch owners at a geometric sampling point")
        result[ix] = state
    return result


def normalized_counts(source):
    f = source / "acquisition_summary.json"
    verify_directory(source, (f.name,))
    r = json.loads(f.read_text())
    # The archive sdss_measures counter precedes normalization and run de-duplication.
    early = r.get("sdss_r_measurements_preserved")
    total = r.get("normalized_detections")
    fallback = total is None or early is None
    if fallback:
        # Complement acquisitions may use different receipt schemas. Obtain
        # actual survey counts only for those cores, from their signed table.
        import pandas as pd

        verify_directory(source, ("detections.csv",))
        early = total = 0
        for chunk in pd.read_csv(source / "detections.csv", usecols=["survey"], chunksize=200000):
            total += len(chunk)
            early += int(chunk.survey.eq("SDSS").sum())
    if not 0 <= early <= total:
        raise ValueError("Invalid signed detection breakdown")
    return early, total, fallback


def run(base, bins, output):
    work = base / "v2.4"
    root = work / "joint"
    reg = root / "registration/private_full_coordinate_plan.json"
    entries = json.loads(reg.read_text())["cores"]
    proof = root / "catalogue/v22_catalogue_ownership_summary.json"
    closure = json.loads(proof.read_text())
    binding = root / "catalogue/private_closure_binding.json"
    if digest(binding) != closure["private_input_binding_sha256"] or json.loads(binding.read_text())[
        "plan_sha256"
    ] != digest(reg):
        raise ValueError("Geometric and catalogue registrations disagree")
    con = duckdb.connect(str(root / "catalogue/catalogue.duckdb"), read_only=True)
    counts = dict(con.execute("SELECT owner,count(*) FROM accepted GROUP BY owner").fetchall())
    con.close()
    boxes = []
    raw_early = raw_late = 0
    fallback = 0
    for entry in entries:
        source = Path(entry["source"])
        early, total, counted = normalized_counts(source)
        fallback += int(counted)
        raw_early += early
        raw_late += total - early
        n = counts.get(entry["core"], 0)
        # A fitted source can move into a neighbouring J2015.5 owner. Resolve
        # delivered input and catalogue presence as independent bits.
        state = 1 + int(0 < early < total) + 2 * int(n > 0)
        boxes.append((entry["spec"], state))
    grid = owner_grid(boxes, bins)
    e0, e1, _, _, _ = layers(work / "geometry", "early", bins)
    n0, n1, _, _, _ = layers(work / "geometry", "late", bins)
    weights = np.rad2deg(np.diff(np.sin(np.deg2rad(np.arange(180 * bins + 1) / bins - 90)))) / bins
    stage = np.zeros((4, 5))
    for i, w in enumerate(weights):
        survey = np.isfinite(e0[i]) & np.isfinite(n0[i])
        time = (
            survey
            & (n1[i] - n0[i] >= 0.6 * 365.25)
            & (np.maximum(e1[i], n1[i]) - np.minimum(e0[i], n0[i]) >= 5 * 365.25)
        )
        for j, mask in enumerate((survey, time)):
            stage[j] += w * np.bincount(grid[i, mask], minlength=5)
    del e0, e1, n0, n1
    points = root / "native-area" / f"private_supported_points_{bins}.parquet"
    native = root / "native-area" / f"v24_native_reference_area_{bins}_summary.json"
    r = json.loads(native.read_text())
    if digest(points) != r["private_points_sha256"]:
        raise ValueError("Changed native support grid")
    for batch in pq.ParquetFile(points).iter_batches(batch_size=250000):
        pid = batch.column(0).to_numpy()
        ok = batch.column(1).to_numpy(zero_copy_only=False)
        tile, sub = np.divmod(pid, bins**2)
        y = tile // 360 * bins + sub // bins
        x = tile % 360 * bins + sub % bins
        state = grid[y, x]
        stage[2] += np.bincount(state, weights=weights[y], minlength=5)
        stage[3] += np.bincount(state[ok], weights=weights[y[ok]], minlength=5)
    np.testing.assert_allclose(
        stage[2:4].sum(axis=1), [r["reference_overlap_sqdeg"], r["reference_and_time_sqdeg"]], rtol=2e-10
    )
    if sum(counts.values()) != closure["retained_common_epoch_solutions"]:
        raise ValueError("Owner source counts do not close")
    names = (
        "unregistered",
        "signed_without_both_surveys_or_output",
        "both_surveys",
        "productive_without_both_local_surveys",
        "both_surveys_and_productive",
    )
    save_json(
        output,
        dict(
            closure_sha256=digest(proof),
            native_area_sha256=digest(native),
            resolution_deg=1 / bins,
            accepted=sum(counts.values()),
            retained_owner_count=len(counts),
            sdss_normalized_with_halo=raw_early,
            nsc_normalized_with_halo=raw_late,
            signed_csv_fallback_cores=fallback,
            square_degrees={
                s: dict(zip(names, v))
                for s, v in zip(
                    ("geometry", "time", "common_reference", "time_and_reference"), stage[:4]
                )
            },
            delivered_geometry_sqdeg=stage[0, 2] + stage[0, 4],
            productive_supported_area_sqdeg=stage[3, 3] + stage[3, 4],
            scope="Actual sampled polygon intersection in J2015.5 owner rectangles. Delivered requires both surveys in that signed core input including halo. "
            "Productive means native common physical reference/time geometry within an owner with at least one globally accepted source. "
            "Sub-cell variations, missing detections and stellar selection remain; source counts do not define polygon area.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--bins", type=int, choices=(20, 40), default=40)
    a = p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        p.error("Run in a Slurm allocation")
    run(a.base, a.bins, a.output)
