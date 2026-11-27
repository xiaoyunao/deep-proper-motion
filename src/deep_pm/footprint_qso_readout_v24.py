"""Conditional scientific quality of the closed, blind catalogue on common spectra."""

import argparse
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from .full_coordinate_quality import ellipse_coverage
from .io import digest, save_json
from .meeting_stats import metrics
from .stripe82_comparison_v24 import binomial

METHODS = ("ours", "xiao", "nsc")
EDGES = (16, 18, 19, 20, 21, 21.5, 22, 22.5, 23, 24)
NSIDE = 32
DISPLAY_MIN = 30


def paired(frame, methods, *, detailed=True):
    mask = frame[[method + "_pm_e" for method in methods]].notna().all(axis=1)
    selected = frame.loc[mask]
    n = len(selected)
    result = dict(parent=len(frame), common=n)
    if n < 5:
        return result
    result["catalogues"] = {}
    for method in methods:
        v = selected[[method + "_pm_e", method + "_pm_n"]].to_numpy(float)
        se = selected[[method + "_se_e", method + "_se_n"]].to_numpy(float)
        if not np.isfinite(v).all() or not np.isfinite(se).all() or np.any(se <= 0):
            raise ValueError("Invalid accepted external or scientific uncertainty")
        out = metrics(v)
        radius = np.linalg.norm(v, axis=1)
        out["tail_gt20"] = binomial(int((radius > 20).sum()), n)
        if detailed:
            out["tail_gt5"] = binomial(int((radius > 5).sum()), n)
            zero = np.median(v, axis=0)
            out["raw_1sigma"] = [binomial(k, n) for k in (abs(v) <= se).sum(axis=0)]
            out["centered_1sigma"] = [binomial(k, n) for k in (abs(v - zero) <= se).sum(axis=0)]
            out["raw_2sigma"] = [binomial(k, n) for k in (abs(v) <= 2 * se).sum(axis=0)]
            out["formal_se_median"] = np.median(se, axis=0)
        result["catalogues"][method] = out
    ours = result["catalogues"]["ours"]["p68"]
    result["improvement_percent"] = {
        m: (
            float(100 * (1 - ours / result["catalogues"][m]["p68"]))
            if result["catalogues"][m]["p68"] > 0
            else None
        )
        for m in methods
        if m != "ours"
    }
    if detailed:
        sigma = selected[["ours_se_e", "ours_se_n"]].to_numpy()
        cov = np.empty((n, 2, 2))
        cov[:, 0, 0], cov[:, 1, 1] = sigma.T**2
        cov[:, 1, 0] = cov[:, 0, 1] = selected.pm_cov_en
        if np.any(np.linalg.det(cov) <= 0):
            raise ValueError("Nonpositive PM covariance")
        result["our_error_ellipse"] = ellipse_coverage(selected[["ours_pm_e", "ours_pm_n"]], cov)
        result["median_component_SE"] = {
            label: np.median(
                np.sqrt(selected[[label + "_pm_variance_e", label + "_pm_variance_n"]]), axis=0
            )
            for label in ("independent", "reference")
        }
    return result


def condition_groups(frame):
    limits = {
        "sdss_runs": (1, 2, 5, np.inf),
        "sdss_years": (1, 2, 3, np.inf),
        "nsc_years": (1, 2, 4, np.inf),
        "joint_span": (0, 10, 15, 20, np.inf),
        "nsc_span": (0, 2, 4, 6, np.inf),
        "nsc_nights": (0, 5, 10, 30, np.inf),
        "nsc_seasons": (0, 3, 6, np.inf),
        "maximum_group_leverage": (0, 0.5, 0.8, 1.0000001),
    }
    for key, edges in limits.items():
        for lo, hi in zip(edges, edges[1:]):
            yield f"{key} {lo:g}–{hi:g}", frame.loc[frame[key].ge(lo) & frame[key].lt(hi)]
    for key in ("kind", "component", "instruments", "bands"):
        for name, group in frame.groupby(key, sort=True):
            yield f"{key} {name}", group


def run(directory, destination):
    data = directory / "private_final_spectral_pairing.parquet"
    proof = directory / "v24_spectral_pairing_summary.json"
    rec = json.loads(proof.read_text())
    if digest(data) != rec["private_pair_sha256"]:
        raise ValueError("Changed immutable QSO pairs")
    frame = pd.read_parquet(data)
    if frame.q.duplicated().any() or len(frame) != rec["nominal_spectroscopic_parent"]:
        raise ValueError("Incomplete or nonunique nominal spectral sample")
    result = dict(
        closure_sha256=rec["closure_sha256"],
        pairing_sha256=digest(proof),
        map_definition=dict(healpix_nside=NSIDE, ordering="RING", public_min=5, display_min=DISPLAY_MIN),
        strict=paired(frame, METHODS),
        fine_magnitude={},
        faint={},
        physical_r={},
        meaning="Absolute PM relative to zero on an identical set of spectra in each comparison. "
        "Strict match rates are conditional on the union spectroscopic parent. "
        "Error-coverages include the fitted reference term. Centering is diagnostic only. "
        "Wilson intervals are conditional object-count intervals.",
    )
    if result["strict"]["common"] != rec["strict_three_catalogue"]:
        raise ValueError("Strict spectral denominator changed")
    for lo, hi in zip(EDGES, EDGES[1:]):
        sample = frame.loc[frame.reference_r.ge(lo) & frame.reference_r.lt(hi)]
        name = f"{lo:g}-{hi:g}"
        result["fine_magnitude"][name] = paired(sample, METHODS)
        result["physical_r"][name] = paired(sample, ("ours", "r"))
    for lo in (21, 22):
        sample = frame.loc[frame.reference_r.ge(lo) & frame.reference_r.lt(lo + 1)].copy()
        sample["map_pixel"] = hp.ang2pix(NSIDE, sample.ra, sample.dec, lonlat=True)
        cells = []
        for pixel, group in sample.groupby("map_pixel", sort=True):
            record = paired(group, METHODS, detailed=False)
            if record["common"] < DISPLAY_MIN:
                record = {k: record[k] for k in ("parent", "common")}
            cells.append(dict(pixel=int(pixel), **record))
        result["faint"][str(lo)] = dict(
            total=paired(sample, METHODS),
            physical_r_pair=paired(sample, ("ours", "r")),
            four_way=paired(sample, (*METHODS, "r")),
            map=cells,
            conditions={k: paired(g, METHODS) for k, g in condition_groups(sample)},
        )
    destination.parent.mkdir(exist_ok=True, parents=True)
    save_json(destination, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("Run in a Slurm allocation")
    run(args.directory, args.destination)
