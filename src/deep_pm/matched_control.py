"""Compare blind fitted motions with fixed-association fits on the same sources."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import detection_arrays, tomllib
from .io import digest, save_json
from .provenance import run_stage
from .public_summary import verify_stage_hashes
from .validation import gaia_labels, groups_by_label, matched_object, residual_stats, truth_requirements


INPUTS = ("detections.csv", "tracks.json", "fits.json", "validation/gaia.csv", "validation/paired.csv")


def matched_motion(fits, tracks, label, required, fraction, gaia, control):
    blind_components = ["pm_x", "pm_y"]
    truth_components = ["pm_tangent_x", "pm_tangent_y"]
    outcomes = {}
    for method in ("fixed", "nsc_like", "motion"):
        groups = {"all_selected": {}}
        if method == "motion":
            groups["secure"] = {}
        for row in fits:
            if row["method"] != method or not row["in_core"]:
                continue
            track = tracks[method][row["track"]]
            key = matched_object(np.asarray(track["indices"]), label, required, fraction)
            if key is None or key not in control.index:
                continue
            truth = gaia.loc[key, truth_components].to_numpy(dtype=float)
            pm = np.array([row[k] for k in blind_components])
            groups["all_selected"][key] = pm - truth
            if method == "motion" and track["status"] == "secure":
                groups["secure"][key] = pm - truth
        for name, values in groups.items():
            blind = np.array(list(values.values())).reshape(-1, 2)
            fixed = control.loc[list(values), ["east", "north"]].to_numpy()
            outcomes[f"{method}/{name}"] = {
                "blind": residual_stats(blind), "fixed_all_visits": residual_stats(fixed),
                "blind_minus_fixed": residual_stats(blind - fixed),
            }
    return outcomes


def run(source, work, region):
    upstream = json.loads((source / "run_manifest.json").read_bytes())
    verify_stage_hashes(upstream["stages"])
    paths = {"tracks.json": ("associate", "outputs"), "fits.json": ("fit", "outputs"),
             "detections.csv": ("associate", "inputs"), "validation/gaia.csv": ("validate", "inputs"),
             "validation/paired.csv": ("validate", "outputs")}
    if upstream["region"] != region:
        raise ValueError("Requested field and source manifest differ")
    for relative, (stage, key) in paths.items():
        attempt = upstream["stages"][stage][-1]
        target = source / relative
        if attempt["state"] != "completed" or attempt[key][relative] != digest(target):
            raise ValueError(f"Original run did not verify {relative}")
        link = work / relative
        link.parent.mkdir(exist_ok=True, parents=True)
        if not link.exists():
            link.symlink_to(target.resolve())
        elif link.resolve() != target.resolve():
            raise ValueError(f"Different source already staged for {relative}")
    config_path = source / "config_snapshot.toml"
    if digest(config_path) != upstream["config_sha256"]:
        raise ValueError("Original configuration differs from its receipt")
    with config_path.open("rb") as fh:
        config = tomllib.load(fh)
    spec = config["regions"][region]
    center = spec["ra"], spec["dec"]
    with run_stage(work, config_path, region, "matched-control", INPUTS,
                   ["matched_control_summary.json"]) as execute:
        if execute:
            data, xy, t, sigma = detection_arrays(work, center, config)
            labels, gaia = gaia_labels(data, xy, t, sigma,
                                      pd.read_csv(work / "validation" / "gaia.csv"), center, config)
            truth = groups_by_label(labels, data, t, gaia, center, spec["radius"], config)
            required = truth_requirements(truth, data.survey.to_numpy(), t, config["min_season_days"])
            control = pd.read_csv(work / "validation" / "paired.csv").query("fit == 'all_sdss'")
            control = control.set_index("source", verify_integrity=True)
            fits = json.loads((work / "fits.json").read_bytes())
            tracks = json.loads((work / "tracks.json").read_bytes())
            outcomes = matched_motion(fits, tracks, labels, required, config["recovery_fraction"],
                                      gaia, control)
            save_json(work / "matched_control_summary.json", dict(
                upstream_commit=upstream["commit"], eligible=len(truth),
                fixed_control_total=len(control), matched_pm=outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", required=True)
    args = parser.parse_args()
    run(args.source, args.work, args.region)
