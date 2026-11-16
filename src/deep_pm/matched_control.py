"""Compare blind fitted motions with fixed-association fits on the same sources."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import detection_arrays, tomllib
from .io import digest, save_json
from .provenance import run_stage
from .public_summary import verify_stage_hashes
from .validation import gaia_labels, groups_by_label, matched_object, residual_stats, truth_requirements


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


def conflict_review(fits, tracks, candidates, components, labels, required, fraction):
    recovered = {matched_object(np.asarray(tracks["motion"][row["track"]]["indices"]), labels,
                                required, fraction) for row in fits
                 if row["method"] == "motion" and row["in_core"]}
    missed = set(required) - recovered
    owner = {}
    in_optimal = []
    for component in components:
        for j in component["selected"]:
            owner.update((k, j) for k in candidates[j]["indices"])
        if component["state"] == "optimal":
            in_optimal.extend(component["candidates"])
    choices = defaultdict(list)
    for j in in_optimal:
        candidate = candidates[j]
        idx = np.asarray(candidate["indices"], dtype=int)
        target = matched_object(idx, labels, required, fraction)
        if target not in missed:
            continue
        blockers = {owner[i] for i in candidate["indices"] if i in owner}
        delta = candidate["score"] - sum(candidates[i]["score"] for i in blockers)
        choices[target].append((delta, j, blockers))
    categories = Counter()
    changes = []
    for target, options in choices.items():
        delta, j, blockers = max(options, key=lambda x: (x[0], candidates[x[1]]["score"]))
        changes.append(delta)
        if delta > 1e-6:
            categories["positive_local_replacement_delta"] += 1
        if len(blockers) > 1:
            categories["multiple_blocking_tracks"] += 1
        if j in blockers:
            categories["selected_candidate_outside_core_or_missing_fit"] += 1
            blockers.remove(j)
        kinds = set()
        for i in blockers:
            visits = labels[candidates[i]["indices"]]
            known = set(visits[visits >= 0])
            if not known:
                kinds.add("blocker_without_gaia_labels")
            elif known == {target}:
                kinds.add("blocker_labels_only_target")
            else:
                kinds.add("blocker_contains_other_gaia_source")
        for kind in kinds:
            categories[kind] += 1
    return dict(eligible=len(required), missed=len(missed), missed_with_optimal_candidate=len(choices),
                replacement_delta_quantiles=np.quantile(changes, [0, .5, 1]) if changes else None,
                conflict_counts=dict(categories))


def run(source, work, region, review_conflicts=False):
    upstream = json.loads((source / "run_manifest.json").read_bytes())
    verify_stage_hashes(upstream["stages"])
    paths = {"tracks.json": ("associate", "outputs"), "fits.json": ("fit", "outputs"),
             "detections.csv": ("associate", "inputs"), "validation/gaia.csv": ("validate", "inputs"),
             "validation/paired.csv": ("validate", "outputs")}
    if review_conflicts:
        paths.update({"candidates.json": ("associate", "outputs"),
                      "association_audit.json": ("associate", "outputs")})
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
    config.update(spec.get("overrides", {}))
    center = spec["ra"], spec["dec"]
    stage = "conflict-review" if review_conflicts else "matched-control"
    output = "conflict_review_summary.json" if review_conflicts else "matched_control_summary.json"
    with run_stage(work, config_path, region, stage, list(paths), [output]) as execute:
        if execute:
            data, xy, t, sigma = detection_arrays(work, center, config)
            labels, gaia = gaia_labels(data, xy, t, sigma,
                                      pd.read_csv(work / "validation" / "gaia.csv"), center, config)
            truth = groups_by_label(labels, data, t, gaia, center, spec["radius"], config)
            required = truth_requirements(truth, data.survey.to_numpy(), t, config["min_season_days"])
            fits = json.loads((work / "fits.json").read_bytes())
            tracks = json.loads((work / "tracks.json").read_bytes())
            if review_conflicts:
                candidates = json.loads((work / "candidates.json").read_bytes())
                audit = json.loads((work / "association_audit.json").read_bytes())
                summary = conflict_review(fits, tracks, candidates, audit["components"],
                                          labels, required, config["recovery_fraction"])
                save_json(work / output, dict(upstream_commit=upstream["commit"], **summary))
                return
            control = pd.read_csv(work / "validation" / "paired.csv").query("fit == 'all_sdss'")
            control = control.set_index("source", verify_integrity=True)
            outcomes = matched_motion(fits, tracks, labels, required, config["recovery_fraction"],
                                      gaia, control)
            save_json(work / output, dict(
                upstream_commit=upstream["commit"], eligible=len(truth),
                fixed_control_total=len(control), matched_pm=outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", required=True)
    parser.add_argument("--review-conflicts", action="store_true")
    args = parser.parse_args()
    run(args.source, args.work, args.region, args.review_conflicts)
