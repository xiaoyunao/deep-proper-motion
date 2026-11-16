"""Aggregate independent v0.2 regions with field-block uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np

from .io import save_json
from .validation import ratio


def block_fraction(rows, seed=45116):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
    rows = rows[rows[:, 1] > 0]
    n, total = rows.sum(axis=0)
    if len(rows) < 3 or total == 0:
        return dict(n=int(n), denominator=int(total), value=n / total if total else None,
                    field_blocks=len(rows), field_bootstrap95=None)
    random = np.random.default_rng(seed)
    draws = rows[random.integers(0, len(rows), size=(3000, len(rows)))].sum(axis=1)
    valid = draws[:, 1] > 0
    interval = np.quantile(draws[valid, 0] / draws[valid, 1], [.025, .975])
    return dict(n=int(n), denominator=int(total), value=n / total,
                field_blocks=len(rows), field_bootstrap95=interval)


def optical_summary(fields):
    regions = [(row["region"], row["role"]) for row in fields]
    if len(set(regions)) != len(regions):
        raise ValueError("Each region can be included once per cohort")
    by_role = {}
    for region in fields:
        by_role.setdefault(region["role"], []).append(region)
    results = {}
    for role, members in by_role.items():
        versions = {(row["receipt"]["commit"], row["receipt"]["config_sha256"]) for row in members}
        if len(versions) != 1:
            raise ValueError(f"Mixed code or configuration in {role}")
        cohort = [row["validation"] for row in members]
        role_out = {}
        for name in ("fixed", "nsc_like", "motion"):
            selections = [entry["methods"][name] for entry in cohort]
            entry = {}
            for metric in ("candidate_recall", "selected_recall", "secure_recall",
                           "ambiguous_recall", "unresolved_fraction"):
                proportions = [row[metric] for row in selections if row[metric] is not None]
                entry[metric] = block_fraction([(p["n"], p["denominator"]) for p in proportions])
            speed_bins = {}
            for row in selections:
                for item in row["pm_bins"]:
                    key = (item["low"], item["high"])
                    speed_bins.setdefault(key, []).append(item)
            entry["proper_motion_bins"] = []
            for (low, high), items in sorted(speed_bins.items()):
                if len(items) != len(selections):
                    raise ValueError(f"Inconsistent proper-motion bins in {role}")
                measurements = {"selected": items}
                for key in ("candidate", "secure"):
                    measurements[key] = [item[key] for item in items if item[key] is not None]
                entry["proper_motion_bins"].append(dict(
                    low=low, high=high, **{key: block_fraction(
                        [(item["n"], item["denominator"]) for item in values])
                        for key, values in measurements.items()}))
            confirmation = [row["object_confirmation"] for row in selections]
            known_wrong = sum(c["discordant"] for c in confirmation)
            known_right = sum(c["confirmed"] for c in confirmation)
            unknown = sum(c["unknown"] for c in confirmation)
            objects = known_wrong + known_right + unknown
            entry["labelled_contamination"] = block_fraction(
                [(p["discordant"], p["discordant"] + p["confirmed"]) for p in confirmation])
            entry["unknown_objects"] = ratio(unknown, objects)
            entry["contamination_identification_bounds"] = (
                [known_wrong / objects, (known_wrong + unknown) / objects] if objects else None)
            role_out[name] = entry
        results[role] = role_out
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args()
    save_json(args.output, optical_summary([json.loads(p.read_text()) for p in args.summaries]))
