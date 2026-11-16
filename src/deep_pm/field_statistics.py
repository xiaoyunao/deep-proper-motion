"""Aggregate independent v0.2 regions with field-block uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np

from .io import save_json
from .validation import ratio


def block_fraction(rows, seed=45116):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
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
        by_role.setdefault(region["role"], []).append(region["validation"])
    results = {}
    for role, cohort in by_role.items():
        role_out = {}
        for name in ("fixed", "nsc_like", "motion"):
            selections = [entry["methods"][name] for entry in cohort]
            entry = {}
            for metric in ("candidate_recall", "selected_recall", "secure_recall",
                           "ambiguous_recall", "unresolved_fraction"):
                proportions = [row[metric] for row in selections if row[metric] is not None]
                entry[metric] = block_fraction([(p["n"], p["denominator"]) for p in proportions])
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
