"""Development-only conflict-budget comparison using frozen upstream candidates."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from .__main__ import detection_arrays, tomllib
from .associate import Track, choose_tracks
from .io import save_json
from .validation import evaluate, gaia_labels, groups_by_label


def run(work, region, config, max_component, seconds):
    spec = config["regions"][region]
    if spec["role"] != "development":
        raise ValueError("Conflict ablation uses only the development region")
    center = (spec["ra"], spec["dec"])
    data, xy, time, sigma = detection_arrays(work, center, config)
    candidates = [Track(tuple(row["indices"]), row["score"], row["status"], row["groups"]) for row in
                  json.loads((work / "candidates.json").read_text())]
    start = monotonic()
    selected, difficult, used = choose_tracks(candidates, max_component, seconds)
    elapsed = monotonic() - start
    labels, gaia = gaia_labels(data, xy, time, sigma, pd.read_csv(work / "validation" / "gaia.csv"),
                              center, config)
    truth = groups_by_label(labels, data, time, gaia, center, spec["radius"], config)
    score = evaluate([vars(track) for track in selected], truth, labels, data,
                     np.hypot(gaia.pmra, gaia.pmdec), config)
    save_json(work / "solver_ablation_summary.json", dict(
        max_component=max_component, time_limit_seconds=seconds, seconds=elapsed,
        unresolved_components=difficult, assigned_detections=len(used), evaluation=score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", default="smoke")
    parser.add_argument("--config", default="configs/experiment-a.toml")
    parser.add_argument("--component", type=int, default=3000)
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()
    with open(args.config, "rb") as file:
        config = tomllib.load(file)
    run(args.work, args.region, config, args.component, args.seconds)
