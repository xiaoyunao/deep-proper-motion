"""Compare seed thinning with all detection pairs in a small real region."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from .__main__ import detection_arrays, tomllib
from .associate import motion_tracks, static_tracks
from .io import save_json
from .validation import evaluate, gaia_labels, groups_by_label


def run(work, region, config):
    spec = config["regions"][region]
    center = (spec["ra"], spec["dec"])
    data, xy, time, sigma = detection_arrays(work, center, config)
    exposure = data.exposure_id.to_numpy()
    fixed = static_tracks(xy, exposure, config["fixed_radius_mas"])
    start = monotonic()
    full = dict(config, season_seed_representatives=False)
    tracks, _, _, stats = motion_tracks(xy, time, sigma, exposure, fixed, full)
    stats["seconds"] = monotonic() - start
    labels, gaia = gaia_labels(data, xy, time, sigma, pd.read_csv(work / "validation" / "gaia.csv"),
                              center, config)
    truth = groups_by_label(labels, data, time, gaia, center, spec["radius"], config)
    stats["evaluation"] = evaluate([vars(track) for track in tracks], truth, labels, data,
                                   np.hypot(gaia.pmra, gaia.pmdec), config)
    stats["thinned_search"] = json.loads((work / "association_resources.json").read_text())
    save_json(work / "exhaustive_summary.json", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", default="north")
    parser.add_argument("--config", default="configs/experiment-a.toml")
    args = parser.parse_args()
    with open(args.config, "rb") as file:
        config = tomllib.load(file)
    run(args.work, args.region, config)
