"""Sensitivity of PM uncertainty to resampling whole observation nights."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import detection_arrays, tomllib
from .astrometry import fit_motion
from .io import save_json
from .validation import gaia_labels, groups_by_label


def pm_uncertainty(indices, time, xy, sigma, data, epoch, rng):
    night = np.floor(data.observation_time.to_numpy()[indices]).astype(int)
    groups = [indices[night == value] for value in np.unique(night)]
    fitted = fit_motion(xy[indices], time[indices], sigma[indices], epoch)
    trials = []
    for _ in range(80):
        draw = rng.integers(len(groups), size=len(groups))
        chosen = np.concatenate([groups[i] for i in draw])
        if np.ptp(time[chosen]) >= .6:
            trials.append(fit_motion(xy[chosen], time[chosen], sigma[chosen], epoch, robust=False).coeff[1])
    formal = np.sqrt(fitted.cov[:, 1, 1])
    return formal, np.std(trials, axis=0, ddof=1) if len(trials) > 1 else np.full(2, np.nan)


def run(work, region, config):
    settings = config["regions"][region]
    center = (settings["ra"], settings["dec"])
    data, xy, t, sigma = detection_arrays(work, center, config)
    gaia = pd.read_csv(work / "validation" / "gaia.csv")
    labels, gaia = gaia_labels(data, xy, t, sigma, gaia, center, config)
    truth = groups_by_label(labels, data, t, gaia, center, settings["radius"], config)
    result = {}
    rng = np.random.default_rng(config["seed"])
    for i in truth.values():
        nsc = i[data.survey.iloc[i].eq("NSC").to_numpy()]
        sdss = i[data.survey.iloc[i].eq("SDSS").to_numpy()]
        if len(nsc) < 2 or np.ptp(t[nsc]) < config["min_seed_years"]:
            continue
        early = sdss[np.argmin(t[sdss])]
        for key, rows in [("nsc_only", nsc), ("one_sdss", np.r_[nsc, early]), ("all_sdss", i)]:
            formal, clustered = pm_uncertainty(rows, t, xy, sigma, data, config["reference_epoch"], rng)
            if np.isfinite(clustered).all():
                result.setdefault(key, []).append(clustered / formal)
    save_json(work / "night_bootstrap_summary.json", {
        key: dict(n_sources=len(value), ratio_night_to_formal=np.median(value, axis=0))
        for key, value in result.items()
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", default="smoke")
    parser.add_argument("--config", default="configs/experiment-a.toml")
    args = parser.parse_args()
    with open(args.config, "rb") as file:
        config = tomllib.load(file)
    run(args.work, args.region, config)
