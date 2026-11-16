"""Independent annual prediction and translated-catalogue control."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import detection_arrays, tomllib
from .associate import motion_tracks, static_tracks
from .astrometry import fit_motion
from .io import save_json
from .validation import gaia_labels, ratio


def associations(xy, t, sigma, exposures, config):
    fixed = static_tracks(xy, exposures, config["fixed_radius_mas"])
    nsc = static_tracks(xy, exposures, config["fixed_radius_mas"], config["dbscan_eps_mas"])
    selected, _, _, _ = motion_tracks(xy, t, sigma, exposures, fixed, config)
    return {"fixed": fixed, "nsc_like": nsc, "motion": [track.indices for track in selected]}


def held_year(data, xy, t, sigma, labels, radius, config):
    year = np.floor(t).astype(int)
    nsc = data.survey.eq("NSC").to_numpy()
    available = np.unique(year[nsc])
    if len(available) < 3:
        return {}
    middle = available[1:-1]
    selection = max(middle, key=lambda candidate: int((nsc & (year == candidate)).sum()))
    hold = year == selection
    train = np.flatnonzero(~hold)
    groups = associations(xy[train], t[train], sigma[train], data.exposure_id.iloc[train].to_numpy(), config)
    obs = [(group.index.to_numpy(), cKDTree(xy[group.index]))
           for _, group in data.loc[hold].groupby("exposure_id")]
    epoch = config["reference_epoch"]
    sample = {}
    for method, curves in groups.items():
        n_predictions = correct = wrong = missing = ambiguous = unverified = 0
        test_distances = []
        for track in curves:
            idx = train[list(track)]
            if len(idx) < 3 or np.ptp(t[idx]) < config["min_seed_years"]:
                continue
            model = fit_motion(xy[idx], t[idx], sigma[idx], epoch)
            if np.linalg.norm(model.coeff[0]) > radius * 3600000:
                continue
            blend = set(labels[idx][labels[idx] >= 0])
            if len(blend) != 1:
                continue
            truth = blend.pop()
            for rows, tree in obs:
                known = rows[labels[rows] == truth]
                if not len(known):
                    continue
                n_predictions += 1
                pred = model.predict(t[known[0]])
                test_distances.append(float(np.linalg.norm(xy[known[0]] - pred)))
                near = rows[tree.query_ball_point(pred, config["gaia_detection_gate_mas"])]
                if not len(near):
                    missing += 1
                elif len(near) > 1:
                    ambiguous += 1
                elif labels[near[0]] == truth:
                    correct += 1
                elif labels[near[0]] < 0:
                    unverified += 1
                else:
                    wrong += 1
        sample[method] = dict(confirmed_prediction=ratio(correct, n_predictions), wrong=wrong,
                              missing=missing, ambiguous=ambiguous, unlabelled=unverified,
                              median_gaia_target_residual_mas=float(np.median(test_distances))
                              if test_distances else None)
    return dict(year=int(selection), holdout_detections=int(hold.sum()), methods=sample)


def translated_control(data, xy, t, sigma, radius, config):
    # Apply the same symmetric crop before and after the shift. The pilot control
    # retains positions relative to each survey and its native observation times.
    shift = np.array([18000., 0.])
    keep = np.linalg.norm(xy, axis=1) <= radius * 3600000 - np.linalg.norm(shift)
    xy = xy[keep]
    t = t[keep]
    sigma = sigma[keep]
    subset = data.loc[keep].reset_index(drop=True)
    sdss = subset.survey.eq("SDSS").to_numpy()
    positions = xy + sdss[:, None] * shift
    counts = {}
    for key, coordinates in [("original", xy), ("translated_sdss", positions)]:
        tracks = associations(coordinates, t, sigma, subset.exposure_id.to_numpy(), config)
        counts[key] = {method: sum(bool(sdss[list(track)].any() and (~sdss[list(track)]).any())
                                   for track in groups) for method, groups in tracks.items()}
    return dict(shift_east_mas=shift[0], detections=len(xy), cross_survey_tracks=counts)


def run(work, region, config):
    settings = config["regions"][region]
    center, radius = (settings["ra"], settings["dec"]), settings["radius"]
    data, xy, t, sigma = detection_arrays(work, center, config)
    gaia = pd.read_csv(work / "validation" / "gaia.csv")
    labels, _ = gaia_labels(data, xy, t, sigma, gaia, center, config)
    result = dict(held_year=held_year(data, xy, t, sigma, labels, radius, config),
                  translated_control=translated_control(data, xy, t, sigma, radius, config))
    save_json(work / "counterfactual_summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", default="smoke")
    parser.add_argument("--config", default="configs/experiment-a.toml")
    args = parser.parse_args()
    with open(args.config, "rb") as file:
        config = tomllib.load(file)
    run(args.work, args.region, config)
