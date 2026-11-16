"""Probe year-out prediction and translated-background associations."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .__main__ import detection_arrays, tomllib
from .associate import Track, motion_tracks, static_tracks
from .astrometry import fit_motion
from .io import save_json
from .provenance import run_stage
from .validation import gaia_labels


def associations(xy, t, sigma, exposures, config):
    fixed = static_tracks(xy, exposures, config["fixed_radius_mas"])
    nsc = static_tracks(xy, exposures, config["fixed_radius_mas"], config["dbscan_eps_mas"])
    selected, _, _, _ = motion_tracks(xy, t, sigma, exposures, fixed, config)
    return {"fixed": [Track(g, 0, "baseline", 0) for g in fixed],
            "nsc_like": [Track(g, 0, "baseline", 0) for g in nsc], "motion": selected}


def held_year(data, xy, t, sigma, labels, radius, config, include_methods=("fixed", "nsc_like", "motion")):
    year = np.floor(t).astype(int)
    survey = data.survey.to_numpy()
    available = np.unique(year[survey == "NSC"])
    if len(available) < 3:
        return {}
    middle = available[1:-1]
    selection = max(middle, key=lambda candidate: int(((survey == "NSC") & (year == candidate)).sum()))
    hold = year == selection
    train = np.flatnonzero(~hold)
    groups = associations(xy[train], t[train], sigma[train], data.exposure_id.iloc[train].to_numpy(), config)
    visits = []
    for _, group in data.loc[hold].groupby("exposure_id"):
        rows = group.index.to_numpy()
        visits.append((rows, float(np.median(t[rows])), cKDTree(xy[rows]), sigma[rows].max()))
    epoch = config["reference_epoch"]
    sample = {}
    for method in include_methods:
        curves = groups[method]
        workflows = []
        for track in curves:
            idx = train[list(track.indices)]
            if len(idx) < 3 or np.ptp(t[idx]) < config["min_seed_years"]:
                continue
            model = fit_motion(xy[idx], t[idx], sigma[idx], epoch)
            if np.linalg.norm(model.coeff[0]) > radius * 3600000:
                continue
            associated = set(survey[idx])
            kind = "cross_survey" if associated == {"SDSS", "NSC"} else "nsc_only"
            if "NSC" not in associated:
                continue
            known = set(labels[idx][labels[idx] >= 0])
            tag = "with_gaia" if len(known) == 1 else ("no_gaia" if not known else "mixed_gaia")
            mags = data.iloc[idx].groupby("band").magnitude.median().to_dict()
            r_nsc = data.iloc[idx].query("survey == 'NSC' and band == 'r'").magnitude.median()
            magnitude_bin = ("r_unmeasured" if not np.isfinite(r_nsc)
                             else "r_ge_20p5" if r_nsc >= 20.5 else "r_lt_20p5")
            workflows.append((model, kind, tag, known, mags, magnitude_bin, track.status))
        sample[method] = {}
        for status_mode in ("all_selected", "secure"):
            eligible = [w for w in workflows if status_mode == "all_selected" or w[-1] == "secure"]
            counts = defaultdict(Counter)
            source_counts = Counter((w[1], w[2], w[5]) for w in eligible)
            for rows, moment, tree, max_sigma in visits:
                if not eligible:
                    continue
                predictions = np.array([w[0].predict(moment) for w in eligible])
                variances = []
                for model, *_ in eligible:
                    vector = np.array([1, moment - model.epoch])
                    variances.append(model.cov @ vector @ vector)
                proposal_tree = cKDTree(predictions)
                for j, (model, kind, tag, known, mags, magnitude_bin, _) in enumerate(eligible):
                    counts_here = counts[(kind, tag, magnitude_bin)]
                    if tag == "mixed_gaia":
                        continue
                    truth_rows = rows[np.isin(labels[rows], list(known))]
                    if tag == "with_gaia" and len(truth_rows) != 1:
                        continue
                    counts_here["attempted"] += 1
                    axis_sigma = np.sqrt(np.maximum(variances[j], 0))
                    gate = min(1500, 3 * np.hypot(axis_sigma.max(), max_sigma))
                    pos = predictions[j]
                    near = np.array(tree.query_ball_point(pos, gate), dtype=int)
                    if not len(near):
                        counts_here["no_candidate"] += 1
                        continue
                    d2 = ((xy[near] - model.predict(t[near])) ** 2
                          / (axis_sigma ** 2 + sigma[near] ** 2)).sum(axis=1)
                    inside = near[d2 <= 9]
                    if len(inside) != 1:
                        counts_here["missing_or_competing_detections"] += 1
                        continue
                    hit = inside[0]
                    if tag == "with_gaia":
                        verdict = ("labelled_correct" if hit == truth_rows[0]
                                   else "unconfirmed" if labels[hit] < 0 else "labelled_wrong")
                        counts_here[verdict] += 1
                        continue
                    # Without Gaia require uncrowded predictions, photometry and unique track support.
                    neighbours = tree.query_ball_point(pos, 5000, return_length=True)
                    chance = -np.expm1(-neighbours * (gate / 5000) ** 2)
                    phot = data.iloc[hit]
                    expected = mags.get(phot.band, np.nan)
                    competing = proposal_tree.query_ball_point(xy[hit], gate, return_length=True)
                    if (competing == 1 and chance < config.get("dark_max_chance", .1)
                            and np.isfinite(expected) and np.isfinite(phot.magnitude)
                            and abs(expected - phot.magnitude) <= config.get("dark_max_band_delta", .75)):
                        counts_here["consistent_unconfirmed"] += 1
                    else:
                        counts_here["inconclusive"] += 1
            sample[method][status_mode] = {
                f"{kind}/{tag}/{magnitude_bin}": dict(training_tracks=n, **counts[(kind, tag, magnitude_bin)])
                for (kind, tag, magnitude_bin), n in source_counts.items()
            }
    return dict(year=int(selection), holdout_detections=int(hold.sum()), methods=sample)


def translated_control(data, xy, t, sigma, radius, config):
    vectors = np.array([[0, 0], [12000, 0], [-18000, 0], [0, 18000],
                        [0, -30000], [18000, 18000]], dtype=float)
    keep = np.linalg.norm(xy, axis=1) <= radius * 3600000 - np.linalg.norm(vectors, axis=1).max()
    xy = xy[keep]
    t = t[keep]
    sigma = sigma[keep]
    subset = data.loc[keep].reset_index(drop=True)
    sdss = subset.survey.eq("SDSS").to_numpy()
    results = []
    for shift in vectors:
        pt = xy + sdss[:, None] * shift
        collected = associations(pt, t, sigma, subset.exposure_id.to_numpy(), config)
        aggregate = {}
        for method, tracks in collected.items():
            evidence = defaultdict(list)
            for track in tracks:
                idx = list(track.indices)
                if not (sdss[idx].any() and (~sdss[idx]).any() and np.ptp(t[idx]) > 0):
                    continue
                model = fit_motion(pt[idx], t[idx], sigma[idx], config["reference_epoch"])
                evidence[track.status].append([track.score, model.span,
                                                1 / model.cov[:, 1, 1].sum()])
            aggregate[method] = {
                key: dict(n=len(value), median_score_span_leverage=np.median(value, axis=0))
                for key, value in evidence.items()
            }
        held = held_year(subset, pt, t, sigma, np.full(len(pt), -1), radius, config, ("motion",))
        results.append(dict(shift_east_north_mas=shift, tracks_by_status=aggregate,
                            heldout_without_truth=held))
    return dict(detections=len(xy), variants=results)


def run(work, region, config):
    settings = config["regions"][region]
    config = dict(config, **settings.get("overrides", {}))
    center, radius = (settings["ra"], settings["dec"]), settings["radius"]
    data, xy, t, sigma = detection_arrays(work, center, config)
    gaia = pd.read_csv(work / "validation" / "gaia.csv")
    labels, _ = gaia_labels(data, xy, t, sigma, gaia, center, config)
    result = dict(held_year=held_year(data, xy, t, sigma, labels, radius, config),
                  translated_control=translated_control(data, xy, t, sigma, radius, config))
    save_json(work / "prediction_control_summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", default="smoke")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment-a-v02.toml"))
    args = parser.parse_args()
    with args.config.open("rb") as file:
        config = tomllib.load(file)
    with run_stage(args.work, args.config, args.region, "prediction-control",
                   ["detections.csv", "validation/gaia.csv"],
                   ["prediction_control_summary.json"]) as execute:
        if execute:
            run(args.work, args.region, config)
