"""Conservative catalogue injection against a real footprint proxy and background."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .__main__ import detection_arrays, tomllib
from .associate import motion_tracks, static_tracks
from .astrometry import seasons
from .io import save_json
from .validation import ratio


def footprints(data, xy):
    entries = []
    for name, group in data.groupby("exposure_id"):
        i = group.index.to_numpy()
        if len(i) < 6:
            continue
        try:
            edges = ConvexHull(xy[i]).equations
        except QhullError:
            continue
        entries.append((name, i, edges, cKDTree(xy[i])))
    return entries


def inject(data, xy, t, sigma, hulls, radius, speeds, rng, config):
    directions = rng.uniform(0, 2 * np.pi, len(speeds))
    location = rng.uniform(-.6, .6, (len(speeds), 2)) * radius * 3600000
    velocities = np.column_stack([np.cos(directions), np.sin(directions)]) * np.asarray(speeds)[:, None]
    result = []
    for k, velocity in enumerate(velocities):
        for exposure, idx, edges, tree in hulls:
            moment = np.median(t[idx])
            position = location[k] + velocity * (moment - config["reference_epoch"])
            # The inner hull is a conservative coverage proxy within the retrieved sample.
            if (np.linalg.norm(position) > (radius + config["access"]["halo_deg"]) * 3600000
                    or np.any(edges[:, :2] @ position + edges[:, 2] > -1000)):
                continue
            distance, neighbor = tree.query(position)
            if distance < 900 or distance > 12000:
                continue
            if rng.random() > 0.8:
                continue
            error = sigma[idx[neighbor]]
            measured = position + rng.normal(0, error)
            result.append((k, exposure, moment, measured, error, data.survey.iloc[idx[neighbor]]))
    return result


def run(work, region, config):
    spec = config["regions"][region]
    center, radius = (spec["ra"], spec["dec"]), spec["radius"]
    data, xy, t, sigma = detection_arrays(work, center, config)
    hulls = footprints(data, xy)
    exposure = data.exposure_id.to_numpy()
    rng = np.random.default_rng(config["seed"])
    summary = {}
    speeds = config["pm_edges"]
    for label in ("main", "stress"):
        cfg = dict(config)
        if label == "stress":
            cfg["mu_max"] = config["stress_mu_max"]
            trial_speeds = [10000]
        else:
            trial_speeds = speeds
        outcomes = {method: {speed: [0, 0] for speed in trial_speeds}
                    for method in ("fixed", "nsc_like", "motion")}
        trials = 3 if label == "main" else 1
        observed = {speed: [] for speed in trial_speeds}
        for _ in range(trials):
            rows = inject(data, xy, t, sigma, hulls, radius, trial_speeds, rng, cfg)
            if not rows:
                continue
            pos = np.vstack([xy, np.stack([row[3] for row in rows])])
            tt = np.r_[t, [row[2] for row in rows]]
            err = np.vstack([sigma, np.stack([row[4] for row in rows])])
            exp = np.r_[exposure, [row[1] for row in rows]]
            group_labels = np.r_[np.full(len(xy), -1, dtype=int), [row[0] for row in rows]]
            fixed = static_tracks(pos, exp, cfg["fixed_radius_mas"])
            nsc = static_tracks(pos, exp, cfg["fixed_radius_mas"], cfg["dbscan_eps_mas"])
            motion, _, _, _ = motion_tracks(pos, tt, err, exp, fixed, cfg)
            for k, speed in enumerate(trial_speeds):
                keep = np.flatnonzero(group_labels == k)
                instruments = {row[5] for row in rows if k == row[0]}
                observed[speed].append(len(keep))
                if len(keep) < 3 or len(instruments) < 2 or len(set(seasons(
                        tt[keep], cfg["min_season_days"]))) < 3:
                    continue
                for method, tracks in [("fixed", fixed), ("nsc_like", nsc),
                                       ("motion", [track.indices for track in motion])]:
                    outcomes[method][speed][1] += 1
                    recovered = False
                    for track in tracks:
                        tagged = group_labels[list(track)]
                        members = (tagged == k).sum()
                        if (members >= cfg["recovery_fraction"] * len(keep)
                                and members >= cfg["recovery_fraction"] * len(track)):
                            recovered = True
                            break
                    outcomes[method][speed][0] += int(recovered)
        summary[label] = dict(
            trials=trials, detections_by_speed={str(k): v for k, v in observed.items()},
            methods={method: {str(speed): ratio(*counts) for speed, counts in group.items()}
                     for method, group in outcomes.items()})
    save_json(work / "injection_summary.json", dict(footprint_exposures=len(hulls), results=summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", default="smoke")
    parser.add_argument("--config", default="configs/experiment-a.toml")
    args = parser.parse_args()
    with open(args.config, "rb") as file:
        config = tomllib.load(file)
    run(args.work, args.region, config)
