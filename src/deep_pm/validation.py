"""Gaia-only evaluation, paired truth-linked fits and annual holdout."""
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .astrometry import fit_motion, project, seasons
from .io import save_json


def wilson(n, d):
    if not d:
        return None
    p = n / d
    z = 1.959963984540054
    center = (p + z * z / (2 * d)) / (1 + z * z / d)
    half = z * np.sqrt(p * (1 - p) / d + (z / (2 * d)) ** 2) / (1 + z * z / d)
    return [float(max(0, center - half)), float(min(1, center + half))]


def ratio(n, d):
    return dict(n=int(n), denominator=int(d), value=n / d if d else None, wilson95=wilson(n, d))


def gaia_labels(data, xy, time, sigma, gaia, center, config):
    required = ["ra", "dec", "pmra", "pmdec", "parallax", "ruwe", "pmra_error", "pmdec_error", "ref_epoch"]
    g = gaia.query("astrometric_params_solved in [31, 95]") if "astrometric_params_solved" in gaia else gaia
    g = g.loc[np.isfinite(g[required]).all(axis=1)].reset_index(drop=True)
    ref = project(g.ra, g.dec, center)
    # Transform the two local Gaia tangent components to the region's projection.
    step = 1.0 / 3600
    cos_dec = np.cos(np.deg2rad(g.dec))
    east = (project(g.ra + step / cos_dec, g.dec, center) - ref) / 1000
    north = (project(g.ra, g.dec + step, center) - ref) / 1000
    velocity = east * g.pmra.to_numpy()[:, None] + north * g.pmdec.to_numpy()[:, None]
    t0 = g.ref_epoch.to_numpy()
    reliable = ((g.ruwe < config["gaia_max_ruwe"]) &
                (g.pmra_error < config["gaia_max_pm_error"]) &
                (g.pmdec_error < config["gaia_max_pm_error"]) &
                (g.parallax.abs() < config["gaia_max_parallax_for_linear_truth"])).to_numpy()
    label = np.full(len(data), -1, dtype=int)
    for _, group in data.groupby("exposure_id", sort=False):
        indices = group.index.to_numpy()
        midpoint = np.median(time[indices])
        pos = ref + (midpoint - t0)[:, None] * velocity
        tree = cKDTree(pos)
        search = 2 * config["gaia_detection_gate_mas"] + np.max(np.linalg.norm(velocity, axis=1),
                                                                initial=0) * np.ptp(time[indices])
        for i in indices:
            if sigma[i].max() * 3 > config["gaia_detection_gate_mas"]:
                continue
            near = np.array(tree.query_ball_point(xy[i], search), dtype=int)
            dot = ref[near] + (time[i] - t0[near])[:, None] * velocity[near]
            dist = np.linalg.norm(dot - xy[i], axis=1)
            order = np.argsort(dist)
            if (len(order) and dist[order[0]] <= config["gaia_detection_gate_mas"]
                    and reliable[near[order[0]]]
                    and (len(order) == 1 or dist[order[1]] - dist[order[0]] >= 300)):
                label[i] = near[order[0]]
        labeled = indices[label[indices] >= 0]
        duplicate = pd.Series(label[labeled]).duplicated(keep=False).to_numpy()
        label[labeled[duplicate]] = -1
    g["pm_tangent_x"], g["pm_tangent_y"] = velocity.T
    return label, g


def groups_by_label(labels, data, t, gaia, center, radius, config):
    pos = project(gaia.ra, gaia.dec, center) + (
        config["reference_epoch"] - gaia.ref_epoch.to_numpy())[:, None] * gaia[
            ["pm_tangent_x", "pm_tangent_y"]].to_numpy()
    inside = np.linalg.norm(pos, axis=1) <= 3600000 * np.tan(np.deg2rad(radius)) * 180 / np.pi
    truth = {}
    for label in sorted(set(labels[labels >= 0])):
        i = np.flatnonzero(labels == label)
        if inside[label] and data.survey.iloc[i].nunique() == 2 and np.ptp(t[i]) >= config["min_seed_years"]:
            truth[label] = i
    return truth


def evaluate(tracks, truth, labels, data, speed, config, minimum_score=None):
    mapping = {key: [] for key in truth}
    fragment = {key: 0 for key in truth}
    mixed = eligible = wrong = known = unknown = ambiguity = 0
    for track in tracks:
        idx = np.array(track["indices"])
        if len(idx) < 2 or (minimum_score is not None and track["score"] < minimum_score):
            continue
        labs = labels[idx]
        if not any(label in truth for label in labs):
            # Include unlabeled cross-survey candidates in the unknown count.
            if data.survey.iloc[idx].nunique() == 2:
                unknown += int((labs < 0).sum())
            continue
        if data.survey.iloc[idx].nunique() < 2:
            for label in set(labs):
                if label in fragment:
                    fragment[label] += 1
            continue
        eligible += 1
        ambiguity += track["status"] != "secure" and track["status"] != "baseline"
        valid = labs[labs >= 0]
        objs, counts = np.unique(valid, return_counts=True)
        mixed += len(objs) > 1
        unknown += len(labs) - len(valid)
        known += len(valid)
        wrong += len(valid) - counts.max()
        for label, count in zip(objs, counts):
            if label in fragment:
                fragment[label] += 1
            if label in mapping and len(objs) == 1:
                both = data.survey.iloc[idx[labs == label]].nunique() == 2
                if both:
                    mapping[label].append(count / len(truth[label]))
    recovered = {label: max(values, default=0) >= config["recovery_fraction"]
                 for label, values in mapping.items()}
    edges = config["pm_edges"]
    binned = []
    for low, high in zip(edges, edges[1:]):
        selected = [label for label in truth if low <= speed[label] < high]
        binned.append(dict(low=low, high=high, **ratio(sum(recovered[n] for n in selected), len(selected))))
    return dict(recovery=ratio(sum(recovered.values()), len(recovered)), pm_bins=binned,
                mixed_tracks=ratio(mixed, eligible), wrong_known_detections=ratio(wrong, known),
                unlabelled_members=unknown, ambiguous_tracks=ratio(ambiguity, eligible),
                fragmented=ratio(sum(n > 1 for n in fragment.values()), len(fragment)))


def paired(truth, data, xy, t, sigma, gaia, config):
    residual_rows = []
    epoch = config["reference_epoch"]
    for label, i in truth.items():
        nsc = i[data.survey.iloc[i].to_numpy() == "NSC"]
        sdss = i[data.survey.iloc[i].to_numpy() == "SDSS"]
        if len(nsc) < 2 or np.ptp(t[nsc]) < config["min_seed_years"]:
            continue
        target = gaia.loc[label, ["pm_tangent_x", "pm_tangent_y"]].to_numpy(dtype=float)
        first = sdss[np.argmin(t[sdss])]
        for tag, idx in [("nsc_only", nsc), ("one_sdss", np.r_[nsc, first]), ("all_sdss", i)]:
            model = fit_motion(xy[idx], t[idx], sigma[idx], epoch)
            residual = model.coeff[1] - target
            residual_rows.append(dict(
                fit=tag, source=label, east=residual[0], north=residual[1],
                z_east=residual[0] / np.hypot(np.sqrt(model.cov[0, 1, 1]), gaia.pmra_error.iloc[label]),
                z_north=residual[1] / np.hypot(np.sqrt(model.cov[1, 1, 1]), gaia.pmdec_error.iloc[label]),
                baseline=model.span, n_sdss=len(sdss), pm=np.linalg.norm(target),
            ))
    result = {}
    if not residual_rows:
        return result, pd.DataFrame()
    rows = pd.DataFrame(residual_rows)
    for name, group in rows.groupby("fit"):
        vals = group[["east", "north"]].to_numpy()
        median = np.median(vals, axis=0)
        result[name] = dict(n=len(vals), median_masyr=median,
                            nmad_masyr=1.4826 * np.median(np.abs(vals - median), axis=0),
                            median_baseline_yr=float(group.baseline.median()),
                            over_20_masyr=ratio(int((np.linalg.norm(vals, axis=1) > 20).sum()), len(vals)),
                            median_abs_normalized=np.median(np.abs(group[["z_east", "z_north"]]), axis=0))
    return result, rows


def validate(work, center, config):
    # Import here so both standalone CLI forms use the same detection preparation.
    from .__main__ import detection_arrays

    gaia_path = work / "validation" / "gaia.csv"
    if not gaia_path.exists():
        raise FileNotFoundError("Place Gaia evaluation input only under validation/gaia.csv on Scratch")
    data, xy, t, sigma = detection_arrays(work, center, config)
    labels, gaia = gaia_labels(data, xy, t, sigma, pd.read_csv(gaia_path), center, config)
    tracks = json.loads((work / "tracks.json").read_text())
    region = json.loads((work / "run_manifest.json").read_text())["region"]
    truth = groups_by_label(labels, data, t, gaia, center, config["regions"][region]["radius"], config)
    speed = np.hypot(gaia.pmra, gaia.pmdec)
    results = {method: evaluate(group, truth, labels, data, speed, config)
               for method, group in tracks.items()}
    curves = {}
    for method, group in tracks.items():
        grid = np.arange(0, 14, 1) if method == "motion" else np.log(np.arange(1, 10))
        curves[method] = [{"threshold": float(cut), **evaluate(group, truth, labels, data, speed, config, cut)}
                          for cut in grid]
    precision, paired_rows = paired(truth, data, xy, t, sigma, gaia, config)
    paired_rows.to_csv(work / "validation" / "paired.csv", index=False)
    save_json(work / "validation.json", dict(gaia_rows=len(gaia), labeled_detections=int((labels >= 0).sum()),
                                           truth_cross_survey=len(truth), methods=results,
                                           curves=curves, paired=precision))
