"""Gaia-only evaluation, paired truth-linked fits and annual holdout."""
import json

import numpy as np
import pandas as pd
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
from scipy.spatial import cKDTree

from .astrometry import apparent_positions, fit_motion, project, seasons
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
    g = g.loc[np.isfinite(g[required].to_numpy(dtype=float)).all(axis=1)].reset_index(drop=True)
    ref = project(g.ra, g.dec, center)
    # Transform the two local Gaia tangent components to the region's projection.
    step = 1.0 / 3600
    cos_dec = np.cos(np.deg2rad(g.dec))
    east = (project(g.ra + step / cos_dec, g.dec, center) - ref) / 1000
    north = (project(g.ra, g.dec + step, center) - ref) / 1000
    velocity = east * g.pmra.to_numpy()[:, None] + north * g.pmdec.to_numpy()[:, None]
    t0 = g.ref_epoch.to_numpy()
    use_parallax = config.get("truth_mode") == "annual_parallax"
    parallax_limit = config.get("gaia_max_parallax_nearby", 2000) if use_parallax else config[
        "gaia_max_parallax_for_linear_truth"]
    reliable = ((g.ruwe < config["gaia_max_ruwe"]) &
                (g.pmra_error < config["gaia_max_pm_error"]) &
                (g.pmdec_error < config["gaia_max_pm_error"]) &
                (g.parallax.abs() < parallax_limit)).to_numpy()
    if use_parallax:
        fitted_parallax = g.parallax.clip(lower=0).to_numpy()
        rv = g.radial_velocity.fillna(0).to_numpy() if "radial_velocity" in g else np.zeros(len(g))
        if "parallax_error" in g:
            nearby = g.parallax.abs() >= config["gaia_max_parallax_for_linear_truth"]
            reliable &= (~nearby | g.parallax_error.between(0, 2)).to_numpy()
        ra, dec, pmra, pmdec = (g[key].to_numpy() for key in ("ra","dec","pmra","pmdec"))
        unique_time, inverse = np.unique(time, return_inverse=True)
        if len(unique_time):
            observers = get_body_barycentric("earth", Time(unique_time, format="jyear", scale="tt")).xyz.to_value("AU").T

    def propagated(indices, value, observer=None):
        if not use_parallax:
            return ref[indices] + (value - t0[indices])[:, None] * velocity[indices]
        return apparent_positions(ra[indices], dec[indices], pmra[indices],
                                  pmdec[indices], fitted_parallax[indices], rv[indices],
                                  value - t0[indices], observer, center)

    label = np.full(len(data), -1, dtype=int)
    for _, group in data.groupby("exposure_id", sort=False):
        indices = group.index.to_numpy()
        midpoint = np.median(time[indices])
        if use_parallax:
            middle = indices[np.argmin(abs(time[indices] - midpoint))]
            midpoint = time[middle]
            mid_observer = observers[inverse[middle]]
        else:
            mid_observer = None
        pos = propagated(np.arange(len(g)), midpoint, mid_observer)
        tree = cKDTree(pos)
        search = 2 * config["gaia_detection_gate_mas"] + np.max(np.linalg.norm(velocity, axis=1),
                                                                initial=0) * np.ptp(time[indices])
        if use_parallax:
            search += 8 * fitted_parallax.max(initial=0) * np.ptp(time[indices])
        eligible = indices[np.max(sigma[indices],axis=1)*3 <= config["gaia_detection_gate_mas"]]
        for i, candidates in zip(eligible,tree.query_ball_point(xy[eligible],search)):
            if not candidates:
                continue
            near = np.asarray(candidates,dtype=int)
            dot = propagated(near, time[i], observers[inverse[i]] if use_parallax else None)
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
    g["truth_reliable"] = reliable
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


def truth_requirements(truth, survey, t, gap):
    requests = {}
    for label, rows in truth.items():
        sdss = rows[survey[rows] == "SDSS"]
        nsc = rows[survey[rows] == "NSC"]
        groups = seasons(t[rows], gap)
        first = t[sdss].min()
        early = sdss[t[sdss] <= first + 1 / 365.25]
        requests[label] = (set(sdss), set(nsc), set(early),
                           [set(rows[groups == g]) for g in np.unique(groups)])
    return requests


def matched_object(indices, labels, required, fraction):
    known = np.unique(labels[indices][labels[indices] >= 0])
    if len(known) != 1 or known[0] not in required:
        return None
    label = known[0]
    sdss, nsc, early, seasons_required = required[label]
    members = set(indices)
    if (len(members & sdss) >= fraction * len(sdss)
            and len(members & nsc) >= fraction * len(nsc)
            and members & early
            and all(members & group for group in seasons_required)):
        return label
    return None


def confirmation_counts(tracks, survey, labels, core_indices):
    counts = dict(confirmed=0, discordant=0, unknown=0, unknown_members=0,
                  discordant_members=0, labelled_members=0)
    for row, track in enumerate(tracks):
        if row not in core_indices:
            continue
        idx = np.asarray(track["indices"])
        if not np.any(survey[idx] == "NSC") or not np.any(survey[idx] == "SDSS"):
            continue
        labs = labels[idx]
        valid = labs[labs >= 0]
        counts["unknown_members"] += int((labs < 0).sum())
        counts["labelled_members"] += len(valid)
        _, populations = np.unique(valid, return_counts=True)
        if len(populations) > 1:
            counts["discordant"] += 1
            counts["discordant_members"] += len(valid) - populations.max()
        elif len(valid) == len(idx):
            counts["confirmed"] += 1
        else:
            counts["unknown"] += 1
    return counts


def residual_stats(values):
    rows = np.asarray(values)
    if not len(rows):
        return dict(n=0)
    median = np.median(rows, axis=0)
    return dict(n=len(rows), median_masyr=median,
                nmad_masyr=1.4826 * np.median(np.abs(rows - median), axis=0))


def stage_evaluation(tracks, candidates, audit, fits, truth, labels, data, t, gaia, config):
    survey = data.survey.to_numpy()
    required = truth_requirements(truth, survey, t, config["min_season_days"])
    fraction = config["recovery_fraction"]

    def matches(groups):
        return [matched_object(np.asarray(g, dtype=int), labels, required, fraction) for g in groups]

    generated = set(matches(audit["generated"]))
    screened = set(matches(audit["screened"]))
    merged = matches([c["indices"] for c in candidates])
    by_object = {key: set() for key in truth}
    for component in audit["components"]:
        for candidate in component["candidates"]:
            key = merged[candidate]
            if key is not None:
                by_object[key].add(component["state"])
    candidate_reached = set(merged)
    fits_by_method = {}
    for row in fits:
        if row["in_core"]:
            fits_by_method.setdefault(row["method"], {})[row["track"]] = row
    speed = np.hypot(gaia.pmra, gaia.pmdec)
    summary = {}
    detail = {}
    for method, group in tracks.items():
        selected = set()
        secure = set()
        ambiguous = set()
        tentative = set()
        compatible = set()
        residuals = {}
        core_fits = fits_by_method.get(method, {})
        for j, key in enumerate(matches([x["indices"] for x in group])):
            if key is None or j not in core_fits:
                continue
            model = core_fits[j]
            status = group[j]["status"]
            selected.add(key)
            if status == "secure":
                secure.add(key)
            elif status == "ambiguous":
                ambiguous.add(key)
            elif status.startswith("tentative_"):
                tentative.add(key)
            goal = gaia.loc[key]
            error = np.array([model["pm_x"] - goal["pm_tangent_x"],
                              model["pm_y"] - goal["pm_tangent_y"]])
            residuals[key] = error
            dx = error[0] / np.hypot(np.sqrt(model["cov_x"][1][1]), goal.pmra_error)
            dy = error[1] / np.hypot(np.sqrt(model["cov_y"][1][1]), goal.pmdec_error)
            if dx * dx + dy * dy <= 5.991:
                compatible.add(key)
        unresolved = {key for key, states in by_object.items()
                      if states and states <= {"unresolved", "feasible_incumbent"}} if method == "motion" else set()
        reached = candidate_reached if method == "motion" else selected
        counts = confirmation_counts(group, survey, labels, set(core_fits))
        secure_counts = confirmation_counts(group, survey, labels,
                                           {j for j in core_fits if group[j]["status"] == "secure"})
        n = len(required)

        def summarize_counts(counts):
            d = counts["confirmed"] + counts["discordant"] + counts["unknown"]
            checked = counts["confirmed"] + counts["discordant"]
            return dict(**counts, wrong_among_confirmed=ratio(counts["discordant"], checked),
                        bounds_with_unknown=[counts["discordant"] / d,
                                             (counts["discordant"] + counts["unknown"]) / d] if d else None)

        stats = dict(candidate_recall=ratio(len(reached - {None}), n),
                     selected_recall=ratio(len(selected), n),
                     end_to_end_selected_pm=residual_stats(list(residuals.values())),
                     end_to_end_secure_pm=residual_stats([residuals[k] for k in secure]),
                     secure_recall=ratio(len(secure), n) if method == "motion" else None,
                     ambiguous_recall=ratio(len(ambiguous), n),
                     tentative_recall=ratio(len(tentative), n),
                     unresolved_fraction=ratio(len(unresolved), n),
                     motion_compatible=ratio(len(compatible), n),
                     object_confirmation=summarize_counts(counts),
                     secure_object_confirmation=summarize_counts(secure_counts))
        stats["pm_bins"] = []
        for low, high in zip(config["pm_edges"], config["pm_edges"][1:]):
            keys = {key for key in truth if low <= speed[key] < high}
            stats["pm_bins"].append(dict(
                low=low, high=high, **ratio(len(keys & selected), len(keys)),
                candidate=ratio(len(keys & reached), len(keys)),
                secure=ratio(len(keys & secure), len(keys)) if method == "motion" else None))
        summary[method] = stats
        detail[method] = dict(selected=selected, secure=secure, compatible=compatible)
    motion_detail = detail["motion"]
    stages = {}
    previous = set(truth)
    for name, keys in [
        ("has_input", previous), ("generated", previous & generated),
        ("screened", previous & screened), ("retained", previous & candidate_reached),
        ("conflict_optimal", {key for key, states in by_object.items() if "optimal" in states}),
        ("selected_optimal", motion_detail["selected"]),
        ("secure", motion_detail["secure"]), ("pm_compatible_secure", motion_detail["compatible"]),
    ]:
        reached = previous & keys
        stages[name] = dict(total=ratio(len(reached), len(truth)),
                            conditional=ratio(len(reached), len(previous)))
        previous = reached
    trace = {str(key): dict(generated=key in generated, screened=key in screened,
                            retained=key in candidate_reached,
                            solver_states=sorted(by_object[key]),
                            selected=key in motion_detail["selected"],
                            secure=key in motion_detail["secure"],
                            pm_compatible=key in motion_detail["compatible"]) for key in truth}
    return summary, stages, trace



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
    target = work / "validation" / "challenge_source.txt"
    if target.exists():
        wanted = int(target.read_text().strip())
        truth = {k: rows for k, rows in truth.items() if int(gaia.source_id.iloc[k]) == wanted}
    candidates = json.loads((work / "candidates.json").read_text())
    audit = json.loads((work / "association_audit.json").read_text())
    fits = json.loads((work / "fits.json").read_text())
    results, stages, trace = stage_evaluation(tracks, candidates, audit, fits, truth, labels, data, t,
                                               gaia, config)
    precision, paired_rows = paired(truth, data, xy, t, sigma, gaia, config)
    paired_rows.to_csv(work / "validation" / "paired.csv", index=False)
    save_json(work / "validation" / "recovery_trace.json", trace)
    save_json(work / "validation.json", dict(gaia_rows=len(gaia), challenge_target_requested=target.exists(), labeled_detections=int((labels >= 0).sum()),
                                           truth_cross_survey=len(truth), methods=results,
                                           recovery_stages=stages, paired=precision))
