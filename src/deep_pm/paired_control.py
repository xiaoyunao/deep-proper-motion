"""Four-way fixed-association control and SDSS run sensitivity."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import access
from .__main__ import detection_arrays, tomllib
from .astrometry import fit_motion, project
from .io import save_json
from .provenance import run_stage
from .validation import gaia_labels, groups_by_label


def summarize(values):
    values = np.asarray(values)
    if not len(values):
        return dict(n=0)
    center = np.median(values, axis=0)
    return dict(n=len(values), median=center, nmad=1.4826 * np.median(abs(values - center), axis=0))


def paired_contrasts(comparisons, seed):
    pairs = (("one_sdss", "nsc_refit"), ("all_sdss", "one_sdss"), ("all_sdss", "nsc_published"))
    n = len(comparisons.get("one_sdss", ()))
    if n < 5:
        return {}
    draws = np.random.default_rng(seed).integers(0, n, (5000, n))
    metrics = {}
    for method, values in comparisons.items():
        observations = np.asarray(values)
        if observations.shape != (n, 2):
            raise ValueError("Paired residual methods must contain the same objects in both components")
        resampled = observations[draws]
        center = np.median(resampled, axis=1)
        metrics[method] = 1.4826 * np.median(abs(resampled - center[:, None, :]), axis=1)
    return {f"{a}_minus_{b}": dict(
        n=n, delta_nmad_masyr=summarize(comparisons[a])["nmad"] - summarize(comparisons[b])["nmad"],
        paired_bootstrap95_masyr=np.quantile(metrics[a] - metrics[b], [.025, .975], axis=0).T)
        for a, b in pairs}


def common_catalogue_ids(truth, labels, data, original):
    ids = data.detection_id.map(original)
    nsc = data.survey.eq("NSC").to_numpy()
    labelled = pd.DataFrame({"obj": ids[nsc].to_numpy(), "label": labels[nsc]})
    purity = labelled[labelled.label >= 0].groupby("obj").label.nunique()
    linked = {}
    status = Counter()
    for g, rows in truth.items():
        visits = rows[nsc[rows]]
        if ids.iloc[visits].isna().any():
            status["missing_nsc_index"] += 1
            continue
        names = set(ids.iloc[visits])
        if len(names) != 1:
            status["split_nsc_index"] += 1
            continue
        name = names.pop()
        if purity.get(name) != 1:
            status["mixed_nsc_index"] += 1
            continue
        linked[g] = name
    return linked, status


def published_rows(cache, identities):
    parts = []
    names = sorted(set(identities))
    for start in range(0, len(names), 100):
        where = "id IN (" + ",".join(map(access.literal, names[start:start + 100])) + ")"
        parts.append(cache.checked("nsc", "id,ra,dec,pmra,pmdec,pmraerr,pmdecerr,deltamjd",
                                   "nsc_dr2.object", where))
    if not parts:
        return pd.DataFrame(columns=["id"]).set_index("id")
    return pd.concat(parts, ignore_index=True).set_index("id", verify_integrity=True)


def catalogue_motion(row, center):
    origin = project([row.ra], [row.dec], center)[0]
    step = 1 / 3600
    east = project([row.ra + step / np.cos(np.deg2rad(row.dec))], [row.dec], center)[0] - origin
    north = project([row.ra], [row.dec + step], center)[0] - origin
    return (row.pmra * east + row.pmdec * north) / 1000


def shared_group_motion(xy, time, sigma, groups, tau):
    """Linear GLS with independent variance plus a per-group common offset."""
    design = np.column_stack([np.ones(len(time)), time - time.mean()])
    coeff = []
    visits = {}
    for i, group in enumerate(groups):
        visits.setdefault(group, []).append(i)
    for axis in range(2):
        weight = sigma[:, axis].astype(float) ** -2
        data = np.column_stack([design, xy[:, axis]]) * weight[:, None]
        for indices in visits.values():
            w = weight[indices]
            correction = tau ** 2 / (1 + tau ** 2 * w.sum())
            data[indices] -= correction * w[:, None] * data[indices].sum(axis=0)
        normal = design.T @ data
        coeff.append(np.linalg.solve(normal[:, :2], normal[:, 2])[1])
    return np.array(coeff)


def run(work, region, config):
    spec = config["regions"][region]
    center = (spec["ra"], spec["dec"])
    data, xy, t, sigma = detection_arrays(work, center, config)
    labels, gaia = gaia_labels(data, xy, t, sigma, pd.read_csv(work / "validation" / "gaia.csv"),
                              center, config)
    truth = groups_by_label(labels, data, t, gaia, center, spec["radius"], config)
    original = pd.read_csv(work / "external_labels.csv", dtype=str).set_index("detection_id")
    linked, excluded = common_catalogue_ids(truth, labels, data, original.original_object_id)
    published = published_rows(access.Cache(work / "cache"), linked.values())
    comparisons = defaultdict(list)
    fit_only = defaultdict(list)
    offsets = defaultdict(list)
    run_impact = defaultdict(list)
    choices = (("all_independent", 50, 0), ("all_shared25", 50, 25),
               ("all_shared50", 50, 50), ("all_floor25", 25, 0), ("all_floor100", 100, 0))
    sensitivities = defaultdict(list)
    epoch = config["reference_epoch"]
    surveys = data.survey.to_numpy()
    raw_error = data[["sigma_east_mas", "sigma_north_mas"]].to_numpy()
    nights = np.floor(data.observation_time.to_numpy()).astype(int)
    visits = data.exposure_id.to_numpy()
    camera = data.detector_id.to_numpy()
    for g, rows in truth.items():
        nsc = rows[surveys[rows] == "NSC"]
        sdss = rows[surveys[rows] == "SDSS"]
        if np.ptp(t[nsc]) < config["min_seed_years"]:
            excluded["short_nsc_baseline"] += 1
            continue
        target = gaia.loc[g, ["pm_tangent_x", "pm_tangent_y"]].to_numpy(dtype=float)
        early = sdss[np.argmin(t[sdss])]
        fits = {}
        all_model = None
        for name, selection in [("nsc_refit", nsc), ("one_sdss", np.r_[nsc, early]), ("all_sdss", rows)]:
            model = fit_motion(xy[selection], t[selection], sigma[selection], epoch)
            fits[name] = model.coeff[1] - target
            fit_only[name].append(fits[name])
            if name == "all_sdss":
                all_model = model
        if g in linked:
            key = linked[g]
            if key in published.index:
                row = published.loc[key]
                values = np.array([row.pmra, row.pmdec, row.pmraerr, row.pmdecerr, row.deltamjd])
                if np.isfinite(values).all() and (values[2:] > 0).all() and abs(values[:2]).max() < 1e5:
                    comparisons["nsc_published"].append(catalogue_motion(row, center) - target)
                    for name, residual in fits.items():
                        comparisons[name].append(residual)
                else:
                    excluded["missing_published_motion"] += 1
            else:
                excluded["missing_published_row"] += 1
        for run in np.unique(visits[sdss]):
            visit = sdss[visits[sdss] == run]
            reduced = np.setdiff1d(rows, visit)
            without = fit_motion(xy[reduced], t[reduced], sigma[reduced], epoch)
            with_run = fit_motion(xy[np.r_[nsc, visit]], t[np.r_[nsc, visit]], sigma[np.r_[nsc, visit]], epoch)
            run_impact[run].append(np.r_[all_model.coeff[1] - without.coeff[1],
                                         with_run.coeff[1] - target])
            for c in np.unique(camera[visit]):
                i = visit[camera[visit] == c]
                offsets[(run, c)].append(np.median(xy[i] - without.predict(t[i]), axis=0))
        group_ids = [(visits[i] if surveys[i] == "SDSS" else str(nights[i])) for i in rows]
        for name, floor, tau in choices:
            pm = shared_group_motion(xy[rows], t[rows], np.hypot(raw_error[rows], floor),
                                     group_ids, tau)
            sensitivities[name].append(pm - target)
    save_json(work / "paired_control_summary.json", dict(
        role=spec["role"], truth_sources=len(truth), exclusion_counts=dict(excluded),
        fixed_truth_three_way={name: summarize(vals) for name, vals in fit_only.items()},
        fixed_truth_common_four_way={name: summarize(vals) for name, vals in comparisons.items()},
        shared_error_sensitivity={name: summarize(vals) for name, vals in sensitivities.items()},
        sdss_run_heldout=dict((str(run), dict(
            all_minus_drop_run=summarize(np.array(vals)[:, :2]),
            nsc_plus_given_run_minus_gaia=summarize(np.array(vals)[:, 2:])))
            for run, vals in run_impact.items()),
        sdss_run_camera_residual={f"{run}/{cam}": summarize(values)
                                  for (run, cam), values in offsets.items()},
    ))
    save_json(work / "paired_contrasts_summary.json", dict(
        paired_nmad=paired_contrasts(comparisons, config["seed"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", default="pal5")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment-a-v02.toml"))
    args = parser.parse_args()
    with args.config.open("rb") as file:
        config = tomllib.load(file)
    inputs = ["detections.csv", "external_labels.csv", "validation/gaia.csv"]
    outputs = ["paired_control_summary.json", "paired_contrasts_summary.json"]
    with run_stage(args.work, args.config, args.region, "paired-control", inputs, outputs) as execute:
        if execute:
            run(args.work, args.region, config)
