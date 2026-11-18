"""Frozen bright controls and a held-out stationary reference zero-point test."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import access
from .__main__ import detection_arrays, tomllib
from .astrometry import fit_motion, project
from .io import save_json
from .paired_control import catalogue_motion, common_catalogue_ids, published_rows, summarize
from .provenance import run_stage
from .validation import gaia_labels, groups_by_label
from .zero_point import fit_offsets


def astrometric_metrics(residual, errors, tail):
    value = np.asarray(residual)
    if not len(value):
        return {"n": 0}
    z = value / np.asarray(errors)
    return dict(**summarize(value), abs68=np.quantile(abs(value), .68, axis=0),
                abs_median=np.median(abs(value), axis=0), fraction_abs_gt_tail=np.mean(abs(value) > tail, axis=0),
                standardized_nmad=summarize(z)["nmad"], fraction_within_1sigma=np.mean(abs(z) <= 1, axis=0),
                fraction_within_2sigma=np.mean(abs(z) <= 2, axis=0))


def paired_change(before, after, seed):
    a, b = np.asarray(before), np.asarray(after)
    if len(a) < 5:
        return {"n": len(a)}
    sample = np.random.default_rng(seed).integers(0, len(a), size=(2000, len(a)))
    return dict(n=len(a), median_residual_change95=np.quantile(
        np.median(b[sample], axis=1) - np.median(a[sample], axis=1), [.025, .975], axis=0),
        abs68_change95=np.quantile(
            np.quantile(abs(b[sample]), .68, axis=1) - np.quantile(abs(a[sample]), .68, axis=1),
            [.025, .975], axis=0))


def quality_summary(values):
    return np.quantile(values, [.16, .5, .84], axis=0) if len(values) else None


def calibration_groups(frame):
    sdss = frame.survey.eq("SDSS")
    group = (frame.exposure_id.astype(str) + ":" + frame.detector_id.astype(str)).to_numpy()
    group[~sdss] = frame.exposure_id.loc[~sdss]
    return group


def calibration_support(groups, offsets):
    """Exact string-group membership without NumPy's object-array full scans.

    Calibration group names are strings. NumPy isin compares every object
    against each allowed name; hashing preserves membership in linear work.
    """
    return pd.Series(groups,copy=False).isin(list(offsets)).to_numpy()


def reference_rows(frame, xy, object_ids, gaia_xy, settings, seed):
    nsc = frame.survey.eq("NSC")
    nsc_good = nsc & object_ids.notna() & frame.magnitude.between(15, 23)
    nsc_data = pd.DataFrame(dict(ref=object_ids[nsc_good], x=xy[nsc_good, 0], y=xy[nsc_good, 1],
                                band=frame.band[nsc_good], morphology=frame.morphology[nsc_good]))
    objects = nsc_data.groupby("ref")[["x", "y"]].median()
    r_shape = nsc_data.loc[nsc_data.band.eq("r")].groupby("ref").morphology.median()
    if len(objects) < 2:
        return frame.iloc[:0].assign(reference=pd.Series(dtype=str), is_holdout=pd.Series(dtype=bool))
    points = objects.to_numpy()
    tree = cKDTree(points)
    isolated = tree.query(points, k=2)[0][:, 1] >= settings["reference_neighbour_mas"]
    extended = r_shape.reindex(objects.index).le(settings["nsc_extended_max"]).to_numpy()
    no_gaia = (cKDTree(gaia_xy).query(points)[0] > settings["gaia_veto_mas"]
               if len(gaia_xy) else np.ones(len(points), dtype=bool))
    eligible = isolated & extended & no_gaia
    sdss = frame.index[~nsc & frame.morphology.eq(3) & frame.nchild.eq(0)
                       & frame.magnitude.between(16, 22) & frame.magnitude_error.between(0, .15)]
    distances, index = tree.query(xy[sdss], k=2)
    good = (distances[:, 0] <= settings["reference_match_mas"]) & eligible[index[:, 0]]
    early = frame.loc[sdss[good]].assign(reference=objects.index[index[:, 0][good]])
    early = early.loc[~early.duplicated(["reference", "exposure_id"], keep=False)]
    names = set(early.reference)
    late = frame.loc[nsc_good & object_ids.isin(names)].assign(reference=object_ids)
    late = late.loc[~late.duplicated(["reference", "exposure_id"], keep=False)]
    refs = pd.concat([early, late])
    qualifies = refs.groupby("reference").survey.transform("nunique").eq(2)
    refs = refs.loc[qualifies].copy()
    refs["is_holdout"] = refs.reference.map(lambda value: int(
        sha256(f"{seed}:{value}".encode()).hexdigest()[:8], 16) / 2**32 < settings["holdout_fraction"])
    return refs


def calibration(work, center, config, veto_positions=None, return_references=False, arrays=None):
    settings = config["calibration"]
    full = dict(config, smoke_max_error_mas=settings["astrometric_max_error_mas"])
    full.pop("smoke_max_mag", None)
    data, xy, _, sigma = detection_arrays(work, center, full) if arrays is None else arrays
    gaia = pd.read_csv(work / "validation" / "gaia.csv").dropna(subset=["ra", "dec"])
    raw_ids = pd.read_csv(work / "external_labels.csv", dtype=str).set_index("detection_id").original_object_id
    veto = project(gaia.ra, gaia.dec, center)
    if veto_positions is not None:
        veto = np.vstack([veto, veto_positions])
    refs = reference_rows(data, xy, data.detection_id.map(raw_ids), veto, settings, config["seed"])
    groups = calibration_groups(data)
    train = refs.loc[~refs.is_holdout]
    offsets = fit_offsets(train.reference, xy[train.index], sigma[train.index], groups[train.index],
                          settings["min_references"], settings["bootstrap_draws"], config["seed"],
                          workers=int(os.environ.get("DEEP_PM_M3_WORKERS","1")))
    anchor, errors, support = {}, {}, {}
    for name, (delta, draws, n) in offsets.items():
        anchor[name] = delta
        errors[name] = draws - draws.mean(axis=0)
        support[name] = n
    # The omitted group defines a gauge. Compare offset SE relative to the
    # median calibrated NSC group instead of to a single arbitrary visit.
    if errors:
        nsc_errors = [e for name, e in errors.items() if name.startswith("NSC:")]
        datum = np.median(nsc_errors, axis=0) if nsc_errors else np.zeros_like(next(iter(errors.values())))
        errors = {name: e - datum for name, e in errors.items()}
    offsets = {name: (anchor[name], errors[name], support[name]) for name in errors
               if np.std(errors[name], axis=0, ddof=1).max() <= settings["max_offset_se_mas"]}
    held = refs.loc[refs.is_holdout & pd.Series(groups, index=data.index).isin(offsets)]
    holdout_positions = defaultdict(list)
    for _, subset in held.groupby("reference", sort=False):
        if subset.survey.nunique() != 2:
            continue
        early = subset.index[subset.survey.eq("SDSS")]
        late = subset.index[subset.survey.eq("NSC")]
        for label, pos in (("M2", xy), ("M3", None)):
            diff = [xy[idx] if pos is not None else xy[idx] - np.array([offsets[groups[i]][0] for i in idx])
                    for idx in (early, late)]
            holdout_positions[label].append(np.median(diff[0], axis=0) - np.median(diff[1], axis=0))
    holds = {k: summarize(v) for k, v in holdout_positions.items()}
    per_group = {k: dict(offset_mas=v[0], se_mas=np.std(v[1], axis=0, ddof=1), references=v[2])
                 for k, v in offsets.items() if v[2] >= 10 and k.startswith("SDSS:")}
    summary = dict(reference_counts=dict(all=refs.reference.nunique(), train=train.reference.nunique(),
                                                holdout=refs.loc[refs.is_holdout].reference.nunique()),
                         groups_before_uncertainty_cut=len(errors),
                         groups_by_survey=dict(Counter(k.split(":")[0] for k in offsets)),
                         heldout_galaxy_position_difference_mas=holds, sdss_run_camcol=per_group)
    if return_references:
        return offsets, summary, refs
    return offsets, summary


def fit_with_calibration(xy, t, sigma, epoch, offsets):
    corrections = np.array([row[0] for row in offsets])
    samples = np.stack([row[1] for row in offsets], axis=1)
    model = fit_motion(xy - corrections, t, sigma, epoch)
    w = model.weight[:, None] / sigma**2
    dt = t[:, None] - np.average(t[:, None] * np.ones((1, 2)), axis=0, weights=w)
    slopes = w * dt / np.sum(w * dt**2, axis=0)
    variance = np.var(np.sum(samples * slopes[None], axis=1), axis=0, ddof=1)
    return model, np.sqrt(model.cov[:, 1, 1] + variance)


def run(work, region, config, do_calibration):
    spec = config["regions"][region]
    center = spec["ra"], spec["dec"]
    data, xy, t, sigma = detection_arrays(work, center, config)
    labels, gaia = gaia_labels(data, xy, t, sigma, pd.read_csv(work / "validation" / "gaia.csv"),
                              center, config)
    truth = groups_by_label(labels, data, t, gaia, center, spec["radius"], config)
    original = pd.read_csv(work / "external_labels.csv", dtype=str).set_index("detection_id")
    linked, excluded = common_catalogue_ids(truth, labels, data, original.original_object_id)
    published = published_rows(access.Cache(work / "cache"), linked.values())
    offsets, zero = calibration(work, center, config) if do_calibration else ({}, {})
    settings = config["calibration"]
    baseline, baseline_errors = defaultdict(list), defaultdict(list)
    corrected, corrected_errors = defaultdict(list), defaultdict(list)
    temporal = defaultdict(list)
    surveys = data.survey.to_numpy()
    groups = calibration_groups(data)
    epoch = config["reference_epoch"]
    for source, rows in truth.items():
        late = rows[surveys[rows] == "NSC"]
        early = rows[surveys[rows] == "SDSS"]
        if np.ptp(t[late]) < config["min_seed_years"]:
            excluded["short_nsc_baseline"] += 1
            continue
        name = linked.get(source)
        if name not in published.index:
            excluded["missing_common_published"] += 1
            continue
        row = published.loc[name]
        pm = np.array([row.pmra, row.pmdec, row.pmraerr, row.pmdecerr, row.deltamjd])
        if not (np.isfinite(pm).all() and (pm[2:] > 0).all() and abs(pm[:2]).max() < 1e5):
            excluded["missing_published_motion"] += 1
            continue
        g = gaia.loc[source]
        truth_pm = g[["pm_tangent_x", "pm_tangent_y"]].to_numpy(dtype=float)
        truth_err = g[["pmra_error", "pmdec_error"]].to_numpy(dtype=float)
        baseline["published_nsc"].append(catalogue_motion(row, center) - truth_pm)
        baseline_errors["published_nsc"].append(np.hypot(pm[2:4], truth_err))
        for tag, use in (("M0", late), ("M1", np.r_[late, early[np.argmin(t[early])]]), ("M2", rows)):
            model = fit_motion(xy[use], t[use], sigma[use], epoch)
            baseline[tag].append(model.coeff[1] - truth_pm)
            baseline_errors[tag].append(np.hypot(np.sqrt(model.cov[:, 1, 1]), truth_err))
            visit = data.iloc[use]
            early_visit = visit.loc[visit.survey.eq("SDSS")]
            temporal[tag].append([np.ptp(t[use]), early_visit.exposure_id.nunique(),
                                  early_visit.observation_time.astype(int).nunique(),
                                  visit.observation_time.astype(int).nunique(),
                                  *list(1 / np.sqrt(model.cov[:, 1, 1]))])
        if do_calibration:
            use = rows[np.array([group in offsets for group in groups[rows]])]
            nsc = use[surveys[use] == "NSC"]
            if len(nsc) < 2 or np.ptp(t[nsc]) < config["min_seed_years"] or len(nsc) == len(use):
                excluded["without_calibration_coverage"] += 1
                continue
            temporal["common_supported"].append([len(use), len(rows), np.ptp(t[use]),
                                                len(use)/len(rows)])
            before = fit_motion(xy[use], t[use], sigma[use], epoch)
            after, error_after = fit_with_calibration(xy[use], t[use], sigma[use], epoch,
                                                      [offsets[name] for name in groups[use]])
            for tag, model, error in (("M2", before, np.sqrt(before.cov[:, 1, 1])),
                                      ("M3", after, error_after)):
                corrected[tag].append(model.coeff[1] - truth_pm)
                corrected_errors[tag].append(np.hypot(error, truth_err))
    stats = lambda a, b: {k: astrometric_metrics(a[k], b[k], settings["tail_masyr"]) for k in a}
    save_json(work / "precision_v03_summary.json", dict(
        baseline=stats(baseline, baseline_errors), calibration=zero, common_support=stats(corrected, corrected_errors),
        paired_m3_m2=paired_change(corrected["M2"], corrected["M3"], config["seed"]) if corrected else {},
        exclusion_counts=dict(excluded), temporal={k: quality_summary(v) for k, v in temporal.items()},
        detection_count=len(data), truth_count=len(truth), tail_threshold_masyr=settings["tail_masyr"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", required=True)
    parser.add_argument("--calibrate", action="store_true", help="test zero points in an independent region")
    parser.add_argument("--config", type=Path, default=Path("configs/precision-v03.toml"))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run on a Slurm compute node.")
    config = tomllib.loads(args.config.read_text())
    inputs = ["detections.csv", "external_labels.csv", "validation/gaia.csv"]
    for name in inputs:
        if not (args.work / name).is_file():
            parser.error(f"Missing frozen input {name}")
    with run_stage(args.work, args.config, args.region, "precision-v03", inputs=inputs,
                   outputs=["precision_v03_summary.json"]) as execute:
        if execute:
            calibrate = args.calibrate or config["regions"][args.region]["role"] == "calibration_pilot"
            run(args.work, args.region, config, calibrate)


if __name__ == "__main__":
    main()
