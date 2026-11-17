"""Strict same-source NSC DR2 comparisons for ordinary and faint sources."""
import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import access
from .__main__ import detection_arrays, tomllib
from .astrometry import fit_motion, project
from .dark_precision import bin_name, query_unique
from .io import save_json
from .paired_control import catalogue_motion, common_catalogue_ids, published_rows
from .precision_v03 import astrometric_metrics, calibration, calibration_groups, fit_with_calibration
from .provenance import run_stage
from .validation import gaia_labels, groups_by_label


def absolute68(data, axis=0):
    norm = np.linalg.norm(data, axis=-1)
    return np.quantile(np.concatenate([abs(data), norm[..., None]], axis=-1), .68, axis=axis)


def comparison(rows, settings, additional_contrasts=()):
    """Use identical resampled objects for every contrast."""
    n = len(rows)
    if n < settings["min_publish_count"]:
        return dict(n=n)
    motion = {k: np.array([row[k][0] for row in rows]) for k in rows[0]}
    uncertainty = {k: np.array([row[k][1] for row in rows]) for k in rows[0]}
    tail = settings["tail_masyr"]
    summaries = {}
    for k, residual in motion.items():
        summary = astrometric_metrics(residual, uncertainty[k], tail)
        norm = np.linalg.norm(residual, axis=1)
        summary.update(radial_abs68=np.quantile(norm, .68), radial_median=np.median(norm),
                       radial_tail_count=int(np.count_nonzero(norm > tail)),
                       axis_tail_count=np.count_nonzero(abs(residual) > tail, axis=0))
        summaries[k] = summary
    resample = np.random.default_rng(settings["seed"]).integers(0, n, (settings["bootstrap_draws"], n))
    boot = {k: absolute68(v[resample], axis=1) for k, v in motion.items()}
    if settings.get("bootstrap_absolute_zero"):
        for key, residual in motion.items():
            summaries[key]["median_bias95"] = np.quantile(
                np.median(residual[resample], axis=1), [.025, .975], axis=0).T
    contrasts = {}
    standard = (("M1", "published_nsc"), ("M2", "published_nsc"), ("M3", "published_nsc"),
                 ("M1", "M0"), ("M2", "M1"), ("M3", "M2"),
                 ("M3_blind", "published_nsc"), ("M3_blind", "M3"))
    for a, b in standard + tuple(additional_contrasts):
        if a not in boot or b not in boot:
            continue
        ref = absolute68(motion[b])
        eligible = (boot[b] > 0).all(axis=0) & (ref > 0)
        fraction = [float(v) if ok else None
                    for v, ok in zip(np.divide(ref-absolute68(motion[a]), ref,
                                               where=ref>0, out=np.zeros(3)), eligible)]
        bounds = [np.quantile(1-boot[a][:,j]/boot[b][:,j], [.025,.975]) if ok else None
                  for j, ok in enumerate(eligible)]
        bias_draw = np.median(motion[a][resample], axis=1) - np.median(motion[b][resample], axis=1)
        contrasts[f"{a}_vs_{b}"] = dict(
            absolute68_delta95=np.quantile(boot[a]-boot[b], [.025, .975], axis=0).T,
            median_bias_delta95=np.quantile(bias_draw, [.025, .975], axis=0).T,
            fraction_improved=fraction, fraction95=bounds)
    return dict(n=n, methods=summaries, paired=contrasts)


def fit_methods(index, xy, t, sigma, nsc, epoch, offsets=None):
    late = index[nsc[index]]
    early = index[~nsc[index]]
    first = early[np.argmin(t[early])]
    fits = {}
    for name, used in (("M0", late), ("M1", np.r_[late, first]), ("M2", index)):
        fit = fit_motion(xy[used], t[used], sigma[used], epoch)
        fits[name] = (fit.coeff[1], np.sqrt(fit.cov[:, 1, 1]))
    if offsets is not None:
        fit, error = fit_with_calibration(xy[index], t[index], sigma[index], epoch, offsets)
        fits["M3"] = (fit.coeff[1], error)
    return fits


def qso_matches(data, xy, flags, qso_xy):
    from scipy.spatial import cKDTree

    labels = np.full(len(data), -1)
    visits = defaultdict(list)
    for _, frame in data.loc[flags].groupby("exposure_id", sort=False):
        idx = frame.index.to_numpy()
        good, closest = query_unique(cKDTree(xy[idx]), qso_xy, 500, 300)
        for j in np.flatnonzero(good):
            row = idx[closest[j]]
            labels[row] = j
            visits[j].append(row)
    return labels, {j: np.asarray(idx) for j, idx in visits.items()}


def visit_influence(index, xy, t, sigma, nsc, epoch, offsets, keys, fit, span):
    """Test the largest single-group slope leverage per survey."""
    w = 1 / sigma[index].mean(axis=1)**2
    leverage = w * abs(t[index]-np.average(t[index], weights=w))
    scores = pd.Series(leverage).groupby(keys[index]).sum()
    shifts = []
    for prefix in ("SDSS:", "NSC:"):
        survey_keys = scores[scores.index.str.startswith(prefix)]
        if survey_keys.empty:
            shifts.append(None)
            continue
        rows = index[keys[index] != survey_keys.idxmax()]
        late = rows[nsc[rows]]
        if len(late) < 2 or len(late) == len(rows) or np.ptp(t[late]) < span:
            shifts.append(None)
            continue
        second, _ = fit_with_calibration(xy[rows], t[rows], sigma[rows], epoch,
                                         [offsets[k] for k in keys[rows]])
        shifts.append(float(np.linalg.norm(second.coeff[1]-fit)))
    return shifts


def run(work, name, spec, config, settings, kind):
    center = spec["ra"], spec["dec"]
    data_config = dict(config)
    if kind == "qso":
        data_config.pop("smoke_max_mag")
        data_config["smoke_max_error_mas"] = config["calibration"]["astrometric_max_error_mas"]
    data, xy, t, sigma = detection_arrays(work, center, data_config)
    nsc = data.survey.eq("NSC").to_numpy()
    keys = calibration_groups(data)
    epoch = config["reference_epoch"]
    span = config["min_seed_years"]
    available = Counter()
    if kind == "qso":
        all_qso = pd.read_csv(work / "spectroscopic_qso.csv")
        candidates = all_qso.loc[all_qso.in_core].reset_index(drop=True)
        qso_xy = project(candidates.ra, candidates.dec, center)
        available["spectra"] = len(candidates)
        magerr = data.magnitude_error
        good = data.band.eq("r") & magerr.between(0, 2.5/(np.log(10)*5), inclusive="right")
        labels, truth = qso_matches(data, xy, good, qso_xy)
        offsets, info = calibration(work, center, config, project(all_qso.ra, all_qso.dec, center))
    else:
        qso_xy = np.zeros((0, 2))
        labels, gaia = gaia_labels(data, xy, t, sigma, pd.read_csv(work / "validation" / "gaia.csv"),
                                  center, config)
        truth = groups_by_label(labels, data, t, gaia, center, spec["radius"], config)
        offsets, info = calibration(work, center, config)
    available["matched_candidates"] = len(truth)
    original = pd.read_csv(work / "external_labels.csv", dtype=str).set_index("detection_id").original_object_id
    linked, exclusion = common_catalogue_ids(truth, labels, data, original)
    published = published_rows(access.Cache(work / "cache"), linked.values())
    models, strict = [], []
    temporal, nested = [], defaultdict(list)
    influence = defaultdict(list)
    individual = []
    known = np.isin(keys, list(offsets))
    for source, index in truth.items():
        late = index[nsc[index]]
        if not len(late) or len(late) == len(index) or np.ptp(t[late]) < span:
            exclusion["insufficient_survey_span"] += 1
            continue
        available["both_surveys"] += 1
        obj = linked.get(source)
        if obj not in published.index:
            exclusion["no_unique_published_match"] += 1
            continue
        row = published.loc[obj]
        pm, error = catalogue_motion(row, center), row[["pmraerr", "pmdecerr"]].to_numpy(dtype=float)
        if not (np.isfinite(pm).all() and np.isfinite(error).all() and (error > 0).all()
                and abs(pm).max() < 1e5 and row.deltamjd > 0):
            exclusion["invalid_published_motion"] += 1
            continue
        reference, ref_error = (np.zeros(2), np.zeros(2))
        if kind == "bright":
            g = gaia.loc[source]
            reference = g[["pm_tangent_x", "pm_tangent_y"]].to_numpy(dtype=float)
            ref_error = g[["pmra_error", "pmdec_error"]].to_numpy(dtype=float)
        published_model = {"published_nsc": (pm-reference, np.hypot(error, ref_error))}
        full = fit_methods(index, xy, t, sigma, nsc, epoch)
        models.append(dict(published_model, **{k:(v[0]-reference, np.hypot(v[1],ref_error))
                                                for k,v in full.items()}))
        use = index[known[index]]
        late = use[nsc[use]]
        if len(late) < 2 or len(late) == len(use) or np.ptp(t[late]) < span:
            exclusion["insufficient_m3_support"] += 1
            continue
        estimates = fit_methods(use, xy, t, sigma, nsc, epoch, [offsets[k] for k in keys[use]])
        result = dict(published_model, **{k:(v[0]-reference, np.hypot(v[1], ref_error))
                                          for k,v in estimates.items()})
        strict.append(result)
        early = use[~nsc[use]]
        base = np.ptp(t[late])
        temporal.append([base, np.ptp(t[use]), len(set(keys[early])), len(set(keys[late])), len(use)/len(index)])
        for label, shift in zip(("remove_sdss", "remove_nsc"), visit_influence(
                use, xy, t, sigma, nsc, epoch, offsets, keys, estimates["M3"][0], span)):
            if shift is not None:
                influence[label].append(shift)
        if kind == "qso":
            r = float(candidates.loc[source,"reference_r"])
            choices = [bin_name(r, settings["reference_r_edges"]),
                       "short_baseline" if base < settings["short_nsc_baseline_years"] else "long_baseline"]
            if 21 <= r < 23:
                choices.append("faint_21_to_23")
            for group in choices:
                nested[group].append(result)
            individual.append((r, base, result))
    info.pop("sdss_run_camcol", None)
    if kind == "qso":
        save_json(work/"private_paired_qso.json", individual)
    save_json(work/"nsc_comparison_summary.json", dict(
        sample_kind=kind, counts=dict(available), exclusions=dict(exclusion),
        strict=comparison(strict, settings), without_support_requirement=comparison(models, settings),
        by_group={k:comparison(v,settings) for k,v in nested.items()},
        timing_p16_median_p84=np.quantile(temporal,[.16,.5,.84],axis=0) if temporal else None,
        observation_group_influence={k:dict(n=len(v), median_shift_masyr=float(np.median(v)))
                                     for k,v in influence.items() if len(v)>=settings["min_publish_count"]},
        calibration=info))


def configuration(path):
    settings = tomllib.loads(path.read_text())
    config = tomllib.loads(path.with_name(settings["frozen_model"]).read_text())
    return config, settings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("region")
    p.add_argument("kind", choices=("qso", "bright"))
    p.add_argument("--config", type=Path, default=Path("configs/precision-v04.toml"))
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Run on a Slurm compute node.")
    config, settings = configuration(args.config)
    spec = (settings["fields"] if args.kind=="qso" else config["regions"])[args.region]
    inputs = ["detections.csv", "external_labels.csv", "validation/gaia.csv"]
    outputs = ["nsc_comparison_summary.json"]
    if args.kind == "qso":
        inputs.append("spectroscopic_qso.csv")
        outputs.append("private_paired_qso.json")
    with run_stage(args.work,args.config,args.region,"nsc-compare-v04",inputs=inputs, outputs=outputs) as go:
        if go:
            run(args.work,args.region,spec,config,settings,args.kind)


if __name__ == "__main__":
    main()
