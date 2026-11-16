"""M3/M4/M5 on common support with separate references and visit holdout."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from .__main__ import detection_arrays, tomllib
from .astrometry import fit_motion, project
from .chromatic import chromatic_fit, directions, stable_color, templates
from .io import save_json
from .paired_control import summarize
from .precision_v03 import astrometric_metrics, calibration_groups, fit_with_calibration, reference_rows
from .provenance import run_stage
from .validation import gaia_labels, groups_by_label
from .zero_point import fit_offsets


def aggregate(values):
    return {key: summarize(value) for key, value in values.items()}


def run(work, region, config):
    spec = config["regions"][region]
    cfg = config["calibration"]
    center = (spec["ra"], spec["dec"])
    full = dict(config, smoke_max_error_mas=cfg["astrometric_max_error_mas"])
    full.pop("smoke_max_mag", None)
    frame, xy, t, sigma = detection_arrays(work, center, full)
    gaia_input = pd.read_csv(work / "validation" / "gaia.csv")
    ids = pd.read_csv(work / "external_labels.csv", dtype=str).set_index("detection_id").original_object_id
    refs = reference_rows(frame, xy, frame.detection_id.map(ids),
                          project(gaia_input.ra, gaia_input.dec, center), cfg, config["seed"])
    metadata = pd.read_csv(work / "nsc_visit_metadata.csv")
    inst_band, direction = directions(frame, metadata)
    groups = calibration_groups(frame)
    colors = refs.reference.map(stable_color(refs, refs.reference))
    refs = refs.assign(c=colors).loc[colors.notna() & np.isfinite(direction[refs.index]).all(axis=1)]
    train = refs.loc[~refs.is_holdout]
    held_groups = {g for g in set(groups) if int(sha256(str(g).encode()).hexdigest()[:8], 16) % 5 == 0}
    is_visit_holdout = pd.Series(groups).isin(held_groups).to_numpy()
    available = train.loc[~is_visit_holdout[train.index]]
    counts = available.assign(inst=inst_band, visit=groups[available.index]).groupby("inst").agg(
        sources=("reference", "nunique"), visits=("visit", "nunique"))
    levels = counts.index[(counts.sources >= 20) & (counts.visits >= 4)].to_list()
    status = {}
    if "sdss/r" not in levels or len(levels) < 2:
        save_json(work / "chromatic_summary.json", dict(
            status={"M4": "insufficient_colour_support", "M5": "insufficient_colour_support"},
            exposure_metadata=counts.to_dict("index"), models={}, comparison={},
            coverage=dict(reference_objects=len(set(refs.reference)))))
        return
    refs = refs.loc[inst_band.loc[refs.index].isin(levels)]
    train = refs.loc[~refs.is_holdout]
    rows = train.index.to_numpy()
    fitting = ~is_visit_holdout[rows]
    all_correction = {"M3": (np.zeros_like(xy), None, None)}
    draws = cfg["bootstrap_draws"]
    color_anchor = float(train.drop_duplicates("reference").c.median())
    for tag, use_dcr in [("M4", False), ("M5", True)]:
        design, names = templates(inst_band.iloc[rows].to_numpy(), direction[rows],
                                  train.c.to_numpy()-color_anchor, levels, use_dcr)
        try:
            beta, sampled, cond = chromatic_fit(train.reference.to_numpy()[fitting],
                                                groups[rows][fitting], xy[rows][fitting],
                                                sigma[rows][fitting], design[fitting],
                                                draws, config["seed"])
        except (ValueError, RuntimeError) as exc:
            status[tag] = str(exc)
            continue
        correction = np.zeros_like(xy)
        correction[rows] = design @ beta
        all_correction[tag] = (correction, beta, sampled)
        status[tag] = dict(condition=cond, terms={name: [float(b), float(std)] for name, b, std in
                         zip(names, beta, sampled.std(axis=0, ddof=1))})

    offsets = {}
    for tag, (extra, _, _) in all_correction.items():
        offsets[tag] = fit_offsets(train.reference, (xy-extra)[rows], sigma[rows], groups[rows],
                                   cfg["min_references"], draws, config["seed"])
    supported = set.intersection(*(set(v) for v in offsets.values()))
    for off in offsets.values():
        if not off:
            continue
        gauge = np.median([v[1] for k, v in off.items() if k.startswith("NSC:")], axis=0)
        for group in tuple(supported):
            if np.std(off[group][1]-gauge, axis=0, ddof=1).max() > cfg["max_offset_se_mas"]:
                supported.remove(group)

    tag_stats = defaultdict(list)
    epoch_checks = defaultdict(list)
    held = refs.loc[refs.is_holdout & pd.Series(groups, index=frame.index).isin(supported)]
    for _, group in held.groupby("reference"):
        idx = group.index.to_numpy()
        early = group.survey.eq("SDSS").to_numpy()
        hold = is_visit_holdout[idx]
        if not early.any() or early.all():
            continue
        for tag, (_, beta, _) in all_correction.items():
            corrected = xy[idx] - np.array([offsets[tag][g][0] for g in groups[idx]])
            if beta is not None:
                features, _ = templates(inst_band.iloc[idx].to_numpy(), direction[idx],
                                         group.c.to_numpy()-color_anchor, levels, tag == "M5")
                corrected -= features @ beta
            tag_stats[tag].append(np.median(corrected[early], axis=0)-np.median(corrected[~early], axis=0))
            if hold.any() and (~hold).any():
                epoch_checks[tag].append(np.median(corrected[hold], axis=0)-np.median(corrected[~hold], axis=0))

    # Fix the bright Gaia control and evaluate each model on identical visits.
    data, bx, bt, bs = detection_arrays(work, center, config)
    labels, gaia = gaia_labels(data, bx, bt, bs, gaia_input, center, config)
    truth = groups_by_label(labels, data, bt, gaia, center, spec["radius"], config)
    full_index = pd.Series(frame.index, index=frame.detection_id)
    stellar, errs = defaultdict(list), defaultdict(list)
    covered = []
    for key, index in truth.items():
        selected = data.iloc[index]
        col = stable_color(selected, pd.Series(1, index=selected.index))
        if col.empty:
            continue
        indices = selected.detection_id.map(full_index).to_numpy(dtype=int)
        mask = (pd.Series(groups[indices]).isin(supported).to_numpy()
                & inst_band.iloc[indices].isin(levels).to_numpy()
                & np.isfinite(direction[indices]).all(axis=1))
        idx = indices[mask]
        late = idx[frame.survey.iloc[idx].eq("NSC").to_numpy()]
        if len(late) < 2 or len(late) == len(idx) or np.ptp(t[late]) < config["min_seed_years"]:
            continue
        covered.append(len(idx) / len(indices))
        src = gaia.loc[key]
        target = src[["pm_tangent_x", "pm_tangent_y"]].to_numpy(dtype=float)
        error = src[["pmra_error", "pmdec_error"]].to_numpy(dtype=float)
        for tag, (_, beta, sample) in all_correction.items():
            visits = [offsets[tag][key] for key in groups[idx]]
            if beta is not None:
                matrix, _ = templates(inst_band.iloc[idx].to_numpy(), direction[idx],
                                      np.full(len(idx), col.iloc[0]-color_anchor), levels, tag == "M5")
                extra = matrix @ beta
                covariance_draw = np.einsum("iap,dp->dia", matrix, sample-sample.mean(axis=0))
                visits = [(row[0]+extra[j], row[1]+covariance_draw[:,j], row[2])
                          for j, row in enumerate(visits)]
            model, formal = fit_with_calibration(xy[idx], t[idx], sigma[idx], config["reference_epoch"], visits)
            stellar[tag].append(model.coeff[1]-target)
            errs[tag].append(np.hypot(formal, error))

    models = {tag: astrometric_metrics(v, errs[tag], cfg["tail_masyr"]) for tag, v in stellar.items()}
    save_json(work / "chromatic_summary.json", dict(
        status=status, exposure_metadata=counts.to_dict("index"), models=models,
        comparison=dict(heldout_galaxy_survey_offset=aggregate(tag_stats),
                        heldout_galaxy_visit_offset=aggregate(epoch_checks)),
        coverage=dict(reference_objects=refs.reference.nunique(), trained_objects=train.reference.nunique(),
                      calibrated_groups_by_survey=dict(Counter(x.split(":")[0] for x in supported)),
                      coefficient_holdout_groups=len(supported & held_groups),
                      bright_truth=len(truth), common_stars=len(covered),
                      median_visit_fraction=float(np.median(covered)) if covered else None)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/precision-v03.toml"))
    p.add_argument("--region", default="test_equator")
    args = p.parse_args()
    inputs = ("detections.csv", "external_labels.csv", "validation/gaia.csv", "nsc_visit_metadata.csv")
    if any(not (args.work/name).is_file() for name in inputs):
        p.error("Missing frozen detections or exact visit metadata")
    with run_stage(args.work, args.config, args.region, "chromatic-v03", inputs=inputs,
                   outputs=["chromatic_summary.json"]) as execute:
        if execute:
            run(args.work, args.region, tomllib.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
