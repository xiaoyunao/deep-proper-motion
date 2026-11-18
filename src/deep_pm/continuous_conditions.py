"""Fixed-detection QSO time-baseline experiments on the frozen C0 graph."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from .__main__ import tomllib
from .astrometry import fit_motion, project
from .continuous_reference import FILES, PRECISION, SPATIAL, bind, read_offsets, reference_frame, verify_directory
from .exposure_catalogues import inside
from .exposure_pilot import matched_legacy
from .io import digest, save_json
from .nsc_comparison import qso_matches, comparison
from .paired_control import common_catalogue_ids
from .pilot_production import quality_detections, valid_epochs
from .precision_v03 import calibration_groups, fit_with_calibration
from .provenance import run_stage
from .public_summary import check_aggregate
from .regional_validation import official_table, valid_published


METHODS = ("NSC_only", "one_SDSS", "all_SDSS")


def experiment_indices(indices, nsc, time):
    """Nested subsets: every NSC detection is identical across all three arms."""
    late, early = indices[nsc[indices]], indices[~nsc[indices]]
    if len(late) < 2 or not len(early) or np.ptp(time[late]) <= 0:
        raise ValueError("Ablation requires two distinct NSC epochs and SDSS")
    return dict(zip(METHODS, (late, np.r_[late, early[np.argmin(time[early])]], indices)))


def time_information(time, sigma, weights, groups):
    precision = weights[:, None] / sigma**2
    mean = np.sum(precision * time[:, None], axis=0) / precision.sum(axis=0)
    leverage = precision * (time[:, None] - mean)**2
    total = leverage.sum(axis=0)
    contributions = np.stack([leverage[groups == k].sum(axis=0) for k in set(groups)])
    return dict(detections=len(time), independent_groups=len(set(groups)), span_years=float(np.ptp(time)),
                time_leverage_years2_per_mas2=total,
                effective_groups=total**2 / np.sum(contributions**2, axis=0),
                max_group_information_fraction=contributions.max(axis=0)/total,
                effective_weighted_epochs=precision.sum(axis=0)**2/(precision**2).sum(axis=0))


def summaries(records, seed):
    settings = dict(min_publish_count=3, tail_masyr=5, bootstrap_draws=400,
                    bootstrap_absolute_zero=True, seed=seed)
    output = {}
    strata = {"all": lambda r: True,
              "ordinary": lambda r: abs(r["latitude_deg"]) >= 3,
              "transition": lambda r: 1 < abs(r["latitude_deg"]) < 3,
              "repeat": lambda r: abs(r["latitude_deg"]) <= 1,
              "one_run": lambda r: r["sdss_runs"] == 1,
              "multiple_runs": lambda r: r["sdss_runs"] > 1,
              "nsc_span_lt_2yr": lambda r: r["nsc_span_years"] < 2,
              "nsc_span_ge_2yr": lambda r: r["nsc_span_years"] >= 2}
    for name, select in strata.items():
        for low in (20, 21, 22):
            chosen = [r for r in records if low <= r["magnitude"] < low+1 and select(r)]
            for cohort in ("all_eligible", "common_releases"):
                use = [r for r in chosen if cohort == "all_eligible" or r["published"] is not None and r["legacy"] is not None]
                rows = []
                for r in use:
                    row = {m: (r["motion"][m], r["error"][m]) for m in METHODS}
                    row["all_raw"] = (r["raw_motion"], r["raw_error"])
                    if cohort == "common_releases":
                        row.update(NSC=r["published"], Xiao=r["legacy"])
                    rows.append(row)
                contrasts = (("all_SDSS", "NSC_only"), ("one_SDSS", "NSC_only"),
                             ("all_SDSS", "one_SDSS"), ("all_SDSS", "all_raw"),
                             ("all_SDSS", "Xiao"), ("all_SDSS", "NSC"))
                stats = comparison(rows, settings, contrasts)
                stats["time_information"] = {m: {
                    k: np.quantile([r["timing"][m][k] for r in use], [.16,.5,.84], axis=0)
                    for k in use[0]["timing"][m]} for m in METHODS} if use else {}
                stats["interval_scope"] = "Object bootstrap; shared-observation sensitivity is separate."
                output[f"{name}__r{low}_{low+1}__{cohort}"] = stats
    return output


def run(work, source, shared, legacy, spec, seed):
    start = monotonic()
    center = reference_frame(shared/"private_c0.npz")
    config = tomllib.loads(PRECISION.read_text())
    settings = tomllib.loads(SPATIAL.read_text())
    data, xy, time, sigma, keep = quality_detections(source, center, config, settings["production"])
    keys = calibration_groups(data)
    nsc = data.survey.eq("NSC").to_numpy()
    offsets, gauges = read_offsets(shared/"private_c0.npz", with_gauge=True)
    available = pd.Series(keys[inside(data.ra_deg, data.dec_deg, spec)&keep]).map(gauges).value_counts()
    gauge = min(available.index, key=lambda k: (-available[k], k)) if len(available) else None
    offsets = {k: v for k,v in offsets.items() if gauges[k] == gauge}
    support = np.isin(keys, list(offsets))
    qso = pd.read_csv(source/"spectra_union.csv")
    qso = qso.loc[inside(qso.ra, qso.dec, spec)].reset_index(drop=True)
    points = project(qso.ra, qso.dec, center)
    labels, truth = qso_matches(data, xy, keep, points)
    original = pd.read_csv(source/"external_labels.csv", dtype=str).set_index("detection_id").original_object_id
    joint = {k: i for k,i in truth.items() if nsc[i].any() and not nsc[i].all()}
    linked, _ = common_catalogue_ids(joint, labels, data, original)
    released = official_table(source, {})
    archive = pd.read_csv(legacy)
    archive = archive.loc[inside(archive.RA, archive.DEC, spec, .06)]
    xiao, _ = matched_legacy(archive, points, center)
    records, exits = [], Counter()
    for k, indices in joint.items():
        magnitude = qso.reference_r.iloc[k]
        if not np.isfinite(magnitude) or not 20 <= magnitude < 23:
            continue
        exits["matched"] += 1
        indices = indices[support[indices]&keep[indices]]
        if not valid_epochs(indices, nsc, time, settings["production"]):
            exits["time_or_reference_support"] += 1
            continue
        sets = experiment_indices(indices, nsc, time)
        motion, errors, timing = {}, {}, {}
        for name, ix in sets.items():
            model, err = fit_with_calibration(xy[ix], time[ix], sigma[ix], config["reference_epoch"],
                                               [offsets[keys[j]] for j in ix])
            motion[name], errors[name] = model.coeff[1], err
            timing[name] = time_information(time[ix], sigma[ix], model.weight, keys[ix])
        raw = fit_motion(xy[indices], time[indices], sigma[indices], config["reference_epoch"])
        pub = valid_published(released.loc[linked[k]], center) if linked.get(k) in released.index else None
        early, late = sets["all_SDSS"][~nsc[indices]], sets["NSC_only"]
        records.append(dict(qso_index=int(k), magnitude=float(magnitude), latitude_deg=float(qso.dec.iloc[k]),
            detection_ids=data.detection_id.iloc[indices].tolist(), motion=motion, error=errors, timing=timing,
            exposures=sorted(set(data.exposure_id.iloc[indices])),
            nsc_epoch_days=sorted(set(np.floor((time[late]-2000)*365.25).astype(int))),
            sdss_runs=data.exposure_id.iloc[early].nunique(), nsc_span_years=float(np.ptp(time[late])),
            sdss_span_years=float(np.ptp(time[early])), joint_span_years=float(np.ptp(time[indices])),
            raw_motion=raw.coeff[1], raw_error=np.sqrt(raw.cov[:,1,1]),
            published=pub, legacy=xiao.get(k)))
    save_json(work/"private_conditions.json", records)
    summary = dict(frozen_baseline="13d4e29", selection="Known-position stationary spectroscopic QSO; no PM-error or morphology cut.",
                   c0_sha256=digest(shared/"private_c0.npz"), exits=exits, eligible_count=len(records),
                   cohorts=summaries(records, seed), elapsed_seconds=monotonic()-start)
    check_aggregate(summary)
    save_json(work/"conditions_summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "source", "shared", "legacy", "config"):
        parser.add_argument(name, type=Path)
    parser.add_argument("region")
    a = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use a Slurm compute node")
    a.work.mkdir(parents=True, exist_ok=True)
    bind(a.work, "source_receipt.json", verify_directory(a.source, FILES))
    bind(a.work, "reference_receipt.json", verify_directory(a.shared, ("private_c0.npz",)))
    bind(a.work, "legacy_input.csv", a.legacy)
    cfg = tomllib.loads(a.config.read_text())
    with run_stage(a.work, a.config, a.region, "continuous-conditions-v13",
                   inputs=("source_receipt.json", "reference_receipt.json", "legacy_input.csv"),
                   outputs=("private_conditions.json", "conditions_summary.json")) as go:
        if go:
            run(a.work, a.source, a.shared, a.legacy, cfg["fields"][a.region], cfg["seed"])


if __name__ == "__main__":
    main()
