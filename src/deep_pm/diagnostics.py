"""Identify candidate-generation versus conflict failures and inspect catalogue residuals."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import access
from .__main__ import detection_arrays, tomllib
from .associate import Track, components
from .astrometry import fit_motion
from .io import save_json
from .validation import gaia_labels, groups_by_label


def run(work, config, region):
    spec = config["regions"][region]
    center, radius = (spec["ra"], spec["dec"]), spec["radius"]
    data, xy, t, sigma = detection_arrays(work, center, config)
    raw = pd.read_csv(work / "validation" / "gaia.csv")
    labels, gaia = gaia_labels(data, xy, t, sigma, raw, center, config)
    truth = groups_by_label(labels, data, t, gaia, center, radius, config)
    raw_candidates = json.loads((work / "candidates.json").read_text())
    candidates = [Track(tuple(row["indices"]), row["score"], row["status"], row["groups"])
                  for row in raw_candidates]
    selected = json.loads((work / "tracks.json").read_text())["motion"]
    selected_keys = {tuple(row["indices"]) for row in selected if len(row["indices"]) > 1}
    component_size = {}
    for group, _ in components(candidates):
        for k in group:
            component_size[k] = len(group)
    matches = {key: [] for key in truth}
    for i, track in enumerate(candidates):
        index = np.asarray(track.indices)
        if data.survey.iloc[index].nunique() < 2:
            continue
        tagged = labels[index]
        unique = set(tagged[tagged >= 0])
        if len(unique) != 1:
            continue
        key = unique.pop()
        if key in matches:
            mine = index[tagged == key]
            coverage = len(mine) / len(truth[key])
            if data.survey.iloc[mine].nunique() == 2:
                matches[key].append((coverage, i))
    failures = Counter()
    cases = []
    residuals = []
    metadata = data.copy()
    where = access.q3c(*center, radius + 1.45)
    cache = access.Cache(work / "cache")
    exp = cache.checked("nsc", "exposure,instrument,airmass", "nsc_dr2.exposure", where)
    exp["exposure_id"] = "NSC:" + exp.exposure
    ex = exp.drop_duplicates("exposure_id").set_index("exposure_id")
    nsc = metadata.survey.eq("NSC")
    metadata.loc[nsc, "airmass"] = metadata.loc[nsc, "exposure_id"].map(ex.airmass)
    metadata.loc[nsc, "instrument"] = metadata.loc[nsc, "exposure_id"].map(ex.instrument)
    labels_file = pd.read_csv(work / "external_labels.csv", dtype=str).set_index("detection_id")
    metadata["nsc_id"] = metadata.detection_id.map(labels_file.original_object_id)
    nsc_truth = metadata.loc[nsc & (labels >= 0), ["nsc_id"]].copy()
    nsc_truth["label"] = labels[nsc_truth.index]
    nsc_objects = nsc_truth.groupby("nsc_id").label.nunique()
    truth_objects = nsc_truth.groupby("label").nsc_id.nunique()
    ext = dict(labelled_nsc_objects=len(nsc_objects), mixed_nsc_objects=int((nsc_objects > 1).sum()),
               labelled_gaia_with_nsc=len(truth_objects),
               gaia_split_across_nsc_ids=int((truth_objects > 1).sum()))
    for case_no, (key, idx) in enumerate(truth.items(), 1):
        best = max(matches[key], default=(0., -1))
        retained = max([cover for cover, k in matches[key] if candidates[k].indices in selected_keys],
                       default=0.)
        target = gaia.loc[key, ["pm_tangent_x", "pm_tangent_y"]].to_numpy(dtype=float)
        target_pm_band = int(np.searchsorted(config["pm_edges"], np.linalg.norm(target)))
        stage = "recovered"
        if retained < config["recovery_fraction"]:
            if best[0] < config["recovery_fraction"]:
                stage = "no_adequate_candidate"
            elif all(component_size[k] > config["max_component"]
                     for cover, k in matches[key] if cover >= config["recovery_fraction"]):
                stage = "adequate_candidates_in_oversize_components"
            else:
                stage = "adequate_candidate_not_selected"
        failures[stage] += 1
        cases.append(dict(case=case_no, pm_band_index=target_pm_band, stage=stage, best=best[0],
                          retained=retained, component_size=component_size.get(best[1], 0),
                          labelled_members=len(idx)))
        model = fit_motion(xy[idx], t[idx], sigma[idx], config["reference_epoch"])
        subset = metadata.iloc[idx].copy()
        subset["east"], subset["north"] = model.residual.T
        subset["case"] = case_no
        residuals.append(subset)
    out = dict(candidates=len(candidates), conflict_component_max=max(component_size.values(), default=0),
               failure_stages=dict(failures), cases=cases, external_nsc_labels=ext,
               nscccd_instrument_counts={str(a): int(b) for a, b in
                                         metadata.loc[nsc].instrument.value_counts().items()})
    if not residuals:
        save_json(work / "diagnostics_summary.json", out)
        return
    rows = pd.concat(residuals)
    rows.to_csv(work / "validation" / "detection_residuals.csv", index=False)
    columns = ["east", "north"]
    out["residual_by_survey_band"] = {}
    for (survey, band), part in rows.groupby(["survey", "band"]):
        values = part[columns].to_numpy()
        median = np.median(values, axis=0)
        out["residual_by_survey_band"][f"{survey}/{band}"] = dict(
            samples=len(part), sources=part.case.nunique(), median=median,
            nmad=1.4826 * np.median(np.abs(values - median), axis=0))
    save_json(work / "diagnostics_summary.json", out)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), layout="constrained")
    for (survey, band), group in rows.groupby(["survey", "band"]):
        name = f"{survey} {band} (n={len(group)})"
        if group.airmass.notna().any():
            axes[0].scatter(group.airmass, group.north, label=name, s=8, alpha=0.3)
        axes[1].scatter(group.magnitude, group.east, label=name, s=8, alpha=0.3)
    axes[0].set(xlabel="Airmass", ylabel="North residual (mas)")
    axes[1].set(xlabel="Native-band magnitude", ylabel="East residual (mas)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{region} | residuals about a linear fit to {len(truth)} Gaia-linked sources")
    fig.savefig(work / "summary" / f"{region}_residuals_summary.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", default="smoke")
    parser.add_argument("--config", default="configs/experiment-a.toml")
    args = parser.parse_args()
    with open(args.config, "rb") as file:
        config = tomllib.load(file)
    run(args.work, config, args.region)
