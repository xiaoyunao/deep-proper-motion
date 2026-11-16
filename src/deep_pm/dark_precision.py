"""Fixed training associations and common held-out exposures for faint sources."""
import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.spatial import cKDTree

from . import access
from .__main__ import detection_arrays, tomllib
from .associate import static_tracks
from .astrometry import fit_motion, project
from .io import save_json
from .paired_control import catalogue_motion, common_catalogue_ids, published_rows
from .precision_v03 import astrometric_metrics, calibration, calibration_groups, fit_with_calibration, paired_change
from .provenance import run_stage


def bin_name(magnitude, edges):
    j = np.searchsorted(edges, magnitude, side="right")
    if j == 0:
        return f"lt{edges[0]:g}"
    if j == len(edges):
        return f"ge{edges[-1]:g}"
    return f"{edges[j-1]:g}_to_{edges[j]:g}"


def query_unique(tree, positions, gates, margin):
    distance, nearest = tree.query(positions, k=2)
    good = (distance[:, 0] <= gates) & (distance[:, 1]-distance[:, 0] > margin)
    duplicate = pd.Series(nearest[good, 0]).duplicated(keep=False).to_numpy()
    good[np.flatnonzero(good)[duplicate]] = False
    return good, nearest[:, 0]


def metrics(rows, tail):
    return {k: astrometric_metrics([r[0] for r in v], [r[1] for r in v], tail)
            for k, v in rows.items()}


def run(work, config, region):
    spec, settings = config["regions"][region], config["dark"]
    center = (spec["ra"], spec["dec"])
    qso_path = work / "spectroscopic_qso.csv"
    qso = pd.read_csv(qso_path) if qso_path.exists() else None
    qso_xy = project(qso.ra, qso.dec, center) if qso is not None else np.zeros((0,2))
    offsets, calibration_info = calibration(work, center, config, qso_xy)
    full = dict(config, smoke_max_error_mas=config["calibration"]["astrometric_max_error_mas"])
    full.pop("smoke_max_mag", None)
    data, xy, t, sigma = detection_arrays(work, center, full)
    groups = calibration_groups(data)
    epoch = config["reference_epoch"]
    nsc = data.survey.eq("NSC").to_numpy()
    r_band = data.band.eq(settings["reference_band"]).to_numpy()
    magnitude = data.magnitude.to_numpy()
    snr_cut = data.magnitude_error.between(0, 2.5/(np.log(10)*settings["min_snr"]),
                                            inclusive="right").to_numpy()
    good = r_band & snr_cut & pd.Series(groups).isin(offsets).to_numpy()
    year = settings["held_year"]
    lower, upper = Time([f"{year}-01-01", f"{year+1}-01-01"], scale="utc").tt.jyear
    is_held = (t >= lower) & (t < upper)
    train = np.flatnonzero(good & ~is_held)
    hold = np.flatnonzero(good & nsc & is_held)
    visits = data.exposure_id.to_numpy()
    gaia = pd.read_csv(work / "validation" / "gaia.csv")
    veto = cKDTree(project(gaia.ra, gaia.dec, center))
    tracks = static_tracks(xy[train], visits[train], 700)
    control = []
    for indices in tracks:
        idx = train[list(indices)]
        late = idx[nsc[idx]]
        if (len(late) < 3 or len(late) == len(idx) or np.ptp(t[late]) < config["min_seed_years"]
                or np.nanmedian(data.morphology.iloc[late]) < .80):
            continue
        before = fit_motion(xy[idx], t[idx], sigma[idx], epoch)
        if (np.linalg.norm(before.coeff[0]) > spec["radius"]*3600000
                or veto.query(before.coeff[0])[0] <= 1500):
            continue
        ref_r = np.median(magnitude[late])
        tag = bin_name(ref_r, settings["r_edges"])
        after, _ = fit_with_calibration(xy[idx], t[idx], sigma[idx], epoch,
                                        [offsets[k] for k in groups[idx]])
        control.append((tag, before, after, idx))
    counts = defaultdict(lambda: dict(tracks=0, evaluated_track_visits=0, unique_matches=0))
    for tag, *_ in control:
        counts[tag]["tracks"] += 1
    residual = defaultdict(list)
    residual_by_track = defaultdict(list)
    diagnostics = defaultdict(list)
    distinct = defaultdict(set)
    visits_by_bin = defaultdict(set)
    fwhm = data["fwhm"].to_numpy()
    magerr = data.magnitude_error.to_numpy()
    scale = 2.5/np.log(10)
    for visit, rows in data.iloc[hold].groupby("exposure_id"):
        if not control or len(rows) < 2:
            continue
        idx = rows.index.to_numpy()
        moment = np.median(t[idx])
        predict = np.array([before.predict(moment) for _, before, _, _ in control])
        for tag, *_ in control:
            counts[tag]["evaluated_track_visits"] += 1
        active, nearest = query_unique(cKDTree(xy[idx]), predict, settings["gate_mas"],
                                       settings["ambiguity_margin_mas"])
        for j in np.flatnonzero(active):
            tag, before, after, support = control[j]
            row = idx[nearest[j]]
            counts[tag]["unique_matches"] += 1
            distinct[tag].add(j)
            visits_by_bin[tag].add(visit)
            delta = offsets[groups[row]]
            # Use a common, position-only match gate for both model residuals.
            # Infer projected prediction covariance on the same detections.
            for name, model, correction in [("M2", before, 0), ("M3", after, delta[0])]:
                vector = np.array([1., t[row]-model.epoch])
                prediction = vector @ model.coeff
                variance = vector @ model.cov @ vector
                if name == "M3":
                    w = model.weight[:, None] / sigma[support]**2
                    dt = t[support, None] - np.average(np.tile(t[support, None], (1,2)), axis=0, weights=w)
                    leverage = w/w.sum(axis=0) + (t[row]-np.average(
                        np.tile(t[support, None], (1,2)), axis=0, weights=w))*w*dt/(w*dt**2).sum(axis=0)
                    sampled = np.stack([offsets[g][1] for g in groups[support]], axis=1)
                    variance += np.var(delta[1] - np.sum(sampled*leverage[None],axis=1),axis=0,ddof=1)
                position_residual = xy[row]-correction-prediction
                residual[tag+"/"+name].append((position_residual, np.sqrt(variance+sigma[row]**2)))
                residual_by_track[tag, name, j].append(position_residual)
                if name == "M3":
                    norm = float(np.linalg.norm(position_residual))
                    if np.isfinite(fwhm[row]) and fwhm[row] > 0:
                        diagnostics[tag, j].append((float(fwhm[row]), scale/magerr[row], norm))
    qso_stats = defaultdict(list)
    qso_names = {}
    qso_targets = {}
    visits_by_qso = defaultdict(list)
    for _, frame in data.loc[good].groupby("exposure_id"):
        if not len(qso_xy) or len(frame) < 2:
            continue
        idx = frame.index.to_numpy()
        unique, closest = query_unique(cKDTree(xy[idx]), qso_xy, 500, 300)
        for j in np.flatnonzero(unique):
            visits_by_qso[j].append(idx[closest[j]])
    for key, rows in visits_by_qso.items():
        idx = np.array(rows)
        late = idx[nsc[idx]]
        if len(late)<3 or len(late)==len(idx) or np.ptp(t[late]) < config["min_seed_years"]:
            continue
        qso_names[key] = idx
        base = fit_motion(xy[idx], t[idx], sigma[idx], epoch)
        change, uncertainty = fit_with_calibration(xy[idx], t[idx], sigma[idx], epoch,
                                                    [offsets[g] for g in groups[idx]])
        qso_stats["M2"].append((base.coeff[1], np.sqrt(base.cov[:, 1, 1])))
        qso_stats["M3"].append((change.coeff[1], uncertainty))
        qso_targets[key] = (qso_stats["M2"][-1], qso_stats["M3"][-1],
                            np.median(magnitude[late]))
    original = pd.read_csv(work / "external_labels.csv", dtype=str).set_index("detection_id").original_object_id
    labels = np.full(len(data), -1)
    for key, idx in qso_names.items():
        labels[idx] = key
    linked, exits = common_catalogue_ids(qso_names, labels, data, original)
    qso_common = defaultdict(list)
    common_qso = {}
    published = published_rows(access.Cache(work / "cache"), linked.values())
    for key, name in linked.items():
        if name not in published.index:
            exits["missing_common_published"] += 1
            continue
        row = published.loc[name]
        error = row[["pmraerr", "pmdecerr"]].to_numpy(dtype=float)
        pm = catalogue_motion(row, center)
        if not (np.isfinite(pm).all() and np.isfinite(error).all() and (error > 0).all()
                and abs(pm).max() < 1e5 and row.deltamjd > 0):
            exits["missing_published_motion"] += 1
            continue
        m2, m3, rmag = qso_targets[key]
        common_qso[key] = (m2, m3, (pm, error))
        for tag in ("all", bin_name(rmag, settings["r_edges"])):
            for label, value in (("M2", m2), ("M3", m3), ("published_nsc", (pm, error))):
                qso_common[tag+"/"+label].append(value)

    ground_path = work / "legacy_ground.npy"
    ground_comparison = {}
    if ground_path.is_file() and common_qso:
        ground = np.load(ground_path, allow_pickle=False)
        if np.any(ground["PM_SOURCE"] != b"DESI"):
            raise ValueError("Ground comparison requires PM_SOURCE=DESI")
        common = list(common_qso)
        ground_rows = defaultdict(list)
        candidates = project(ground["RA"], ground["DEC"], center)
        good_match, positions = query_unique(cKDTree(candidates), qso_xy[common], 500, 300)
        for i in np.flatnonzero(good_match):
            fields = ground[positions[i]]
            row = SimpleNamespace(ra=fields["RA"], dec=fields["DEC"],
                                  pmra=fields["PMRA"], pmdec=fields["PMDEC"])
            pm = catalogue_motion(row, center)
            errors = np.array([fields["PMRA_ERR"], fields["PMDEC_ERR"]])
            if not (np.isfinite(pm).all() and np.isfinite(errors).all() and (errors > 0).all()
                    and abs(pm).max() < 1e5):
                continue
            for model, values in zip(("M2", "M3", "published_nsc", "Xiao2025_DESI"),
                                     (*common_qso[common[i]], (pm, errors))):
                ground_rows["all/"+model].append(values)
        ground_comparison = dict(range_rows=len(ground), common_with_nsc=len(common),
                                 with_position_match=int(good_match.sum()),
                                 metrics=metrics(ground_rows, config["calibration"]["tail_masyr"]))

    paired = {}
    for tag, tracks_in_bin in distinct.items():
        per_model = [[np.median(residual_by_track[tag, name, j], axis=0)
                      for j in sorted(tracks_in_bin)] for name in ("M2", "M3")]
        paired[tag] = paired_change(*per_model, config["seed"])
    quality = {}
    for tag in distinct:
        per_track = np.array([np.median(values, axis=0) for (band, _), values in diagnostics.items()
                              if tag == band])
        if len(per_track) < 10:
            continue
        quality[tag] = {}
        for i, name in enumerate(("fwhm_catalogue_units", "snr")):
            middle = np.median(per_track[:, i])
            groups_at_cut = per_track[:, i] < middle
            quality[tag][name] = dict(split=middle,
                lower_count=int(groups_at_cut.sum()), upper_count=int((~groups_at_cut).sum()),
                lower_median_radial_mas=float(np.median(per_track[groups_at_cut, 2])) if groups_at_cut.any() else None,
                upper_median_radial_mas=float(np.median(per_track[~groups_at_cut, 2])))
    save_json(work/"dark_precision_summary.json", dict(
        held_year=settings["held_year"],
        coverage_by_r={k: dict(v, matched_tracks=len(distinct[k]), distinct_matched_visits=len(visits_by_bin[k]))
                                                       for k,v in counts.items()},
        residual_by_r_mas=metrics(residual, 300), qso_coverage=dict(queried=len(qso_xy),common=len(qso_names)),
        qso_motion_masyr=metrics(qso_stats, config["calibration"]["tail_masyr"]),
        qso_published_common=dict(exclusion_counts=dict(exits), ground_catalogue=ground_comparison,
                                  metrics=metrics(qso_common, config["calibration"]["tail_masyr"])),
        paired_object_median_change_mas=paired, detection_quality_control=quality,
        reference_counts=calibration_info["reference_counts"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("--region", default="test_equator")
    p.add_argument("--config", type=Path, default=Path("configs/precision-v03.toml"))
    args = p.parse_args()
    inputs = ["detections.csv", "external_labels.csv", "validation/gaia.csv", "spectroscopic_qso.csv"]
    if (args.work / "legacy_ground.npy").is_file():
        inputs.append("legacy_ground.npy")
    with run_stage(args.work,args.config,args.region,"dark-precision-v03",inputs=inputs,
                   outputs=["dark_precision_summary.json"]) as run_now:
        if run_now:
            run(args.work,tomllib.loads(args.config.read_text()),args.region)


if __name__ == "__main__":
    main()
