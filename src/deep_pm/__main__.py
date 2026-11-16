"""Experiment A command line."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import pandas as pd

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from . import access, normalize
from .associate import motion_tracks, static_tracks
from .astrometry import epoch_halo_deg, fit_motion, project, seasons, unproject, years
from .io import save_json
from .provenance import run_stage


def in_cone(frame, center, radius):
    xy = project(frame.ra, frame.dec, center)
    return frame.loc[np.linalg.norm(xy, axis=1) <= 3600000 * np.tan(np.deg2rad(radius))
                     * 180 / np.pi].copy()


def fetch(work, center, radius, config, route):
    cache = access.Cache(work / "cache")
    halo = epoch_halo_deg(config)
    expanded = radius + halo
    if route in {"object", "all"}:
        raw, obj = access.fetch_object_index(cache, *center, expanded + halo)
        in_cone(raw, center, expanded).to_csv(work / "nsc_object.csv", index=False)
        print(f"NSC object index: {len(obj)} objects, {len(raw)} measurements before cone trim", flush=True)
    if route in {"exposure", "all"}:
        chips = access.covering_chips(cache, *center, expanded,
                                       config["access"]["chip_center_margin_deg"])
        raw = access.fetch_exposure_index(cache, chips, *center, expanded)
        in_cone(raw, center, expanded).to_csv(work / "nsc_exposure.csv", index=False)
        chips.to_csv(work / "chips.csv", index=False)
    if route in {"sdss", "all"}:
        raw = access.fetch_sdss(cache, *center, expanded)
        in_cone(raw, center, expanded).to_csv(work / "sdss.csv", index=False)


def fetch_gaia(work, center, radius, config):
    cache = access.Cache(work / "validation" / "cache")
    where = access.q3c(*center, radius + 2 * epoch_halo_deg(config) + 0.004)
    columns = ("source_id,ra,dec,ref_epoch,pmra,pmdec,pmra_error,pmdec_error,parallax,"
               "ruwe,phot_g_mean_mag,phot_rp_mean_mag,parallax_error,radial_velocity,astrometric_params_solved")
    table = cache.checked("nsc", columns, "gaia_dr3.gaia_source", where)
    table.to_csv(work / "validation" / "gaia.csv", index=False)


def standardize(work, config, center, radius, route):
    frames = []
    for survey, name, adapter in [
        ("SDSS", "sdss.csv", normalize.sdss), ("NSC", f"nsc_{route}.csv", normalize.nsc),
    ]:
        path = work / name
        if path.exists():
            d = pd.read_csv(path, low_memory=False)
            if survey == "NSC" and (work / "chips.csv").exists():
                meta = pd.read_csv(work / "chips.csv")[["exposure", "ccdnum", "instrument", "airmass"]]
                d = d.merge(meta, on=["exposure", "ccdnum"], how="left", validate="many_to_one")
            frames.append(adapter(d))
    if not frames:
        raise ValueError("No raw detection files found")
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["observation_time", "detection_id"]).reset_index(drop=True)
    xy = project(data.ra_deg, data.dec_deg, center)
    data["in_core"] = np.linalg.norm(xy, axis=1) <= np.tan(np.deg2rad(radius)) * 180 / np.pi * 3600000
    # Remove index labels from the physical input to association.
    data[["detection_id", "original_object_id"]].to_csv(work / "external_labels.csv", index=False)
    data.drop(columns="original_object_id").to_csv(work / "detections.csv", index=False)
    if (work / "nsc_object.csv").exists() and (work / "nsc_exposure.csv").exists():
        left, right = (set(pd.read_csv(work / f"nsc_{r}.csv", usecols=["measid"]).measid)
                       for r in ("object", "exposure"))
        save_json(work / "access_comparison.json", dict(shared=len(left & right),
                                                       object_only=len(left - right),
                                                       exposure_only=len(right - left)))


def detection_arrays(work, center, config):
    raw = pd.read_csv(work / "detections.csv", low_memory=False)
    needed = ["ra_deg", "dec_deg", "observation_time", "sigma_east_mas", "sigma_north_mas"]
    good = np.isfinite(raw[needed].to_numpy(dtype=float)).all(axis=1)
    limits = raw[["sigma_east_mas", "sigma_north_mas"]].max(axis=1)
    if "smoke_max_error_mas" in config:
        good &= limits <= config["smoke_max_error_mas"]
    if "smoke_max_mag" in config:
        good &= raw.magnitude.le(config["smoke_max_mag"]) | raw.magnitude.isna()
    data = raw.loc[good].reset_index(drop=True)
    xy = project(data.ra_deg, data.dec_deg, center)
    time = years(data.observation_time)
    sigma = np.hypot(data[["sigma_east_mas", "sigma_north_mas"]].to_numpy(dtype=float),
                     config["position_floor_mas"])
    return data, xy, time, sigma


def run_association(work, center, config):
    data, xy, t, sigma = detection_arrays(work, center, config)
    exposure = data.exposure_id.to_numpy()
    collected = {}
    start = time.monotonic()
    for method, eps in [("fixed", None), ("nsc_like", config["dbscan_eps_mas"])]:
        groups = static_tracks(xy, exposure, config["fixed_radius_mas"], eps)
        collected[method] = [dict(indices=list(idx), score=float(np.log(len(idx))),
                                  status="unmatched" if len(idx) == 1 else "baseline")
                             for idx in groups]
    fixed = [tuple(track["indices"]) for track in collected["fixed"]]
    # Save baseline products even when the explicit motion budget is exceeded.
    save_json(work / "tracks.json", collected)
    audit = {}
    selected, candidates, unmatched, summary = motion_tracks(xy, t, sigma, exposure, fixed, config, audit)
    collected["motion"] = [vars(track) for track in selected]
    collected["motion"].extend(dict(indices=[i], score=0, status="unmatched") for i in unmatched)
    save_json(work / "candidates.json", [vars(track) for track in candidates])
    save_json(work / "association_audit.json", audit)
    save_json(work / "tracks.json", collected)
    save_json(work / "association_resources.json", dict(summary, seconds=time.monotonic() - start,
                                                        maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def run_fits(work, center, radius, config):
    data, xy, t, sigma = detection_arrays(work, center, config)
    result = []
    tracks = json.loads((work / "tracks.json").read_text())
    epoch = config["reference_epoch"]
    for method, group in tracks.items():
        for track_id, track in enumerate(group):
            i = np.array(track["indices"])
            if len(i) < 2 or np.ptp(t[i]) < config["min_seed_years"]:
                continue
            weighted = fit_motion(xy[i], t[i], sigma[i], epoch, robust=False)
            model = fit_motion(xy[i], t[i], sigma[i], epoch)
            ra, dec = unproject(model.coeff[0], center)[0]
            result.append(dict(method=method, track=track_id, status=track["status"],
                               epoch=epoch, ra=ra, dec=dec,
                               in_core=np.linalg.norm(model.coeff[0]) <= np.tan(np.deg2rad(radius)) *
                               180 / np.pi * 3600000,
                               pm_x=model.coeff[1, 0], pm_y=model.coeff[1, 1],
                               wls_pm_x=weighted.coeff[1, 0], wls_pm_y=weighted.coeff[1, 1],
                               cov_x=model.cov[0].tolist(), cov_y=model.cov[1].tolist(),
                               median_weight=float(np.median(model.weight)), n=len(i), rejected=0,
                               downweighted=int((model.weight < 0.5).sum()), baseline=model.span,
                               chi2=model.chi2, groups=len(set(seasons(t[i], config["min_season_days"]))),
                               weighted_time_leverage=float(1 / model.cov[:, 1, 1].sum())))
    save_json(work / "fits.json", result)


def inventory(work, config, center, radius):
    raw = pd.read_csv(work / "detections.csv", low_memory=False)
    _, xy, t, sigma = detection_arrays(work, center, config)
    info = {}
    for name, survey in raw.groupby("survey"):
        valid = survey.observation_time.dropna()
        epoch = years(valid.to_numpy())
        info[name] = dict(detections=len(survey), core_detections=int(survey.in_core.sum()),
                          exposures=survey.exposure_id.nunique(), nights=int(np.floor(valid).nunique()),
                          years={str(int(k)): int(v) for k, v in
                                 pd.Series(np.floor(epoch)).value_counts().sort_index().items()},
                          groups=len(set(seasons(epoch, config["min_season_days"]))) if len(epoch) else 0,
                          bands={str(k): int(v) for k, v in survey.band.value_counts().items()},
                          missing_time=int(survey.observation_time.isna().sum()),
                          missing_error=int(survey[["sigma_east_mas", "sigma_north_mas"]].isna().any(axis=1).sum()))
    summary = dict(area_sq_deg=2 * np.pi * (1 - np.cos(np.deg2rad(radius))) * (180 / np.pi) ** 2,
                   counts=info, usable_for_association=len(t))
    if (work / "access_comparison.json").exists():
        summary["nsc_index_comparison"] = json.loads((work / "access_comparison.json").read_text())
    save_json(work / "inventory.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment-a.toml")
    parser.add_argument("--work", type=Path, default=os.environ.get("DEEP_PM_WORK"))
    parser.add_argument("--region", default="smoke")
    parser.add_argument("--route", choices=["object", "exposure", "sdss", "gaia", "all"], default="object")
    parser.add_argument("command", choices=["discover", "fetch", "normalize", "inventory", "associate",
                                            "fit", "validate", "report"])
    args = parser.parse_args()
    if args.work is None:
        parser.error("Give --work or DEEP_PM_WORK for a specific NERSC Scratch run directory")
    if args.command != "discover" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run the experiment in a Slurm compute allocation")
    config_path = Path(args.config)
    with config_path.open("rb") as f:
        config = tomllib.load(f)
    region = config["regions"][args.region]
    config.update(region.get("overrides", {}))
    center, radius = (region["ra"], region["dec"]), region["radius"]
    work = args.work / args.region
    work.mkdir(parents=True, exist_ok=True)
    if args.command == "discover":
        for root in ("HOME", "PSCRATCH", "SCRATCH"):
            print(root, os.environ.get(root))
        for command in (["showquota"], ["sacctmgr", "-nP", "show", "associations",
                                        f"where", f"user={os.environ.get('USER')}", "format=Account,QOS"]):
            subprocess.run(command, check=False)
        return
    stage = args.command
    route = args.route
    fetch_products = {"sdss": ["sdss.csv"], "object": ["nsc_object.csv"],
                      "exposure": ["nsc_exposure.csv", "chips.csv"], "gaia": ["validation/gaia.csv"]}
    association = ["tracks.json", "candidates.json", "association_resources.json", "association_audit.json"]
    inputs, outputs = {
        "fetch": ([], fetch_products.get(route, sum(list(fetch_products.values())[:3], []))),
        "normalize": (["sdss.csv", f"nsc_{route}.csv", "chips.csv"],
                      ["detections.csv", "external_labels.csv", "access_comparison.json"]),
        "inventory": (["detections.csv"], ["inventory.json"]),
        "associate": (["detections.csv"], association),
        "fit": (["detections.csv", "tracks.json"], ["fits.json"]),
        "validate": (["detections.csv", "validation/gaia.csv", "validation/challenge_source.txt", "fits.json", *association],
                     ["validation.json", "validation/paired.csv", "validation/recovery_trace.json"]),
        "report": (["inventory.json", "validation.json", "association_resources.json",
                    "validation/paired.csv"], [f"summary/{args.region}_summary.json"]),
    }[stage]
    entry = f"fetch-{route}" if stage == "fetch" else stage
    with run_stage(work, config_path, args.region, entry, inputs, outputs) as execute:
        if not execute:
            print(f"{args.region} {entry} already complete and verified")
            return
        if stage == "fetch":
            if route == "gaia":
                fetch_gaia(work, center, radius, config)
            else:
                fetch(work, center, radius, config, route)
        elif stage == "normalize":
            standardize(work, config, center, radius, route)
        elif stage == "inventory":
            inventory(work, config, center, radius)
        elif stage == "associate":
            run_association(work, center, config)
        elif stage == "fit":
            run_fits(work, center, radius, config)
        elif stage == "validate":
            from .validation import validate
            validate(work, center, config)
        elif stage == "report":
            from .reporting import report
            report(work, args.region, region["role"])


if __name__ == "__main__":
    main()
