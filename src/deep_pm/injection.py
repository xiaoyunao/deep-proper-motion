"""Inject trajectories in actual exposure footprints over real catalogue background."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import access
from .__main__ import detection_arrays, tomllib
from .associate import motion_tracks, static_tracks
from .astrometry import epoch_halo_deg, seasons, unproject
from .footprints import nsc_visits, sdss_visits
from .io import save_json
from .provenance import run_stage


def simulation(visits, center, config, origin, velocity, magnitude, noise, rng):
    geometrical = []
    detections = []
    for visit in visits:
        position = origin + velocity * (visit.year - config["reference_epoch"])
        sky = unproject(position, center)[0]
        if not visit.coverage(position, sky):
            continue
        geometrical.append(visit)
        # The geometry count precedes the separately configurable detection model.
        detection_p = config.get("injection_detection_probability", {}).get(visit.survey, .8)
        if rng.random() >= detection_p:
            continue
        error = noise.get(visit.exposure, noise.get(f"{visit.survey}/{visit.band}", np.array([100., 100.])))
        scatter = error * 10 ** (.1 * (magnitude - 19.0))
        detections.append((visit, position + rng.normal(0, scatter), scatter))
    return geometrical, detections


def required_visits(detections, config):
    if not detections:
        return False
    t = np.array([visit.year for visit in detections])
    return ({visit.survey for visit in detections} == {"SDSS", "NSC"}
            and len(set(seasons(t, config["min_season_days"]))) >= 3
            and np.ptp(t) >= config["min_seed_years"])


def injection_pass(members, truth, required):
    track = set(members)
    return len(track & truth) >= required * max(len(track), len(truth))


def run(work, region, config, sdss_window, stress, trials):
    spec = config["regions"][region]
    center, radius = (spec["ra"], spec["dec"]), spec["radius"]
    if stress:
        config = dict(config, mu_max=config["stress_mu_max"])
    required_halo = epoch_halo_deg(config)
    data, xy, t, sigma = detection_arrays(work, center, config)
    sampling = np.linalg.norm(xy, axis=1) / 3600000
    declared_halo = spec.get("background_halo_deg", config["access"]["halo_deg"])
    if declared_halo + 1e-8 < required_halo:
        raise ValueError(f"Re-fetch background with halo {required_halo:.5f} deg before injection")
    data = data.loc[sampling <= radius + declared_halo].reset_index(drop=True)
    xy, t, sigma = (a[sampling <= radius + declared_halo] for a in (xy, t, sigma))
    chips = access.covering_chips(access.Cache(work / "cache"), *center, radius + required_halo,
                                 config["access"]["chip_center_margin_deg"])
    visits = [*nsc_visits(chips, center), *sdss_visits(sdss_window, center, radius + required_halo)]
    measurements = data.assign(sx=sigma[:, 0], sy=sigma[:, 1])
    noise = {name: group[["sx", "sy"]].median().to_numpy()
             for name, group in measurements.groupby("exposure_id")}
    noise.update({f"{survey}/{band}": group[["sx", "sy"]].median().to_numpy()
                  for (survey, band), group in measurements.groupby(["survey", "band"])})
    speeds = [config["stress_mu_max"]] if stress else config["pm_edges"]
    rng = np.random.default_rng(config["seed"])
    count = {speed: Counter() for speed in speeds}
    queued = []
    exposure = data.exposure_id.to_numpy()
    for speed in speeds:
        for _ in range(trials):
            stats = count[speed]
            stats["attempted"] += 1
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            origin = direction * (np.sqrt(rng.random()) * radius * 3600000)
            theta = rng.uniform(0, 2 * np.pi)
            velocity = speed * np.array([np.cos(theta), np.sin(theta)])
            geometric, draws = simulation(visits, center, config, origin, velocity, rng.uniform(18, 21),
                                         noise, rng)
            stats["two_survey_geometry"] += {v.survey for v in geometric} == {"SDSS", "NSC"}
            if required_visits([v for v, *_ in draws], config):
                stats["required_time_coverage"] += 1
                queued.append((speed, draws))
    rng.shuffle(queued)
    batch_size = 6
    for start in range(0, len(queued), batch_size):
        batch = queued[start:start + batch_size]
        injected = [row for _, draws in batch for row in draws]
        positions = np.vstack([xy, np.array([row[1] for row in injected])])
        times = np.r_[t, [row[0].year for row in injected]]
        uncertainty = np.vstack([sigma, [row[2] for row in injected]])
        visit_ids = np.r_[exposure, [row[0].exposure for row in injected]]
        order = np.argsort(times, kind="stable")
        reverse = np.empty_like(order)
        reverse[order] = np.arange(len(order))
        positions, times, uncertainty, visit_ids = (a[order] for a in
                                                     (positions, times, uncertainty, visit_ids))
        fixed = static_tracks(positions, visit_ids, config["fixed_radius_mas"])
        nsc_like = static_tracks(positions, visit_ids, config["fixed_radius_mas"], config["dbscan_eps_mas"])
        audit = {}
        selected, candidates, _, _ = motion_tracks(positions, times, uncertainty, visit_ids, fixed, config, audit)
        offset = len(xy)
        for speed, draws in batch:
            truth = set(reverse[offset:offset + len(draws)])
            offset += len(draws)
            def matches(group):
                return injection_pass(group, truth, config["recovery_fraction"])
            stats = count[speed]
            stats["generated_correct_candidate"] += any(matches(g) for g in audit["generated"])
            stats["retained_correct_candidate"] += any(matches(g.indices) for g in candidates)
            stats["fixed_recovery"] += any(matches(g) for g in fixed)
            stats["nsc_like_recovery"] += any(matches(g) for g in nsc_like)
            stats["selected_recovery"] += any(matches(g.indices) for g in selected)
            stats["secure_recovery"] += any(matches(g.indices) and g.status == "secure" for g in selected)
    keys = ("attempted", "two_survey_geometry", "required_time_coverage", "generated_correct_candidate",
            "retained_correct_candidate", "fixed_recovery", "nsc_like_recovery", "selected_recovery",
            "secure_recovery")
    result = {str(speed): {key: count[speed][key] for key in keys} for speed in speeds}
    save_json(work / ("v02_stress_summary.json" if stress else "v02_injection_summary.json"),
              dict(role=spec["role"], footprint_visits=dict(Counter(v.survey for v in visits)),
                   halo_deg=required_halo, bins=result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", default="smoke")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment-a-v02.toml"))
    parser.add_argument("--sdss-window", required=True, type=Path)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    with args.config.open("rb") as file:
        config = tomllib.load(file)
    config.update(config["regions"][args.region].get("overrides", {}))
    mode = "stress" if args.stress else "injection"
    with run_stage(args.work, args.config, args.region, f"{mode}-{args.trials}",
                   ["detections.csv"], [f"v02_{mode}_summary.json"]) as execute:
        if execute:
            run(args.work, args.region, config, args.sdss_window, args.stress, args.trials)
