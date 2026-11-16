"""Inject trajectories in actual exposure footprints over real catalogue background."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from . import access
from .__main__ import detection_arrays, tomllib
from .associate import motion_tracks, static_tracks
from .astrometry import epoch_halo_deg, seasons, unproject
from .footprints import nsc_visits, sdss_visits
from .io import save_json
from .provenance import run_stage
from .validation import truth_requirements


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


def injection_pass(members, requirements, fraction):
    track = set(members)
    sdss, nsc, early, groups = requirements
    s, n = len(track & sdss), len(track & nsc)
    return (s >= fraction * len(sdss) and n >= fraction * len(nsc)
            and s + n >= fraction * len(track) and bool(track & early)
            and all(track & group for group in groups))


def evaluate_batch(batch, xy, t, sigma, exposures, surveys, config):
    injected = [row for _, draws in batch for row in draws]
    positions = np.vstack([xy, np.array([row[3] for row in injected])])
    times = np.r_[t, [row[0] for row in injected]]
    uncertainty = np.vstack([sigma, [row[4] for row in injected]])
    visit_ids = np.r_[exposures, [row[1] for row in injected]]
    survey = np.r_[surveys, [row[2] for row in injected]]
    order = np.argsort(times, kind="stable")
    reverse = np.empty_like(order)
    reverse[order] = np.arange(len(order))
    positions, times, uncertainty, visit_ids, survey = (a[order] for a in
                                                        (positions, times, uncertainty, visit_ids, survey))
    fixed = static_tracks(positions, visit_ids, config["fixed_radius_mas"])
    nsc_like = static_tracks(positions, visit_ids, config["fixed_radius_mas"], config["dbscan_eps_mas"])
    audit = {}
    selected, candidates, _, _ = motion_tracks(positions, times, uncertainty, visit_ids, fixed, config, audit)
    offset = len(xy)
    trial_indices = {}
    for index, (_, draws) in enumerate(batch):
        trial_indices[index] = reverse[offset:offset + len(draws)]
        offset += len(draws)
    required = truth_requirements(trial_indices, survey, times, config["min_season_days"])
    result = []
    for index, (speed, _) in enumerate(batch):
        def matches(group):
            return injection_pass(group, required[index], config["recovery_fraction"])

        result.append((speed, dict(
            generated_correct_candidate=any(matches(g) for g in audit["generated"]),
            retained_correct_candidate=any(matches(g.indices) for g in candidates),
            fixed_recovery=any(matches(g) for g in fixed),
            nsc_like_recovery=any(matches(g) for g in nsc_like),
            selected_recovery=any(matches(g.indices) for g in selected),
            secure_recovery=any(matches(g.indices) and g.status == "secure" for g in selected))))
    return result


def run(work, region, config, sdss_window, stress, trials, workers=1):
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
                queued.append((speed, [(v.year, v.exposure, v.survey, p, s) for v, p, s in draws]))
    rng.shuffle(queued)
    batch_size = 6
    jobs = [queued[start:start + batch_size] for start in range(0, len(queued), batch_size)]
    evaluate = partial(evaluate_batch, xy=xy, t=t, sigma=sigma, exposures=exposure,
                       surveys=data.survey.to_numpy(), config=config)
    if workers == 1:
        for job in jobs:
            for speed, row in evaluate(job):
                count[speed].update(row)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for trials_in_batch in pool.map(evaluate, jobs):
                for speed, row in trials_in_batch:
                    count[speed].update(row)
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
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be at least one")
    with args.config.open("rb") as file:
        config = tomllib.load(file)
    config.update(config["regions"][args.region].get("overrides", {}))
    mode = "stress" if args.stress else "injection"
    window_link = args.work / "injection_sdss_window.fits"
    if not window_link.is_file():
        window_link.symlink_to(args.sdss_window.resolve(strict=True))
    if window_link.resolve() != args.sdss_window.resolve(strict=True):
        raise ValueError("A different SDSS window is already bound to this run")
    with run_stage(args.work, args.config, args.region, f"{mode}-{args.trials}",
                   ["detections.csv", window_link.name], [f"v02_{mode}_summary.json"]) as execute:
        if execute:
            run(args.work, args.region, config, args.sdss_window, args.stress, args.trials, args.workers)
