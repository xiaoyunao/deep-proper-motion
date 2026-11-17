"""Freeze v0.8 batches from completed geometry receipts, before fitting motions."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from .footprints import sdss_visits
from .nsc_comparison import configuration
from .regional_cohort import load_output
from .regional_preflight import sample_cap


def reject(summary, options):
    limits = {
        "joint_fraction": options["min_joint_fraction"],
        "nsc_span_fraction": options["min_two_nsc_span_fraction"],
        "faint_qso_positions": options["min_faint_qso_positions"],
        "core_archive_candidates": options["min_core_object_count"],
    }
    reasons = [key for key, minimum in limits.items() if summary[key] < minimum]
    if summary["buffered_archive_candidates"] > options["max_archive_candidates"]:
        reasons.append("archive_budget")
    return reasons


def observing_stratum(summary):
    duration = summary["joint_nsc_span_median"]
    instrument = max(sorted(summary["nscr_exposures_by_instrument"]),
                     key=summary["nscr_exposures_by_instrument"].get)
    repeat = summary["repeated_sdss_fraction"]
    return ("short" if duration < 2 else "long",
            "repeated" if repeat >= .4 else "mostly_single", instrument)


def choose(candidates, batch_sizes, seed, frozen_first=()):
    remaining = set(candidates)
    known_runs = set()
    used_roles = Counter()
    used_conditions = Counter()
    batches = []
    for size in batch_sizes:
        batch = []
        roles = Counter()
        conditions = Counter()
        runs = set()
        for _ in range(size):
            if not remaining:
                break

            def rank(name):
                _, spec, row, run_ids = candidates[name]
                condition = observing_stratum(row)
                # Diminishing return over previously represented conditions, with
                # distinct physical SDSS runs favored over neighboring caps.
                return (sum(run not in known_runs for run in run_ids) > 0,
                        len(run_ids & runs) == 0,
                        -roles[spec["role"]], -conditions[condition],
                        -used_roles[spec["role"]], -used_conditions[condition],
                        -sum(row["nscr_exposures_by_instrument"].values()),
                        sha256(f"{seed}/{name}".encode()).hexdigest())

            name = frozen_first[len(batch)] if not batches and len(batch)<len(frozen_first) else max(remaining, key=rank)
            geometry_sha, spec, row, run_ids = candidates[name]
            condition = observing_stratum(row)
            roles[spec["role"]] += 1
            conditions[condition] += 1
            used_roles[spec["role"]] += 1
            used_conditions[condition] += 1
            known_runs.update(run_ids)
            runs.update(run_ids)
            remaining.remove(name)
            batch.append(dict(name=name, **spec, **row, geometry_sha256=geometry_sha,
                              condition=list(condition), sdss_run_count=len(run_ids)))
        batches.append(batch)
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("window", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/spatial-v08.toml"))
    parser.add_argument("--frozen-first", nargs="*", default=[])
    args = parser.parse_args()
    _, settings = configuration(args.config)
    options = settings["regional"]
    spatial = settings["spatial"]
    candidates = {}
    rejected = Counter()
    stage = "regional-preflight-" + settings["run_series"]
    for name, spec in settings["fields"].items():
        work = args.root / name
        try:
            row = load_output(work, stage, "regional_preflight_summary.json")
        except (FileNotFoundError, KeyError, ValueError):
            rejected["missing_completed_geometry"] += 1
            continue
        reasons = reject(row, options)
        if reasons:
            rejected.update(reasons)
            continue
        center = spec["ra"], spec["dec"]
        size = spec["radius"]
        xy, sky, _ = sample_cap(center, size, options["preflight_points"], options["seed"])
        visits = sdss_visits(args.window, center, size, rerun=301, vectorized=True)
        runs = {visit.exposure for visit in visits if visit.coverage(xy, sky).any()}
        candidates[name] = (sha256((work / "regional_preflight_summary.json").read_bytes()).hexdigest(),
                            spec, row, runs)
    frozen = args.frozen_first
    if (len(frozen)>spatial["batch_sizes"][0] or len(frozen)!=len(set(frozen))
            or set(frozen)-candidates.keys()):
        parser.error("Previously frozen fields must still have valid geometry receipts.")
    batches = choose(candidates, spatial["batch_sizes"], options["seed"], frozen_first=frozen)
    print(json.dumps(dict(candidate_count=len(settings["fields"]), eligible_count=len(candidates),
                          rejected=rejected, batches=batches), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
