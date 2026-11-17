"""Evaluate expansion batches and distinguish zero-point bias from statistical support."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from .io import save_json
from .nsc_cohort import overlap_from_visits, visit_ids
from .nsc_comparison import comparison, configuration
from .provenance import run_stage
from .regional_cohort import load_output, resampling, summarize
from .regional_validation import CONTRASTS, tag_magnitude
from .validation import ratio


def zero_status(n, zero, minimum_count, blocks, minimum_blocks, limit):
    intervals = None if zero is None else np.asarray(zero["zero95_masyr"])
    if intervals is None:
        return dict(n=n, observation_blocks=blocks, adequate_support=False,
                    half_width_masyr=None, significant_axes=[], exceeds_limit_axes=[], certified=False)
    narrow = np.diff(intervals, axis=1).ravel() / 2
    significant = [axis for axis, (lower, upper) in zip(("east", "north"), intervals)
                   if lower > 0 or upper < 0]
    supported = n >= minimum_count and blocks >= minimum_blocks
    return dict(n=n, observation_blocks=blocks, adequate_support=supported,
                half_width_masyr=narrow.tolist(), significant_axes=significant,
                exceeds_limit_axes=[axis for axis, (lower, upper) in zip(("east", "north"), intervals)
                                   if lower > limit or upper < -limit],
                certified=bool(supported and np.all(np.abs(intervals) <= limit)))


def yield_and_prediction(fields):
    counts = defaultdict(Counter)
    for data in fields.values():
        holdout = data["holdout"]
        if holdout["status"] != "completed":
            continue
        for sample, bins in holdout["by_r"].items():
            for band, row in bins.items():
                item = counts[sample, band]
                item.update(eligible=row["train_sources"], matched=row["matched_sources"]["n"],
                        near_r_detections=row["near_r_detections"], fields=1)
            if "chip_geometry" in row:
                item.update(row["chip_geometry"])
                item.update(geometry_train_sources=row["train_sources"])
    result = defaultdict(dict)
    for (sample, band), item in counts.items():
        result[sample][band] = dict(item, matched_fraction=ratio(item["matched"], item["eligible"]))
    return result


def gaia_cohort(directories, blocks, settings):
    samples=defaultdict(lambda:defaultdict(list))
    totals={}
    for directory in directories:
        if settings["fields"][directory.name].get("series")!="v08":
            continue
        pairs=load_output(directory,"regional-production-v08","private_gaia_pairs.json")
        totals[directory.name]=len(pairs)
        for mag,obs in pairs:
            for tag in tag_magnitude(mag,settings):
                samples[tag][directory.name].append(obs)
    results={}
    for tag,tables in samples.items():
        rows=[r for part in tables.values() for r in part]
        groups=[[r for name in block for r in tables.get(name,[])] for block in blocks]
        groups=[v for v in groups if v]
        results[tag]=dict(observation_blocks=len(groups),statistics=comparison(rows,settings,CONTRASTS),
                          w0_vs_nsc_block_bootstrap=resampling(groups,settings,"W0"))
    return dict(per_field_common=totals,pooled=results)


def coverage_checks(fields, matrix):
    verified = []
    for name, row in fields.items():
        if name not in matrix:
            continue
        geometry = matrix[name]
        archive = row["acquisition"]
        if (archive["objects"] != geometry["buffered_archive_candidates"]
                or min(archive["measurements"], archive["sdss_measurements"]) <= 0):
            raise ValueError(f"{name}: checked archive counts disagree with the locked coverage.")
        verified.append(geometry)
    return dict(new_fields=len(verified),
                sampled_joint_deg2=sum(v["area_sqdeg"]*v["joint_fraction"] for v in verified),
                sampled_joint_nsc_span_deg2=sum(v["area_sqdeg"]*v["nsc_span_fraction"] for v in verified))


def assess(directories, settings, previous, overlap, matrix):
    product = summarize(directories, settings, previous, observation_overlap=overlap)
    rules = settings["regional"]
    spatial = settings["spatial"]
    data = product["pooled"].get("22_to_23", {}).get("W0", {})
    all_count = data.get("statistics", {}).get("n", 0)
    sample = product["aggregate_qso_exits"].get("W0_22_to_23", {})
    fields = product["fields"]
    diagnostics = {}
    field_limit = rules["quality_zero_limit"]
    for name, row in fields.items():
        faint = row["faint_precision"].get("n", 0)
        diagnostics[name] = dict(conditional_zero=zero_status(
            faint, row["zero95"], rules["quality_min_faint_qso"], int(faint > 0), 1, field_limit),
            w0_final_fraction=ratio(row["faint_w0"].get("final", 0), row["faint_w0"].get("initial", 0)),
            historical_gate_exits=row["quality_exits"])
    for directory in directories:
        spec=settings["fields"][directory.name]
        if spec.get("series")=="v08":
            regional=load_output(directory,"regional-production-v08","regional_summary.json")
            fields[directory.name]["coverage"] = regional["coverage"]
            fields[directory.name]["astrometry"] = regional["astrometry"]
    gain = (data.get("block_bootstrap") or {}).get("p68_gain95")
    stats = data.get("statistics", {})
    delta = stats.get("paired", {}).get("W0_vs_published_nsc", {}).get("fraction_improved")
    replication = bool(gain is not None and gain[0] > 0 and delta is not None
                       and delta[2] >= settings["practical_gain"])
    product.update(spatial_quality=dict(
        geometric_coverage=coverage_checks(fields,matrix),
        conditional_by_field=diagnostics,
        pooled_zero=zero_status(all_count, data.get("block_bootstrap"),
                                rules["quality_min_faint_qso"], data.get("observation_blocks", 0),
                                spatial["min_independent_blocks_for_high_precision"], spatial["zero_limit_masyr"]),
        strict_faint_w0_final=ratio(sample.get("final", 0), sum(sample.values())),
        reproducible_precision_gain=replication,
        whole_year_prediction=yield_and_prediction(fields),
        gaia=gaia_cohort(directories,product["observation_blocks"],settings)))
    return product


def link_input(work, relative, target):
    link = work / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve(strict=True)
    if not link.exists():
        link.symlink_to(target)
    if link.resolve() != target:
        raise ValueError("Use a fresh aggregation directory for different inputs.")
    return str(relative)


def visit_snapshot(directory, cache):
    source=(directory/"detections.csv").resolve(strict=True)
    stat=source.stat()
    identity=[str(source),stat.st_ino,stat.st_size,stat.st_mtime_ns]
    key=sha256(json.dumps(identity).encode()).hexdigest()
    path=cache/(key+".json")
    if not path.is_file():
        visits=visit_ids(directory)
        current=source.stat()
        if (current.st_size,current.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):
            raise ValueError(f"{directory.name}: detections changed while recording exposure IDs.")
        save_json(path,{k:sorted(v) for k,v in visits.items()})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("cohort")
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--comparison", nargs="*", type=Path, default=[])
    parser.add_argument("--config", type=Path, default=Path("configs/spatial-v08.toml"))
    parser.add_argument("--matrix", type=Path, default=Path("configs/spatial-v08-fields.toml"))
    parser.add_argument("--visit-workers",type=int,default=4)
    parser.add_argument("--batches", nargs="*", type=int, default=[])
    parser.add_argument("--include-old", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use a Slurm compute node.")
    from .__main__ import tomllib

    work = args.work / args.cohort
    work.mkdir(parents=True, exist_ok=True)
    _, settings = configuration(args.config)
    past_config = args.config.with_name("regional-v07.toml")
    previous = configuration(past_config)[1]
    matrix = tomllib.loads(args.matrix.read_text())
    fields = dict(previous["fields"], **{
        name: dict(settings["fields"][name], series="v08") for name in matrix["fields"]})
    if args.include_old:
        directories = [p.parent for p in sorted(args.old_root.glob("*/regional_summary.json"))]
    else:
        directories = []
    directories.extend(args.new_root / k for k, spec in matrix["fields"].items()
                       if spec["batch"] in args.batches)
    if not directories or any(not (p / "regional_summary.json").is_file() for p in directories):
        parser.error("All locked cores in the requested cohort must have completed production.")
    if args.visit_workers<1:
        parser.error("Set at least one visit reader.")
    tracked = [link_input(work, "field_matrix.toml", args.matrix),
               link_input(work, "previous_config.toml", past_config)]
    all_cores=[*directories,*args.comparison]
    if len(all_cores)!=len({d.name for d in all_cores}):
        parser.error("An observation core may only appear once in a cohort.")
    cache=args.work/"private_visits"
    cache.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.visit_workers) as workers:
        snapshots=list(workers.map(lambda d:visit_snapshot(d,cache),all_cores))
    visits={}
    for core,snapshot in zip(all_cores,snapshots):
        tracked.append(link_input(work,Path(core.name)/"private_visits.json",snapshot))
        visits[core.name]={k:set(v) for k,v in json.loads(snapshot.read_text()).items()}
        if core in directories:
            names=["regional_summary.json","private_qso_pairs.json","run_manifest.json"]
            if fields[core.name].get("series")=="v08":
                names.append("private_gaia_pairs.json")
            for name in names:
                tracked.append(link_input(work, Path(core.name) / name, core / name))
    out = "spatial_cohort_summary.json"
    with run_stage(work, args.config, args.cohort, "spatial-cohort-v08",
                   inputs=tracked, outputs=[out]) as go:
        if go:
            data = assess(directories, dict(settings, fields=fields, bootstrap_draws=1200),
                          args.comparison, overlap_from_visits(visits), matrix["fields"])
            save_json(work / out, data)


if __name__ == "__main__":
    main()
