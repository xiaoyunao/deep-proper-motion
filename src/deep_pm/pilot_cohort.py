"""Audit overlap, independent dark-QSO results and measured pilot throughput."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np

from .io import digest, save_json
from .nsc_cohort import field_resample, leave_field_out, survey_overlap
from .nsc_comparison import comparison, configuration
from .provenance import run_stage


def checked_pilot(directory):
    record = json.loads((directory/"run_manifest.json").read_text())
    stage = record["stages"]["production-v05"][-1]
    artifacts = ("production_summary.json","private_qso_pairs.json")
    if stage["state"] != "completed" or any(
            stage["outputs"].get(name) != digest(directory/name) for name in artifacts):
        raise ValueError("Incomplete pilot product or invalid computation receipt.")
    return tuple(json.loads((directory/name).read_text()) for name in artifacts)


def aggregate(work, directories, stripe, settings):
    new_blocks, overlap = survey_overlap(directories)
    _, prior_links = survey_overlap([*directories,*stripe]) if stripe else ([],{})
    names = {d.name for d in directories}
    old = {d.name for d in stripe}
    aliases = {}
    for pair, shared in prior_links.items():
        parts = set(pair.split("__"))
        newer, older = parts&names, parts&old
        if newer and older:
            aliases[pair] = shared
    independent = names - {n for pair, count in aliases.items() if any(count.values())
                           for n in pair.split("__") if n in names}
    totals = defaultdict(Counter)
    population = defaultdict(dict)
    resources = {}
    area = 0.
    for directory in directories:
        aggregates, pairs = checked_pilot(directory)
        name = directory.name
        area += aggregates["core_area_sqdeg"]
        for label, counts in aggregates["funnel"].items():
            totals[label].update(counts)
        stats = aggregates["resources"]
        acquisition = directory/"detections.csv"
        input_dir = acquisition.resolve().parent
        counts_path = input_dir/"pilot_input_counts.json"
        counts = json.loads(counts_path.read_text()) if counts_path.is_file() else {}
        acquisition_receipt = json.loads((input_dir/"run_manifest.json").read_text())["stages"]["pilot-inputs-v05"][-1]
        if acquisition_receipt["state"] != "completed":
            raise ValueError("Input acquisition is not complete.")
        acquisition_seconds = (datetime.fromisoformat(acquisition_receipt["finished_utc"]) -
                               datetime.fromisoformat(acquisition_receipt["started_utc"])).total_seconds()
        resources[name] = dict(stats,acquisition_wall_seconds=acquisition_seconds,area_sqdeg=aggregates["core_area_sqdeg"],
                               raw_bytes=sum((input_dir/n).stat().st_size for n in ("nsc_object.csv","sdss.csv")),
                               **counts)
        if name not in independent:
            continue
        for r, base, result in pairs:
            groups = ["all"]
            if 21<=r<23:
                groups.extend(("faint_21_to_23","faint_22_to_23" if r>=22 else "faint_21_to_22"))
            for tag in groups:
                population[tag].setdefault(name,[]).append(result)
    pooled = {}
    for tag, by_field in population.items():
        entries = [r for one in by_field.values() for r in one]
        grouped = [[r for name in block for r in by_field.get(name,[])] for block in new_blocks]
        groups = [g for g in grouped if g]
        pooled[tag] = dict(fields=len(by_field), observation_blocks=len(groups),
                           stats=comparison(entries,settings), block_ci=field_resample(groups,settings),
                           leave_field_out=leave_field_out(by_field,settings))
    final = sum(record["final"] for record in totals.values())
    rates = [part["cpu_seconds"]/part["area_sqdeg"] for part in resources.values()]
    download = [part["acquisition_wall_seconds"]/part["area_sqdeg"] for part in resources.values()]
    raw = [part["raw_bytes"]/part["area_sqdeg"] for part in resources.values()]
    forecast = {str(scale):dict(fit_validation_cpu_hours_range=[scale*min(rates)/3600,scale*max(rates)/3600],
                                raw_gb_range=[scale*min(raw)/1e9,scale*max(raw)/1e9],
                                acquisition_job_hours_range=[scale*min(download)/3600,
                                                             scale*max(download)/3600])
                for scale in (100,300)}
    save_json(work/"pilot_cohort_summary.json",dict(core_area_sqdeg=area,fields=sorted(names),funnel=totals,
              final_sources=final, final_per_sqdeg=final/area, independent_qso=pooled,
              v04_stripe_overlap=aliases, pilot_overlap=overlap, resources_by_field=resources,
              planning_range=forecast))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("--pilot",type=Path,nargs="+",required=True)
    p.add_argument("--stripe",type=Path,nargs="+",required=True)
    p.add_argument("--config",type=Path,default=Path("configs/production-v05.toml"))
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Use a Slurm compute node.")
    if len({d.name for d in args.pilot+args.stripe}) != len(args.pilot+args.stripe):
        p.error("Field names must be unique.")
    _, settings = configuration(args.config)
    args.work.mkdir(parents=True,exist_ok=True)
    inputs = []
    for group in args.pilot+args.stripe:
        required = ["detections.csv"]
        if group in args.pilot:
            required += ["production_summary.json","private_qso_pairs.json","run_manifest.json"]
        for name in required:
            local = args.work/(group.name+"_"+name)
            target = (group/name).resolve(strict=True)
            if not local.exists():
                local.symlink_to(target)
            if local.resolve()!=target:
                p.error("A different run directory is required for other pilot inputs.")
            inputs.append(local.name)
    with run_stage(args.work,args.config,"pilot_pool","pilot-pool-v05",inputs=inputs,
                   outputs=["pilot_cohort_summary.json"]) as go:
        if go:
            aggregate(args.work,args.pilot,args.stripe,settings)


if __name__=="__main__":
    main()
