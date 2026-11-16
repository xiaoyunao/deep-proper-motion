"""Aggregate paired QSO experiments over independently selected fields."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .io import digest, save_json
from .nsc_comparison import absolute68, comparison, configuration
from .provenance import run_stage


def ranking(coverage, settings):
    eligible = [(field, row) for field, row in coverage.items()
                if row["candidates_with_both"] >= settings["min_candidates_with_sdss"]]
    return sorted(eligible, key=lambda item: (-item[1]["faint_candidates_with_both"],
                                              -item[1]["candidates_with_both"], item[0]))


def field_resample(fields, settings):
    """Resample fields then sources, preserving matched model estimates."""
    if len(fields) < 2:
        return None
    random = np.random.default_rng(settings["seed"])
    data = [(np.array([obs["M3"][0] for obs in rows]),
             np.array([obs["published_nsc"][0] for obs in rows])) for rows in fields]
    samples = []
    for _ in range(settings["bootstrap_draws"]):
        observed = [[], []]
        for i in random.integers(len(data), size=len(data)):
            m3, nsc = data[i]
            indices = random.integers(len(m3), size=len(m3))
            observed[0].append(m3[indices])
            observed[1].append(nsc[indices])
        new, ref = (absolute68(np.concatenate(values)) for values in observed)
        samples.append(np.divide(ref-new, ref, where=ref>0, out=np.full(3, np.nan)))
    draws = np.asarray(samples)
    intervals = [np.quantile(col,[.025,.975]).tolist() if np.isfinite(col).all() else None
                 for col in draws.T]
    return dict(fields=len(data), fraction95=intervals)


def leave_field_out(fields, settings):
    if len(fields) < 2:
        return None
    gains = []
    for excluded in fields:
        subset = [item for name, rows in fields.items() if name != excluded for item in rows]
        if len(subset) < settings["min_publish_count"]:
            continue
        ref, new = (absolute68(np.array([r[m][0] for r in subset])) for m in ("published_nsc","M3"))
        if (ref > 0).all():
            gains.append((ref-new)/ref)
    return dict(n_cases=len(gains), min=np.min(gains,axis=0), max=np.max(gains,axis=0)) if gains else None


def checked_rows(directory):
    manifest = json.loads((directory/"run_manifest.json").read_text())
    attempt = manifest["stages"]["nsc-compare-v04"][-1]
    relative = "private_paired_qso.json"
    if attempt["state"] != "completed" or attempt["outputs"].get(relative) != digest(directory/relative):
        raise ValueError("Invalid private paired-control receipt")
    return json.loads((directory/relative).read_text())


def survey_overlap(directories):
    visits = {}
    for directory in directories:
        table = pd.read_csv(directory/"detections.csv", usecols=["survey", "exposure_id"])
        visits[directory.name] = {k: set(group.exposure_id.dropna()) for k, group in table.groupby("survey")}
    blocks = {k:{k} for k in visits}
    counts = {}
    for a,b in combinations(sorted(visits),2):
        shared = {survey.lower():len(visits[a].get(survey,set()) & visits[b].get(survey,set()))
                  for survey in ("SDSS","NSC")}
        counts[a+"__"+b] = shared
        if any(shared.values()):
            merged = blocks[a] | blocks[b]
            for name in merged:
                blocks[name] = merged
    return sorted({tuple(sorted(names)) for names in blocks.values()}), counts


def run_pool(work, directories, settings):
    groups = defaultdict(dict)
    for directory in directories:
        name = directory.name
        rows = checked_rows(directory)
        for r, baseline, result in rows:
            groups["all"].setdefault(name, []).append(result)
            if 21 <= r < 23:
                groups["faint_21_to_23"].setdefault(name, []).append(result)
                interval = "faint_21_to_22" if r < 22 else "faint_22_to_23"
                groups[interval].setdefault(name, []).append(result)
                stratum = "faint_short" if baseline < settings["short_nsc_baseline_years"] else "faint_long"
                groups[stratum].setdefault(name, []).append(result)
    blocks, overlap = survey_overlap(directories)
    pooled = {}
    for key, fields in groups.items():
        by_field = list(fields.values())
        data = [row for part in by_field for row in part]
        block_rows = [[row for name in block for row in fields.get(name,[])] for block in blocks]
        block_rows = [rows for rows in block_rows if rows]
        pooled[key] = dict(field_count=len(fields), observation_block_count=len(block_rows),
                           stats=comparison(data, settings),
                           field_bootstrap=field_resample(by_field, settings)
                           if len(data)>=settings["min_publish_count"] else None,
                           observation_block_bootstrap=field_resample(block_rows,settings)
                           if len(data)>=settings["min_publish_count"] else None,
                           leave_field_out=leave_field_out(fields, settings))
    save_json(work/"nsc_cohort_summary.json", dict(fields=sorted(d.name for d in directories),
                                                    pooled=pooled, practical_gain=settings["practical_gain"],
                                                    observation_blocks=blocks, archive_visit_overlap=overlap))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--config",type=Path,default=Path("configs/precision-v04.toml"))
    p.add_argument("--pool", nargs="+", type=Path)
    args = p.parse_args()
    _, settings = configuration(args.config)
    if not args.pool:
        candidates = {}
        for folder in settings["fields"]:
            f = args.root/folder/"qso_coverage_summary.json"
            if f.exists():
                candidates[folder] = json.loads(f.read_text())
        selected = ranking(candidates, settings)[:settings["max_fields"]]
        print(json.dumps({"inventoried":len(candidates),"ranked":selected}, indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Run on a Slurm compute node.")
    work = args.root/"qso_pool"
    work.mkdir(parents=True,exist_ok=True)
    if not len(args.pool)==len({d.name for d in args.pool}):
        p.error("Cannot pool a field twice.")
    artifacts = ("private_paired_qso.json","run_manifest.json","detections.csv")
    for directory in args.pool:
        for name in artifacts:
            link=work/f"{directory.name}_{name}"
            target=directory.resolve()/name
            if not link.exists():
                link.symlink_to(target)
            if target != link.resolve():
                p.error("Use a new pool directory for different field versions.")
    inputs = [f"{directory.name}_{name}" for directory in args.pool
              for name in artifacts]
    with run_stage(work,args.config,"qso_pool","nsc-pool-v04",inputs=inputs,
                   outputs=["nsc_cohort_summary.json"]) as go:
        if go:
            run_pool(work,args.pool,settings)


if __name__=="__main__":
    main()
