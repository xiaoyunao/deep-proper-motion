"""Summarize the unchanged v0.7 control on v0.8 diagnostics."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

import numpy as np

from .dark_precision import bin_name
from .io import save_json
from .nsc_comparison import configuration
from .provenance import run_stage
from .regional_cohort import load_output
from .spatial_cohort import link_input
from .validation import ratio


def matched_conditions(private, blocks, settings):
    pairs = defaultdict(lambda: defaultdict(list))
    edges = settings["reference_r_edges"]
    categories = (("21_to_22", "20_to_21"), ("22_to_23", "21_to_22"))
    for block in blocks:
        estimates = defaultdict(list)
        for field in block:
            grouped = defaultdict(lambda: defaultdict(list))
            for r, camera, instrument, duration, era, color, motion in private.get(field, []):
                tag = bin_name(r, edges)
                grouped["observing_conditions", camera, instrument, duration, era][tag].append(motion)
                if color != "missing":
                    grouped["additionally_color_matched", camera, instrument, duration, era, color][tag].append(motion)
            for (mode, *_), values in grouped.items():
                for faint, bright in categories:
                    if min(len(values[faint]), len(values[bright])) < 3:
                        continue
                    delta = np.median(values[faint], axis=0) - np.median(values[bright], axis=0)
                    estimates[mode, faint, bright].append((len(values[faint]), len(values[bright]), delta))
        for label, rows in estimates.items():
            count = np.sum([row[:2] for row in rows], axis=0)
            pairs[label[0]][label[1]+"_minus_"+label[2]].append((len(rows), count, np.median([r[2] for r in rows],axis=0)))
    result = {}
    for mode, values in pairs.items():
        result[mode] = {}
        for contrast, rows in values.items():
            sample = np.array([row[2] for row in rows])
            seeds = np.random.default_rng(settings["seed"])
            ci = None
            if len(sample) >= settings["spatial"]["min_independent_blocks_for_high_precision"]:
                drawn = sample[seeds.integers(0,len(sample),(1200,len(sample)))]
                ci = np.quantile(np.median(drawn,axis=1),[.025,.975],axis=0).T
            result[mode][contrast] = dict(observation_blocks=len(rows), strata=sum(r[0] for r in rows),
                faint_and_bright_counts=np.sum([r[1] for r in rows],axis=0),
                median_delta_masyr=np.median(sample,axis=0), block95=ci)
    return result


def summarize_audits(summaries, private, blocks, settings):
    exits = Counter()
    trials = defaultdict(lambda: defaultdict(Counter))
    for row in summaries:
        exits.update({k: v["n"] for k,v in row["qso"]["strict_exits"].get("W0_22_to_23", {}).items()})
        for branch in ("qso", "gaia"):
            for cut, result in row["morphology"][branch].items():
                for key, item in result["by_condition"].items():
                    fraction = item["retained"]
                    trials[branch][cut][key+"__retained"] += fraction["n"]
                    trials[branch][cut][key+"__denominator"] += fraction["denominator"]
        trials["catalogue_sizes"]["global"].update(row["morphology"]["catalogue_sizes"])
    results = {}
    for branch in ("qso", "gaia"):
        results[branch] = {}
        for cut, item in trials[branch].items():
            fractions = {}
            for key, denominator in item.items():
                tag, meaning = key.rsplit("__",1)
                if meaning == "denominator":
                    fractions[tag] = ratio(item[tag+"__retained"],denominator)
            results[branch][cut] = fractions
    results["catalogue_sizes"] = trials["catalogue_sizes"]["global"]
    return dict(strict_faint_exits=exits,morphology=results,same_conditions=matched_conditions(private,blocks,settings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work",type=Path)
    parser.add_argument("audit",type=Path)
    parser.add_argument("v07_cohort",type=Path)
    parser.add_argument("--overrides",nargs="*",type=Path,default=[])
    parser.add_argument("--config",type=Path,default=Path("configs/spatial-v08.toml"))
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Use a Slurm compute node.")
    work=args.work/"history"
    work.mkdir(parents=True,exist_ok=True)
    tracked=[link_input(work,"v07_cohort.json",args.v07_cohort)]
    pool=json.loads(args.v07_cohort.read_text())
    expected=pool["fields"]
    inputs={name:args.audit/name for name in expected}
    for path in args.overrides:
        if path.name not in expected:
            parser.error("Override must be a v0.7 field.")
        inputs[path.name]=path
    stages={}
    for name,path in inputs.items():
        tracked.extend(link_input(work,Path(name)/file,path/file) for file in
                       ("spatial_audit_summary.json","private_diagnostics.json","run_manifest.json"))
        stages[name]=path
    out="spatial_history_summary.json"
    with run_stage(work,args.config,"history","spatial-history-v08",inputs=tracked,outputs=[out]) as go:
        if go:
            _,settings=configuration(args.config)
            summaries=[load_output(path,"spatial-audit-v08","spatial_audit_summary.json") for path in stages.values()]
            private={name:load_output(path,"spatial-audit-v08","private_diagnostics.json") for name,path in stages.items()}
            save_json(work/out,summarize_audits(summaries,private,pool["observation_blocks"],settings))


if __name__=="__main__":
    main()
