"""Publication figures from the public NSC comparison summaries."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PALETTE = dict(published_nsc="#546472", M1="#b07342", M2="#8b5660", M3="#267570")
LABEL = dict(published_nsc="Published NSC", M1="M1", M2="M2", M3="M3")
FIELDS = dict(test_equator="Equator", test_north="North",
              test_crowding="Crowding", pal5="Pal 5")


def plot_bright(summaries, destination):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    for axis, component in zip(axes, ("East", "North", "Radial")):
        for i, method in enumerate(PALETTE):
            values = []
            for summary in summaries:
                strict = summary["strict"]
                selected = strict if "methods" in strict else summary["without_support_requirement"]
                row = selected["methods"].get(method)
                q = (row["radial_abs68"] if component == "Radial"
                     else row["abs68"][component == "North"]) if row is not None else np.nan
                values.append(q)
            axis.bar(np.arange(len(values))+(i-1.5)*.19,values,.19,
                     color=PALETTE[method],label=LABEL[method])
        axis.set_xticks(np.arange(len(summaries)), [
            FIELDS[d["region"]]+f"\n$n={d['strict']['n'] or d['without_support_requirement']['n']}$"
            for d in summaries])
        axis.set_title(component)
        axis.tick_params(direction="in")
    axes[0].set_ylabel("68th pct. absolute PM error (mas/yr)")
    axes[-1].legend(frameon=False)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def plot_qso(summary, destination):
    names = {"all": "All common QSO", "faint_21_to_23": "r = 21–23",
             "faint_21_to_22": "r = 21–22", "faint_22_to_23": "r = 22–23",
             "faint_short": "r = 21–23, NSC <2 yr", "faint_long": "r = 21–23, NSC ≥2 yr"}
    observed = []
    for key,label in names.items():
        row = summary["pooled"].get(key, {})
        stats = row.get("stats", {})
        pair = stats.get("paired", {}).get("M3_vs_published_nsc")
        if pair is None or pair["fraction95"][2] is None:
            continue
        field = (row.get("observation_block_bootstrap") or {}).get("fraction95", [None]*3)[2]
        observed.append((label, stats["n"], pair["fraction_improved"][2], pair["fraction95"][2], field))
    if not observed:
        return
    fig, axis = plt.subplots(figsize=(7.6, 3.8), layout="constrained")
    for i, (label,n,center,interval,group) in enumerate(observed):
        axis.hlines(i, 100*interval[0],100*interval[1],color="#267570",linewidth=2,
                    label="Paired objects" if i==0 else None)
        if group is not None:
            axis.hlines(i+.19, 100*group[0],100*group[1],color="#8b5660",linewidth=2,
                        label="Observation blocks and objects" if i==0 else None)
        axis.scatter(100*center,i,color="#267570",zorder=3,s=23)
    axis.set_yticks(np.arange(len(observed)), [f"{label}  ($n={n}$)" for label,n,*_ in observed])
    axis.invert_yaxis()
    axis.axvline(0,color=".45",linewidth=.8)
    axis.axvline(100*summary["practical_gain"],color=".55",linewidth=.8,linestyle="--")
    axis.set_xlabel("Gain in radial 68th pct. absolute PM error (%)")
    axis.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2)
    axis.tick_params(direction="in")
    fig.savefig(destination,dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("bright_summaries",nargs="+",type=Path)
    p.add_argument("--qso",type=Path)
    p.add_argument("--qso-output",type=Path)
    args=p.parse_args()
    if bool(args.qso)!=bool(args.qso_output):
        p.error("Give both QSO summary and output.")
    rows=[json.loads(path.read_text()) for path in args.bright_summaries]
    with plt.rc_context({"font.family":"serif","axes.spines.top":False,"axes.spines.right":False}):
        plot_bright(rows,args.output)
        if args.qso is not None:
            plot_qso(json.loads(args.qso.read_text()),args.qso_output)


if __name__=="__main__":
    main()
