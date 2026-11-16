"""Plot aggregate paired controls without loading individual source results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def report(summary, output, region):
    common = summary["fixed_truth_common_four_way"]
    if not common:
        return
    plt.rcParams.update({"font.family": "serif", "xtick.direction": "in", "ytick.direction": "in"})
    tags = ("nsc_published", "nsc_refit", "one_sdss", "all_sdss")
    x = np.arange(len(tags))
    values = np.array([common[k]["nmad"] for k in tags])
    n = common[tags[0]]["n"]
    fig, ax = plt.subplots(figsize=(7.2, 3.7), layout="constrained")
    for axis, color, text in ((0, "#356a9c", "East"), (1, "#aa5736", "North")):
        ax.bar(x + (axis - .5) * .3, values[:, axis], .28, color=color, label=text)
    ax.set(xticks=x, xticklabels=["NSC published", "NSC refit", "+ first SDSS", "+ all SDSS"],
           ylabel="PM residual NMAD (mas/yr)", title=f"{region} | same confirmed {n} sources")
    ax.legend(frameon=False)
    fig.savefig(output / f"{region}_v02_paired_summary.png", dpi=170)
    plt.close(fig)

    runs = summary["sdss_run_heldout"]
    if not runs:
        return
    items = [(name, values["all_minus_drop_run"]) for name, values in runs.items()]
    score = np.array([entry["median"] for _, entry in items])
    fig, ax = plt.subplots(figsize=(7.2, max(3.6, len(items) * .34)), layout="constrained")
    order = np.arange(len(items))
    ax.scatter(score[:, 0], order - .12, label="East", color="#356a9c")
    ax.scatter(score[:, 1], order + .12, label="North", color="#aa5736")
    ax.axvline(0, color="#777777", ls=":")
    ax.set(yticks=order, yticklabels=[f"{run} (n={row['n']})" for run, row in items],
           xlabel="Median all-run minus drop-run motion (mas/yr)",
           title=f"{region} | one SDSS run removed at a time")
    ax.legend(frameon=False)
    fig.savefig(output / f"{region}_v02_run_influence_summary.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--region", required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    report(json.loads(args.summary.read_text()), args.output, args.region)
