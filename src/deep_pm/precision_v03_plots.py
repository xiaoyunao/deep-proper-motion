"""Plot verified v0.3 aggregate outputs without reading catalogue rows."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(zero, chromatic, output):
    colors = ["#657082", "#257572", "#ad6545", "#65488d"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    axis = axes[0]
    for i, method in enumerate(("M2", "M3")):
        values = zero["calibration"]["heldout_galaxy_position_difference_mas"][method]["median"]
        axis.bar(np.arange(2)+(i-.5)*.30, values, .30, label=method, color=colors[i])
    axis.axhline(0, color=".5", linewidth=.75)
    axis.set_xticks([0, 1], ["East", "North"])
    axis.set_ylabel("Median SDSS − NSC position (mas)")
    axis.set_title("Held-out galaxies (57)")
    axis.legend(frameon=False)

    axis = axes[1]
    axis.axhspan(-.5, .5, color=".92")
    for i, method in enumerate(("M2", "M3")):
        row = zero["common_support"][method]
        axis.scatter(np.arange(2)+(i-.5)*.16, row["median"], label=method, color=colors[i], zorder=3)
    axis.set_xticks([0, 1], ["East", "North"])
    axis.set_ylabel("Median motion residual (mas/yr)")
    axis.set_title("Same 47 Gaia stars")
    axis.legend(frameon=False)

    axis = axes[2]
    for i, method in enumerate(("M3", "M4", "M5")):
        row = chromatic["models"][method]
        axis.bar(np.arange(2)+(i-1)*.23, row["nmad"], .23,
                 label=method, color=colors[i+1])
    axis.set_xticks([0, 1], ["East", "North"])
    axis.set_ylabel("Motion residual NMAD (mas/yr)")
    axis.set_title("Common chromatic support (47)")
    axis.legend(frameon=False)
    for axis in axes:
        axis.tick_params(direction="in")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_dark(summary, output):
    bins = [tag for tag in ("20_to_21", "21_to_22", "22_to_23")
            if tag in summary["coverage_by_r"]]
    labels = []
    for tag in bins:
        row = summary["coverage_by_r"][tag]
        labels.append(tag.replace("_to_", "–") + f"\n{row['matched_tracks']} stars")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), layout="constrained")
    for axis, component, label in zip(axes, (0, 1), ("East", "North")):
        for i, method in enumerate(("M2", "M3")):
            rows = summary["residual_by_r_mas"]
            values = [rows[tag + "/" + method]["abs68"][component] for tag in bins]
            axis.bar(np.arange(len(bins))+(i-.5)*.30, values, .3,
                     label=method, color=["#657082", "#257572"][i])
        axis.set_xticks(np.arange(len(bins)), labels)
        axis.set_title(label)
        axis.set_xlabel("NSC reference r (mag)")
        axis.tick_params(direction="in")
    axes[0].set_ylabel("Held-out position |68th pct.| (mas)")
    axes[1].legend(frameon=False)
    fig.suptitle("Predictions for NSC r exposures held out in 2016")
    fig.savefig(output, dpi=160)
    plt.close(fig)



def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("zero", "chromatic", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--dark-summary", type=Path)
    p.add_argument("--dark-output", type=Path)
    args = p.parse_args()
    if bool(args.dark_summary) != bool(args.dark_output):
        p.error("Supply both dark plot arguments together.")
    with plt.rc_context({"font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlepad": 9}):
        plot(json.loads(args.zero.read_text()), json.loads(args.chromatic.read_text()), args.output)
        if args.dark_summary:
            plot_dark(json.loads(args.dark_summary.read_text()), args.dark_output)


if __name__ == "__main__":
    main()
