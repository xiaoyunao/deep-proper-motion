"""Render only aggregate statistics from completed runs."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import save_json

LABELS = {"fixed": "Fixed 1 arcsec", "nsc_like": "NSC-like", "motion": "Motion-aware"}
COLORS = {"fixed": "#444444", "nsc_like": "#336aa8", "motion": "#bc4b31"}


def read(path):
    return json.loads(path.read_text())


def report(work, region):
    plt.rcParams.update({"font.family": "serif", "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True, "axes.labelsize": 11})
    out = work / "summary"
    out.mkdir(exist_ok=True)
    inventory = read(work / "inventory.json")
    validation = read(work / "validation.json") if (work / "validation.json").exists() else {}
    resources = read(work / "association_resources.json") if (work / "association_resources.json").exists() else {}
    summary = dict(region=region, inventory=inventory, validation=validation, resources=resources,
                   manifest=read(work / "run_manifest.json"))
    save_json(out / f"{region}_summary.json", summary)

    fig, ax = plt.subplots(figsize=(8.4, 3.4), layout="constrained")
    for survey, values in sorted(inventory["counts"].items()):
        epochs = values["years"]
        ax.plot([int(k) for k in epochs], list(epochs.values()), "o-",
                label=f"{survey}: {values['detections']:,} detections")
    ax.set(xlabel="Observation year (TT)", ylabel="Catalogue detections", yscale="log",
           title=f"{region} | core and halo after within-run deduplication")
    ax.legend(frameon=False)
    fig.savefig(out / f"{region}_epochs_summary.png", dpi=180)
    plt.close(fig)
    if not validation or not validation["truth_cross_survey"]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for name, results in validation["methods"].items():
        bins = [item for item in results["pm_bins"] if item["denominator"]]
        for row in bins:
            x = (row["low"] + row["high"]) / 2
            y = row["value"]
            lo, hi = row["wilson95"]
            axes[0].errorbar(x, y, yerr=[[max(0, y - lo)], [max(0, hi - y)]],
                             color=COLORS[name], fmt="o", capsize=3)
        for row in validation["curves"][name]:
            x = row["wrong_known_detections"]["value"]
            y = row["recovery"]["value"]
            if x is not None and y is not None:
                axes[1].scatter(x, y, marker={"fixed": "o", "nsc_like": "s", "motion": "^"}[name],
                                color=COLORS[name], facecolors="none", s=48)
        axes[0].plot([], [], "o", color=COLORS[name], label=LABELS[name])
    axes[0].set(xlabel="Gaia proper motion bin midpoint (mas/yr)", ylabel="Recovery fraction",
                ylim=(-0.08, 1.10))
    axes[1].set(xlabel="Discordant fraction among labelled detections", ylabel="Recovery fraction",
                ylim=(-0.08, 1.10))
    axes[0].legend(frameon=False)
    fig.suptitle(f"{region} | {validation['truth_cross_survey']} Gaia-labelled cross-survey sources")
    fig.savefig(out / f"{region}_recovery_summary.png", dpi=180)
    plt.close(fig)

    if not validation["paired"]:
        return
    frame = pd.read_csv(work / "validation" / "paired.csv")
    tags = ["nsc_only", "one_sdss", "all_sdss"]
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.8), layout="constrained", sharey=True)
    rng = np.random.default_rng(1)
    for ax, column in zip(axes, ["east", "north"]):
        table = frame.pivot(index="source", columns="fit", values=column)[tags]
        for row in table.to_numpy():
            shift = rng.uniform(-0.07, 0.07)
            ax.plot(np.arange(3) + shift, row, "o-", color="#345871", alpha=0.4, ms=3)
        ax.axhline(0, linestyle=":", color="#424242")
        ax.set(xticks=np.arange(3), xticklabels=["NSC", "+ one SDSS", "+ all SDSS"],
               title=f"{column.title()} component")
    axes[0].set_ylabel("Fit minus Gaia (mas/yr)")
    fig.suptitle(f"{region} | fixed Gaia-labelled associations, {len(table)} sources")
    fig.savefig(out / f"{region}_paired_summary.png", dpi=180)
    plt.close(fig)
