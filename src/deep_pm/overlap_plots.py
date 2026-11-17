"""Plot held positions and strictly common proper motions from v1.1 public aggregates."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .overlap_blocks import CATALOGUES, MODELS
from .validation_plots import NAMES

COLORS = dict(zip(CATALOGUES, ("#6f6a6b", "#aa7756", "#4479a4", "#008b70", "#91485f")))
FIELDS = {**NAMES, "south_17_overlap": "Independent south"}


def boundary_plot(pilots, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), layout="constrained")
    x = np.arange(len(pilots))
    for ax, title in zip(axes, ("Same SDSS scan, adjacent fields", "SDSS–NSC offsets")):
        counts = []
        for i, p in enumerate(pilots.values()):
            if ax is axes[0]:
                boundary = p["overlap_boundary"]
                counts.append(boundary["common_pairs"])
                values = boundary.get("radial_p68_mas")
                if values is None:
                    continue
            else:
                stats = p["galaxies"]["comparison"]
                counts.append(stats["n"])
                if "methods" not in stats:
                    continue
                values = [stats["methods"][model]["radial_abs68"] for model in MODELS]
            for j, (model, value) in enumerate(zip(MODELS, values)):
                ax.bar(i + .25*(j-1), value, .24, color=COLORS[model],
                       label=model if i == 0 else None)
        ax.set_xticks(x, [f"{FIELDS[k]}\nn={n:,}" for k,n in zip(pilots,counts)],
                      rotation=25, ha="right", fontsize=8)
        ax.set(title=title, ylabel="Held-galaxy radial difference P68 [mas]")
    axes[0].legend(frameon=False)
    fig.savefig(output/"v11_overlap_boundary_summary.png",dpi=170)
    plt.close(fig)


def motion_plot(blocks, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), layout="constrained")
    for ax, family, label in zip(axes, ("three","five"), ("Fixed detections", "All five catalogues")):
        for i, tag in enumerate(("21_to_22", "22_to_23")):
            stats = blocks[tag]["ordinary"][family]["night"]
            for j, (method, value) in enumerate(zip(stats["methods"], stats.get("radial_p68_masyr", []))):
                width = .78/len(stats["methods"])
                ax.bar(i+(j-(len(stats["methods"])-1)/2)*width,value,width*.94,
                       color=COLORS[method],label=method if i==0 else None)
            n, b = stats["n"], stats["independent_blocks"]
            ax.text(i,.02,f"n={n}, blocks={b}",transform=ax.get_xaxis_transform(),
                    ha="center",va="bottom",fontsize=8,
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.90,pad=2))
        ax.set_xticks([0,1],["21 ≤ r < 22","22 ≤ r < 23"])
        ax.set(title=label,ylabel="Spectroscopic QSO radial P68 [mas/yr]")
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output/"v11_overlap_motion_summary.png",dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path)
    p.add_argument("figures",type=Path)
    a=p.parse_args()
    order = ("repeat_rich", "repeat_sparse", "sparse_north_outer", "sparse_south_east_ready", "south_17_overlap")
    pilots = {}
    for path in sorted(a.results.glob("*overlap_pilot_v11_*_summary.json")):
        data = json.loads(path.read_text())
        pilots[data["region"]] = data
    if not pilots:
        p.error("No validated public v1.1 pilot summaries")
    a.figures.mkdir(parents=True,exist_ok=True)
    boundary_plot({k:pilots[k] for k in order if k in pilots},a.figures)
    blocks = json.loads((a.results/"v11_overlap_blocks_summary.json").read_text())
    motion_plot(blocks,a.figures)


if __name__ == "__main__":
    main()
