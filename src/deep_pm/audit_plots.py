"""Plot aggregate sky coverage and anonymized annual trajectory residuals."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .__main__ import detection_arrays, tomllib
from .astrometry import years


def run(work, region, config):
    spec = config["regions"][region]
    data, xy, _, _ = detection_arrays(work, (spec["ra"], spec["dec"]), config)
    plt.rcParams.update({"font.family": "serif", "xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.9), sharey=True, layout="constrained")
    extent = (spec["radius"] + config["access"]["halo_deg"]) * 60
    bins = np.linspace(-extent, extent, 19)
    for ax, survey in zip(axes, ["SDSS", "NSC"]):
        i = data.survey.eq(survey)
        east, north = xy[i].T / 60000
        grid, _, _ = np.histogram2d(east, north, bins=bins)
        masked = np.ma.masked_where(grid < 3, np.log10(np.maximum(grid, 1)))
        mesh = ax.pcolormesh(bins, bins, masked.T, cmap="viridis")
        ax.add_patch(plt.Circle((0, 0), spec["radius"] * 60, fill=False, linestyle=":",
                                linewidth=1.4, color="#b43e29"))
        ax.set(title=f"{survey} | {int(i.sum()):,} detections", xlabel="East (arcmin)", aspect="equal")
        fig.colorbar(mesh, ax=ax, label="log₁₀ detections per bin")
    axes[0].set_ylabel("North (arcmin)")
    fig.suptitle(f"{region} | association sample, bins with ≥3 detections")
    fig.savefig(work / "summary" / f"{region}_coverage_summary.png", dpi=180)
    plt.close(fig)

    path = work / "validation" / "detection_residuals.csv"
    if not path.exists():
        return
    rows = pd.read_csv(path)
    rows["year"] = np.floor(years(rows.observation_time))
    cases = list(rows.case.unique())
    fig, axes = plt.subplots(len(cases), 2, figsize=(8.5, 2.2 * len(cases) + 1), squeeze=False,
                            layout="constrained")
    for i, case in enumerate(cases):
        part = rows[rows.case.eq(case)]
        for j, column in enumerate(("east", "north")):
            ax = axes[i, j]
            for survey, group in part.groupby("survey"):
                binned = group.groupby("year")[column].agg(["median", "count"])
                visible = binned[binned["count"] >= 2]
                ax.plot(visible.index, visible["median"], "o-", label=survey, ms=4)
            ax.axhline(0, linestyle=":", color="#555555")
            ax.set_ylabel(f"Case {case} (mas)")
            if i == 0:
                ax.set_title(f"{column.title()} residual")
            if i == len(cases) - 1:
                ax.set_xlabel("Observation year")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"{region} | annual medians, ≥2 detections per survey and year")
    fig.savefig(work / "summary" / f"{region}_cases_summary.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--region", default="smoke")
    parser.add_argument("--config", default="configs/experiment-a.toml")
    args = parser.parse_args()
    with open(args.config, "rb") as file:
        config = tomllib.load(file)
    run(args.work, args.region, config)
