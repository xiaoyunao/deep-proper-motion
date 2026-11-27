"""Draw F3 and F4 once from the aggregate certified against the blind source closure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from .full_coordinate_figures import COLD, panel, save

LABEL = {"ours": "SDSS + NSC", "xiao": "Xiao et al. (2025)", "nsc": "NSC DR2", "r": "Physical r only"}
COLOR = {"ours": "#173d66", "xiao": "#ae511e", "nsc": "#43836e", "r": "#97649e"}


def precision(summary, directory):
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.2), layout="constrained", sharex=True)
    for table, methods, col in (
        ("fine_magnitude", ("ours", "xiao", "nsc"), 0),
        ("physical_r", ("ours", "r"), 1),
    ):
        rows = summary[table]
        edges = [tuple(map(float, v.split("-"))) for v in rows]
        order = np.argsort([v[0] for v in edges])
        names = [list(rows)[i] for i in order]
        x = np.array([sum(edges[i]) / 2 for i in order])
        for method in methods:
            data = [rows[key].get("catalogues", {}).get(method, {}) for key in names]
            axes[0, col].plot(
                x,
                [d.get("p68", np.nan) for d in data],
                "o-",
                ms=3.5,
                color=COLOR[method],
                label=LABEL[method],
            )
            tails = [d.get("tail_gt20", {}) for d in data]
            axes[1, col].plot(
                x, [100 * d.get("rate", np.nan) for d in tails], "o-", ms=3.5, color=COLOR[method]
            )
            limits = np.array([d.get("wilson95", [np.nan, np.nan]) for d in tails])
            axes[1, col].fill_between(
                x, limits[:, 0] * 100, limits[:, 1] * 100, color=COLOR[method], alpha=0.11
            )
        counts = [rows[k]["common"] for k in names]
        axes[2, col].stairs(
            counts,
            [edges[i][0] for i in order] + [edges[order[-1]][1]],
            color="#284963",
            fill=True,
            alpha=0.7,
        )
        axes[2, col].set_yscale("symlog", linthresh=1)
        axes[2, col].set_xlabel("Spectroscopic r magnitude")
        axes[0, col].legend(loc="best", frameon=False, fontsize=9)
        for ax in axes[:, col]:
            ax.grid(axis="y", alpha=0.22)
    axes[0, 0].set_title("Strict common spectroscopic QSOs")
    axes[0, 1].set_title("Fixed all-band and physical-r pairs")
    for ax, text in zip(
        axes[:, 0], ["Radial P68 (mas yr⁻¹)", "Fraction above 20 mas yr⁻¹ (%)", "Common QSOs per bin"]
    ):
        ax.set_ylabel(text)
    save(fig, directory, "F3_qso_precision")


def sky(summary, directory):
    fig, axes = plt.subplots(
        4, 2, figsize=(13.8, 13.4), subplot_kw=dict(projection="aitoff"), layout="constrained"
    )
    for line, (kind, limit, cmap, label) in enumerate(
        (
            ("east", 4.0, plt.get_cmap("RdBu_r"), "Median east PM (mas yr⁻¹)"),
            ("north", 4.0, plt.get_cmap("RdBu_r"), "Median north PM (mas yr⁻¹)"),
            ("xiao", 50.0, COLD, "P68 improvement relative to Xiao (%)"),
            ("nsc", 90.0, COLD, "P68 improvement relative to NSC (%)"),
        )
    ):
        norm = TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
        for col, lo in enumerate((21, 22)):
            cells = summary["faint"][str(lo)]["map"]
            covered = {c["pixel"]: 1 for c in cells if c["parent"]}
            good = [c for c in cells if c["common"] >= summary["map_definition"]["display_min"]]
            if kind in ("east", "north"):
                values = {
                    c["pixel"]: c["catalogues"]["ours"]["median"][int(kind == "north")] for c in good
                }
            else:
                values = {c["pixel"]: c["improvement_percent"][kind] for c in good}
            n = sum(c["common"] for c in good)
            artist = panel(
                axes[line, col],
                values,
                summary["map_definition"]["healpix_nside"],
                f"{lo} ≤ r < {lo+1}    ·    {n:,} QSOs",
                norm,
                cmap,
                coverage=covered,
            )
        bar = fig.colorbar(
            artist,
            ax=axes[line].tolist(),
            orientation="horizontal",
            shrink=0.55,
            pad=0.015,
            aspect=55,
            extend="both",
        )
        bar.set_label(label)
    save(fig, directory, "F4_qso_spatial_quality")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    data = json.loads(a.summary.read_text())
    precision(data, a.directory)
    sky(data, a.directory)
