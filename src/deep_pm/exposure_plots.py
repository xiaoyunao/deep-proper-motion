"""Render v0.9 figures from the verified public aggregate receipts."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def read(pattern, directory):
    files = sorted(directory.glob(pattern))
    if len(files) != 1:
        raise ValueError(f"Expected one receipt for {pattern}, found {len(files)}")
    return json.loads(files[0].read_text())


def geometry(summary, figures):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True, sharey=True)
    for name, ax in zip("abc", axes):
        receipt = read(f"v09_{name}_exposure_preflight_*_summary.json", summary)
        for tile in receipt["tiles"]:
            west, east = tile["region"]["ra_bounds_deg"]
            south, north = tile["region"]["dec_bounds_deg"]
            fraction = tile["joint_time"]
            ax.add_patch(Rectangle((west, south), east-west, north-south,
                                   facecolor=plt.cm.viridis(fraction), edgecolor="white"))
            ax.text((east+west)/2, (south+north)/2, f"{fraction:.0%}",
                    ha="center", va="center", color="black" if fraction > .58 else "white", fontsize=8)
        if name == "c":
            for west, south in ((345, -5), (353, -1)):
                ax.add_patch(Rectangle((west, south), 2, 2, fill=False, ec="#c13b29", lw=2.2))
        ax.set(title=f"{name.upper()}   {receipt['joint_timed_fraction']:.1%} ($n$={receipt['sampled_positions']:,})",
               xlim=receipt["box"]["ra_bounds_deg"][::-1], ylim=receipt["box"]["dec_bounds_deg"],
               xlabel="ICRS RA [deg]", aspect="equal")
    axes[0].set_ylabel("ICRS Dec [deg]")
    fig.suptitle("SDSS + NSC r coverage with at least 0.60 yr of NSC r time span")
    fig.savefig(figures/"v09_exposure_footprint_summary.png", dpi=180)
    plt.close(fig)


def motion(pilots, figures):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = {"NSC": "#777777", "Xiao": "#a7622a", "C0": "#3b638c", "C1": "#009475"}
    for (field, data), row in zip(pilots.items(), axes):
        for ax, control in zip(row, ("qso", "gaia")):
            cohorts = data[control]["by_cohort"].get("all_four", {})
            values = [(x, cohorts.get(f"strict__all__{x}", {})) for x in
                      ("all", "lt20", "20_to_21", "21_to_22", "22_to_23")]
            bins = [(key, v) for key, v in values if "methods" in v]
            if bins:
                x = np.arange(len(bins))
                for j, (key, color) in enumerate(colors.items()):
                    ax.bar(x+(j-1.5)*.19, [v["methods"][key]["radial_abs68"] for _, v in bins],
                           width=.18, label=key, color=color)
                ax.set_xticks(x, [f"{key.replace('_to_', '–').replace('lt', '<')}\n$n$={v['n']}" for key,v in bins])
                ax.legend(frameon=False, fontsize=8, ncols=2)
            else:
                n = cohorts.get("strict__all__all", {}).get("n", 0)
                ax.text(.5, .5, f"Four-way common sample: {n}", transform=ax.transAxes, ha="center")
                ax.set_xticks([])
            ax.set(title=f"{field.replace('_',' ')} · {control.upper()}", ylabel="Radial P68 [mas yr$^{-1}$]")
    fig.suptitle("Strict same-source comparisons on the common four-catalogue sample")
    fig.savefig(figures/"v09_exposure_motion_summary.png", dpi=180)
    plt.close(fig)


def calibration(pilots, figures):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.0), constrained_layout=True, sharex=True)
    for ax, (field, data) in zip(axes, pilots.items()):
        galaxy = data["galaxies"]
        comparison = galaxy["comparison"]
        methods = comparison.get("methods", {})
        rows = [(f"All ($n$={comparison['n']:,})", [methods[m]["radial_abs68"] for m in ("C0", "C1")])] if methods else []
        rows.extend((f"{key.replace('_', ' ')} ($n$={v['n']:,})", v["radial_p68_mas"])
                    for key,v in sorted(galaxy["residual_strata"].items()))
        if not rows:
            ax.text(.5, .5, f"Held-out common galaxies: {comparison['n']}", transform=ax.transAxes, ha="center")
            continue
        pos = np.arange(len(rows))
        values = np.asarray([v for _,v in rows])
        for i, (label, color) in enumerate((("C0", "#3b638c"), ("C1", "#009475"))):
            ax.barh(pos+(i-.5)*.36, values[:,i], height=.34, label=label, color=color)
        ax.set_yticks(pos, [key for key,_ in rows], fontsize=8)
        ax.invert_yaxis()
        ax.set(title=field.replace("_", " "), xlabel="Radial P68 [mas]")
        ax.legend(frameon=False)
    fig.savefig(figures/"v09_exposure_galaxy_summary.png", dpi=180)
    plt.close(fig)


def zero_points(pilots, figures):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5), constrained_layout=True, sharey=True)
    rows = []
    for field, data in pilots.items():
        cohorts = data["qso"]["by_cohort"].get("c0_c1", {})
        for kind in ("strict", "blind"):
            control = cohorts.get(f"{kind}__all__all", {})
            rows.append((f"{field.replace('_',' ')} · {kind} ($n$={control.get('n',0)})",
                         control.get("methods", {})))
    for axis, ax in enumerate(axes):
        ax.axvspan(-.5, .5, color="#dadada", zorder=0)
        ax.axvline(0, color="0.35", lw=.7)
        for i, (method, color) in enumerate((("C0", "#3b638c"), ("C1", "#009475"))):
            for y, (_, control) in enumerate(rows):
                if method not in control:
                    continue
                stats = control[method]
                lo, hi = stats["median_bias95"][axis]
                pos = y+(i-.5)*.26
                ax.hlines(pos, lo, hi, color=color)
                ax.scatter(stats["median"][axis], pos, color=color, s=19)
            ax.plot([], [], "o-", color=color, label=method)
        ax.set_yticks(np.arange(len(rows)), [key for key,_ in rows], fontsize=9)
        if axis == 0:
            ax.invert_yaxis()
        ax.set(xlabel="Median QSO proper motion [mas yr$^{-1}$]", title=("East", "North")[axis])
    axes[-1].legend(frameon=False)
    fig.savefig(figures/"v09_exposure_zero_summary.png", dpi=180)
    plt.close(fig)


def support(pilots, figures):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)
    pos = np.arange(len(pilots))
    fine = [p["c1"] for p in pilots.values()]
    ticks = [name.replace("_", " ") for name in pilots]
    for j, (key, label, color) in enumerate((("candidate_children", "Candidate", "#c5c9cc"),
                                             ("supported_children", "C1 supported", "#009475"))):
        values = [row[key] for row in fine]
        bars = axes[0].bar(pos+(j-.5)*.34, values, width=.30, label=label, color=color)
        axes[0].bar_label(bars, [f"{v:,}" for v in values], fontsize=8, padding=3)
    axes[0].set(ylabel="Observation groups", title="CCD / SDSS field support")
    axes[0].legend(frameon=False, fontsize=8)
    spans = [p["galaxies"]["boundary"] for p in pilots.values()]
    for j, (key, label, color) in enumerate((
            ("cross_pixel_position", "HEALPix crossing", "#3b638c"),
            ("physical_detector", "Same-parent detector overlap", "#a7622a"))):
        values = [row.get(key, {}).get("n", 0) for row in spans]
        bars = axes[1].bar(pos+(j-.5)*.34, values, width=.30, label=label, color=color)
        axes[1].bar_label(bars, [str(v) for v in values], fontsize=8, padding=2)
    axes[1].set(ylabel="Held galaxy comparisons", title="Boundary support")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(pos, ticks)
        ax.margins(y=.26)
    fig.savefig(figures/"v09_exposure_support_summary.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/summary"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    geometry(args.summary, args.figures)
    if not args.geometry_only:
        pilots = {name: read(f"v09_{name}_exposure_pilot_*_summary.json", args.summary)
                  for name in ("repeat_rich", "repeat_sparse")}
        motion(pilots, args.figures)
        calibration(pilots, args.figures)
        zero_points(pilots, args.figures)
        support(pilots, args.figures)


if __name__ == "__main__":
    main()
