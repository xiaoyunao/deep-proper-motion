"""Figures from public held-out, regression, or injection aggregates."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .field_statistics import optical_summary


STEPS = ("has_input", "generated", "screened", "retained", "conflict_optimal",
         "selected_optimal", "secure", "pm_compatible_secure")
LABELS = ("Eligible", "Generated", "Screened", "Retained", "Optimal\ncomponent",
          "Selected", "Secure", "Secure +\nGaia PM")
COLORS = ("#29567e", "#a64d36", "#3c785c", "#665089")


def plot_regression(rows, out):
    rows = sorted(rows, key=lambda row: row["region"])
    steps = np.arange(len(STEPS))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), layout="constrained",
                             gridspec_kw={"width_ratios": [2.15, 1]})
    left, right = axes
    for k, row in enumerate(rows):
        metrics = row["validation"]["recovery_stages"]
        if not metrics[STEPS[0]]["total"]["denominator"]:
            continue
        n = metrics[STEPS[0]]["total"]["denominator"]
        points = [metrics[name]["total"]["value"] for name in STEPS]
        left.plot(steps, points, "o-", label=f"{row['region']} ({n})",
                  color=COLORS[k % len(COLORS)], alpha=.87, linewidth=1.3)
    left.set(xticks=steps, xticklabels=LABELS, ylim=(-.04, 1.05),
             ylabel="Fraction of eligible Gaia sources", title="Motion-aware recovery")
    left.tick_params(axis="x", labelsize=8)
    left.legend(loc="lower left", frameon=False, fontsize=8, ncols=2)
    stats = optical_summary(rows)["regression"]
    for k, (key, label) in enumerate((("fixed", "Fixed radius"), ("nsc_like", "NSC-like"),
                                       ("motion", "Motion-aware"))):
        lower, upper = stats[key]["contamination_identification_bounds"]
        right.plot([lower * 100, upper * 100], [k, k], "|-", color=COLORS[k],
                   markersize=10, linewidth=2)
        unknown = stats[key]["unknown_objects"]
        right.text(upper * 100, k + .16, f"{unknown['n']}/{unknown['denominator']} unknown",
                   va="bottom", ha="right", fontsize=8)
    right.set(yticks=[0, 1, 2], yticklabels=["Fixed radius", "NSC-like", "Motion-aware"],
              xlim=(0, 101), ylim=(-.38, 2.55), xlabel="Contamination bounds (%)",
              title="Unlabelled-object sensitivity")
    for ax in axes:
        ax.grid(alpha=.18, linewidth=.5)
    fig.savefig(out / "v02_regression_funnel_summary.png", dpi=175)
    plt.close(fig)


def plot_injection(row, out):
    speeds = sorted(row["bins"], key=float)
    bins = [row["bins"][speed] for speed in speeds]
    x = np.arange(len(bins))
    pairs = (("two_survey_geometry", "Geometry"), ("required_time_coverage", "Usable sampling"),
             ("generated_correct_candidate", "Correct proposal"), ("secure_recovery", "Secure"))
    fig, ax = plt.subplots(figsize=(8.2, 3.7), layout="constrained")
    for k, (key, label) in enumerate(pairs):
        values = [entry[key] / entry["attempted"] for entry in bins]
        ax.plot(x, values, "o-", label=label, color=COLORS[k])
    for i, row_bin in enumerate(bins):
        ax.text(i, .91, str(row_bin["attempted"]), ha="center", fontsize=8)
    ax.set(xlabel="Injected motion (mas/yr)", ylabel="Fraction of attempts",
           xticks=x, xticklabels=speeds, ylim=(-.05, 1.04),
           title=f"{row['region']} | injected sources on actual exposure geometry")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(out / f"v02_{row['region']}_injection_summary.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["regression", "injection"])
    parser.add_argument("out", type=Path)
    parser.add_argument("summaries", type=Path, nargs="+")
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in args.summaries]
    plt.rcParams.update({"font.family": "serif", "xtick.direction": "in", "ytick.direction": "in"})
    args.out.mkdir(exist_ok=True, parents=True)
    if args.kind == "regression":
        plot_regression(rows, args.out)
    else:
        for row in rows:
            plot_injection(row, args.out)
