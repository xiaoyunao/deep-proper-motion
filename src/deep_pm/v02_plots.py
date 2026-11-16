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
    if len(bins) == 1:
        entry = bins[0]
        stages = (("attempted", "Attempted"), ("two_survey_geometry", "Geometry"),
                  ("required_time_coverage", "Usable sampling"),
                  ("generated_correct_candidate", "Correct proposal"),
                  ("selected_recovery", "Selected"), ("secure_recovery", "Secure"))
        counts = [entry[key] for key, _ in stages]
        fig, ax = plt.subplots(figsize=(7.1, 3.5), layout="constrained")
        ax.barh(np.arange(len(stages)), counts, color=COLORS[0], height=.6)
        ax.set(yticks=np.arange(len(stages)), yticklabels=[label for _, label in stages],
               xlabel="Injected trajectories", xlim=(0, entry["attempted"] * 1.12),
               title=f"Northern field | {speeds[0]} mas/yr injection")
        for k, count in enumerate(counts):
            ax.text(count + .35, k, str(count), va="center")
        ax.invert_yaxis()
        fig.savefig(out / f"v02_{row['region']}_injection_summary.png", dpi=175)
        plt.close(fig)
        return
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


def plot_matched(row, out):
    methods = ("motion/all_selected", "motion/secure")
    groups = [(tag, row["matched_pm"][tag]) for tag in methods
              if row["matched_pm"][tag]["blind"]["n"]]
    if not groups:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), layout="constrained")
    for axis, ax in enumerate(axes):
        for k, (tag, values) in enumerate(groups):
            marker = ("o", "s")[k]
            label = "All selected" if tag.endswith("all_selected") else "Secure"
            for j, method in enumerate(("blind", "fixed_all_visits")):
                data = values[method]
                ax.scatter([j], [data["nmad_masyr"][axis]], s=49, marker=marker,
                           color=COLORS[k], label=f"{label} (n={data['n']})" if j == 0 else None)
            ax.plot([0, 1], [values[m]["nmad_masyr"][axis] for m in ("blind", "fixed_all_visits")],
                    color=COLORS[k], linewidth=1, alpha=.7)
        ax.set(xticks=[0, 1], xticklabels=["Blind association", "Fixed visits"],
               xlim=(-.38, 1.38), title=("East", "North")[axis])
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("PM residual NMAD (mas/yr)")
    axes[1].legend(frameon=False)
    fig.suptitle(f"{row['region']} | same matched Gaia sources in each comparison")
    fig.savefig(out / f"v02_{row['region']}_matched_precision_summary.png", dpi=175)
    plt.close(fig)


def plot_prediction(row, out):
    held = row["held_year"]
    if not held:
        return
    hist = held["methods"]
    methods = ("fixed", "nsc_like", "motion")
    names = ("Fixed radius", "NSC-like", "Motion-aware")
    slices = (("with_gaia", "labelled_correct", "Gaia-labelled matches"),
              ("no_gaia", "consistent_unconfirmed", "Consistent, unverified matches"))
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.4), layout="constrained")
    for ax, (sample, key, title) in zip(axes, slices):
        for j, (method, name) in enumerate(zip(methods, names)):
            cell = hist[method]["all_selected"].get(f"cross_survey/{sample}/r_lt_20p5", {})
            n, total = cell.get(key, 0), cell.get("attempted", 0)
            if total:
                ax.barh(j, n / total, height=.54, color=COLORS[j], alpha=.8)
                ax.text(n / total + .023, j, f"{n}/{total}", va="center", fontsize=9)
        ax.set(yticks=range(3), yticklabels=names, xlim=(0, 1.26), xticks=[0, .25, .5, .75, 1],
               title=title, xlabel="Fraction of prediction visits")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.18)
    fig.suptitle(f"{row['region']} | held-out {held['year']} | crosses surveys, NSC r < 20.5")
    fig.savefig(out / f"v02_{row['region']}_held_year_summary.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["regression", "injection", "matched", "prediction"])
    parser.add_argument("out", type=Path)
    parser.add_argument("summaries", type=Path, nargs="+")
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in args.summaries]
    plt.rcParams.update({"font.family": "serif", "xtick.direction": "in", "ytick.direction": "in"})
    args.out.mkdir(exist_ok=True, parents=True)
    if args.kind == "regression":
        plot_regression(rows, args.out)
    else:
        renderer = {"injection": plot_injection, "matched": plot_matched, "prediction": plot_prediction}[args.kind]
        for row in rows:
            renderer(row, args.out)
