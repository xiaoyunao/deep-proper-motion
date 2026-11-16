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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("bright_summaries",nargs="+",type=Path)
    args=p.parse_args()
    rows=[json.loads(path.read_text()) for path in args.bright_summaries]
    with plt.rc_context({"font.family":"serif","axes.spines.top":False,"axes.spines.right":False}):
        plot_bright(rows,args.output)


if __name__=="__main__":
    main()
