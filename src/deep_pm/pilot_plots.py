"""Figures from allow-listed regional pilot summaries."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BINS = {"lt20":"<20","20_to_21":"20–21","21_to_22":"21–22","22_to_23":"22–23"}
LABELS = {"cosmos":"COSMOS","g15":"GAMA G15","g09":"GAMA G09","egs":"EGS",
          "ngc_short":"North (short)","ngc_mid":"North (mid)"}


def plot_funnel(summary, output):
    steps = ("initial","sdss_nearby","associated","fit_eligible","calibrated","final")
    fig,ax = plt.subplots(figsize=(8,3.8),layout="constrained")
    for name,label in BINS.items():
        bucket = summary["funnel"].get(name,{})
        initial = bucket.get("initial",0)
        if initial:
            ax.plot([100*bucket.get(step,0)/initial for step in steps],"o-",
                    label=f"r {label} ($n={initial:,}$)",linewidth=1.5,markersize=4)
    ax.set_xticks(range(len(steps)),["Initial r\nsource","Nearby\nSDSS","Associated",
                                   "Time\ncoverage","M3\nsupport","Final\nquality"])
    ax.set_ylabel("Fraction of initial sources (%)")
    ax.set_ylim(0,104)
    ax.legend(frameon=False,ncols=2)
    fig.savefig(output,dpi=165)
    plt.close(fig)


def plot_qso(summaries, output):
    rows = []
    for doc in summaries:
        qso = doc["validation"]["qso"]
        anchored = qso["anchored"].get("22_to_23",{})
        blind = qso["common_blind"].get("22_to_23",{})
        for label, data, method in [("Known positions",anchored,"M3"),("Blind positions",blind,"M3_blind")]:
            result = data.get("paired",{}).get(method+"_vs_published_nsc")
            if result is not None and result["fraction95"][2] is not None:
                rows.append((LABELS.get(doc["region"],doc["region"]),label,data["n"],
                             result["fraction_improved"][2],result["fraction95"][2]))
    if not rows:
        return
    fig,ax = plt.subplots(figsize=(8,3.8),layout="constrained")
    palette = {"Known positions":"#31776f","Blind positions":"#854964"}
    for i, (_,kind,_,point,interval) in enumerate(rows):
        color = palette[kind]
        ax.hlines(i, interval[0]*100,interval[1]*100,color=color,linewidth=1.8)
        ax.scatter(point*100,i,s=22,color=color,zorder=3)
    ax.set_yticks(np.arange(len(rows)),
                  [f"{name}, {kind} ($n={n}$)" for name,kind,n,*_ in rows])
    ax.axvline(0,color=".6",linewidth=.8)
    ax.axvline(15,color=".6",linewidth=.8,linestyle="--")
    ax.invert_yaxis()
    ax.set_xlabel("Gain in radial P68 relative to published NSC (%)")
    ax.set_title("Spectroscopic quasars at r = 22–23")
    fig.savefig(output,dpi=165)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pool",type=Path)
    p.add_argument("funnel",type=Path)
    p.add_argument("qso",type=Path)
    p.add_argument("fields",type=Path,nargs="+")
    args = p.parse_args()
    with plt.rc_context({"font.family":"serif","axes.spines.top":False,"axes.spines.right":False}):
        plot_funnel(json.loads(args.pool.read_text()),args.funnel)
        plot_qso([json.loads(path.read_text()) for path in args.fields],args.qso)


if __name__=="__main__":
    main()
