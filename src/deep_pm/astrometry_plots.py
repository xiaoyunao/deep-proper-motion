"""Render the weight, repeated-scan and multiband pilot aggregates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .epoch_statistics import SCAN_BINS


FIELDS = {"g09": "G09", "g15": "G15", "ngc_mid": "North mid", "egs": "EGS",
          "ngc_short": "North short", "s82_east": "Stripe 82 east*"}


def weight_plot(fields, directory):
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for ax,key,title in zip(axes,("qso_all","qso_22_to_23"),("All common QSOs",r"$22\leq r<23$")):
        ax.axvline(0,color=".45",lw=1)
        labels = []
        for y,(name,record) in enumerate(fields.items()):
            item = record["controls"]["motion"].get(key, {})
            labels.append(f'{FIELDS[name]} ({item.get("n",0)})')
            for label,shift,color in (("W1",-.12,"#187586"),("W2",.12,"#c2662d")):
                entry = item.get("paired",{}).get(f"{label}_vs_W0",{})
                if not entry or entry["fraction95"][2] is None:
                    continue
                low,high = np.asarray(entry["fraction95"][2])*100
                point = entry["fraction_improved"][2]*100
                ax.plot([low,high],[y+shift,y+shift],color=color)
                ax.scatter(point,y+shift,s=19,color=color,label=label if y==0 else "_nolegend_")
        ax.set_yticks(range(len(fields)),labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Change in radial P68 vs W0 (%)\nPositive values indicate improvement")
        ax.grid(axis="x",alpha=.23)
    axes[0].legend(frameon=False)
    fig.savefig(directory/"v06_weight_qso_summary.png",dpi=170)
    plt.close(fig)


def coverage_plot(fields,directory):
    fig,axes = plt.subplots(1,2,figsize=(11.5,5.2))
    fig.subplots_adjust(left=.14,right=.97,top=.92,bottom=.25,wspace=.42)
    colors = plt.get_cmap("Blues")(np.linspace(.22,.9,len(SCAN_BINS)))
    labels = ["0","1","2","3–5","6–10",">10"]
    for ax,mode in zip(axes,("source","area")):
        for y,summary in enumerate(fields.values()):
            coverage = summary["coverage"][mode]
            if mode=="source":
                counts = [coverage["all"]["scans"][k]["value"] for k in SCAN_BINS]
            else:
                counts = [coverage["scan_area_sqdeg"][k]/coverage["core_area_sqdeg"] for k in SCAN_BINS]
            left = 0.
            for value,color,label in zip(counts,colors,labels):
                ax.barh(y,100*value,left=100*left,color=color,edgecolor="white",height=.65,
                        label=label if y==0 else "_nolegend_")
                left += value
        ax.set_yticks(range(len(fields)),[FIELDS[name] for name in fields])
        ax.invert_yaxis()
        ax.set_xlim(0,100)
        ax.set_xlabel("Fraction (%)")
        ax.set_title("Sources with an NSC r anchor" if mode=="source" else "DR17 geometry in the core")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,title="SDSS physical scans",ncol=6,frameon=False,
               loc="lower center",bbox_to_anchor=(.5,.055),fontsize=8)
    fig.text(.1,.018,"* Stripe 82 source sample is targeted. Area uses the ordinary DR17 window index.",fontsize=8)
    fig.savefig(directory/"v06_sdss_scans_summary.png",dpi=170)
    plt.close(fig)


def scans_by_magnitude(fields,directory):
    fig, axes = plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True,layout="constrained")
    labels = ["0","1","2","3–5","6–10",">10"]
    colors = plt.get_cmap("tab10").colors
    for panel,ax in enumerate(axes.flat):
        tag,title = (
            ("lt20",r"$r<20$"), ("20_to_21",r"$20\leq r<21$"),
            ("21_to_22",r"$21\leq r<22$"), ("22_to_23",r"$22\leq r<23$"),
        )[panel]
        for j,(name,summary) in enumerate(fields.items()):
            source = summary["coverage"]["source"].get(tag)
            if not source:
                continue
            values = [100*source["scans"][key]["value"] for key in SCAN_BINS]
            ax.plot(labels,values,marker="o",ms=3,lw=1,color=colors[j],
                    label=f"{FIELDS[name]} ({source['n']:,})")
        ax.set_title(title)
        ax.grid(axis="y",alpha=.2)
        ax.legend(frameon=False,fontsize=7,ncol=2)
    fig.supxlabel("Observed SDSS scans per NSC r anchor")
    fig.supylabel("Sources (%)")
    fig.savefig(directory/"v06_sdss_scans_by_r_summary.png",dpi=170)
    plt.close(fig)


def multiband_plot(fields,directory):
    fig,ax = plt.subplots(figsize=(8,4.8),layout="constrained")
    key = "22_to_23"
    for y,(name,summary) in enumerate(fields.items()):
        multi = summary["multiband"]
        for method,delta,color,short in (
                ("W0_r",-.22,"#5b5b5b","W0 r"),("W1_r",0.,"#77a9ca","W1 r"),
                ("W1_multiband",.22,"#24677f","W1 g/r/i/z")):
            stats = multi[method].get(key)
            if stats:
                v = stats["final"]
                ax.barh(y+delta,v,height=.2,color=color,label=short if y==0 else "_nolegend_")
                ax.text(v+5,y+delta,str(v),va="center",fontsize=8)
    ax.set_title(r"Blind trial solutions at $22\leq r<23$")
    ax.set_yticks(range(len(fields)),[FIELDS[name] for name in fields])
    ax.invert_yaxis()
    ax.set_xlabel("Retained solutions")
    ax.legend(frameon=False)
    ax.margins(x=.1)
    fig.savefig(directory/"v06_multiband_yield_summary.png",dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path,nargs="+")
    p.add_argument("--figures",type=Path,default=Path("figures"))
    args = p.parse_args()
    records = [json.loads(path.read_text()) for path in args.input]
    lookup = {row["region"]:row for row in records}
    fields = {name:lookup[name] for name in FIELDS if name in lookup}
    args.figures.mkdir(parents=True,exist_ok=True)
    for plot in (weight_plot,coverage_plot,scans_by_magnitude,multiband_plot):
        plot(fields,args.figures)


if __name__=="__main__":
    main()
