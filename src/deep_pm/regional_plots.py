"""Draw spatial zero points, blind QSO precision and regional selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from .pilot_production import STAGES


def draw_zeros(summary,directory):
    fields=summary["fields"]
    figure, axes=plt.subplots(2,1,figsize=(10,8),sharex=True,layout="constrained")
    cm=plt.get_cmap("coolwarm")
    norm=TwoSlopeNorm(vmin=-4,vcenter=0,vmax=4)
    for axis_index,axis in enumerate(axes):
        for region,row in fields.items():
            ra,dec=row["center_deg"]
            item=row["faint_precision"].get("methods",{}).get("W0")
            n=row["faint_precision"].get("n",0)
            zero=item["median"][axis_index] if item else None
            if zero is None or n<5:
                axis.scatter(ra,dec,facecolors="none",edgecolors=".45",s=30)
            else:
                axis.scatter(ra,dec,c=[zero],cmap=cm,norm=norm,s=35+4*n,edgecolors=".2",lw=.5)
            axis.annotate(f"{region} ({n})",(ra,dec),xytext=(4,5),textcoords="offset points",fontsize=7)
        axis.grid(alpha=.2)
        axis.set_ylabel("Dec (deg)")
        axis.set_title(("East","North")[axis_index]+r" median residual, blind W0 QSO ($22\leq r<23$)")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("RA (deg)")
    figure.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),ax=axes,label="mas/yr",shrink=.8)
    figure.savefig(directory/"v07_spatial_zero_summary.png",dpi=170)
    plt.close(figure)


def draw_selection(summary,directory):
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    names=["Initial","Near SDSS","Associated","Time","M3","Final"]
    for tag,label,method in (("lt20","r < 20","W0"),("20_to_21","20–21","W0"),
                             ("21_to_22","21–22","W0"),("22_to_23","22–23","W0"),
                             ("22_to_23","22–23 extra bands","multi")):
        cohort=summary["counts"].get(method,{}).get(tag,{})
        if not cohort.get("initial"):
            continue
        y=np.array([cohort[k] for k in STAGES])/cohort["initial"]
        axes[0].plot(names,100*y,marker="o",lw=1.4,
                     label=f"{label}, n={cohort['initial']:,}",ls="--" if method=="multi" else "-")
    axes[0].set_ylabel("Initial anchor fraction (%)")
    axes[0].legend(frameon=False,fontsize=8)
    axes[0].grid(axis="y",alpha=.25)
    axes[0].set_ylim(0,108)
    axes[0].set_title("Blind production funnel")
    fields=summary["fields"]
    for y,(field,item) in enumerate(fields.items()):
        w=item.get("faint_precision",{})
        n=w.get("n",0)
        methods=w.get("methods")
        if n and methods:
            a,b=[methods[m]["radial_abs68"] for m in ("published_nsc","W0")]
            axes[1].plot([a,b],[y,y],c=".65",lw=1)
            axes[1].scatter(a,y,s=18,c="#666666",label="NSC DR2" if y==0 else "_nolegend_")
            axes[1].scatter(b,y,s=21,c="#007e8c",label="W0" if y==0 else "_nolegend_")
    axes[1].set_yticks(range(len(fields)),[f"{k} ({row['faint_precision'].get('n',0)})" for k,row in fields.items()],fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Radial P68 (mas/yr)")
    axes[1].set_title(r"Common blind QSO, $22\leq r<23$")
    axes[1].grid(axis="x",alpha=.2)
    handles,labels=axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles,labels,frameon=False)
    fig.savefig(directory/"v07_regional_selection_summary.png",dpi=170)
    plt.close(fig)


def draw_holdout(summary,directory):
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    fields=summary["fields"]
    for i,(k,row) in enumerate(fields.items()):
        add=row["multiband"]["gained"].get("22_to_23",0)
        loss=row["multiband"]["lost"].get("22_to_23",0)
        axes[0].barh(i,add,color="#027a7a",label="Gained" if i==0 else "_nolegend_")
        axes[0].barh(i,-loss,color="#d07439",label="Lost" if i==0 else "_nolegend_")
        result=row["holdout"].get("by_r",{}).get("new",{}).get("22_to_23",{})
        if result:
            successes=result["matched_sources"]["n"]
            trials=result["train_sources"]
            axes[1].barh(i,trials,color="#b4c7d4",label="Train candidates" if i==0 else "_nolegend_")
            axes[1].barh(i,successes,color="#027a7a",label="Unique held match" if i==0 else "_nolegend_")
    for ax,title in zip(axes,("All visits, extra NSC bands vs W1 r","Whole-year holdout, extra-band-only sources")):
        ax.set_yticks(range(len(fields)),list(fields),fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(r"Sources at $22\leq r<23$")
        ax.grid(axis="x",alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False,fontsize=8)
    fig.savefig(directory/"v07_multiband_holdout_summary.png",dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("aggregate",type=Path)
    p.add_argument("--figures",type=Path,default=Path("figures"))
    args=p.parse_args()
    values=json.loads(args.aggregate.read_text())
    args.figures.mkdir(parents=True,exist_ok=True)
    for draw in (draw_zeros,draw_selection,draw_holdout):
        draw(values,args.figures)


if __name__=="__main__":
    main()
