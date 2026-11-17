"""Render v0.8 quality figures from the aggregate allow list."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .__main__ import tomllib


DARK = "22_to_23"
PRIORITY = ("north_c", "east_d", "west_a", "crowd_a")
BINS = ("20_to_21", "21_to_22", "22_to_23")


def output(fig, folder, stem):
    fig.savefig(folder/f"v08_{stem}_summary.png",dpi=170,bbox_inches="tight")
    plt.close(fig)


def context(matrix, cohort, folder):
    fig,ax = plt.subplots(figsize=(9,4.3), layout="constrained")
    marker = {"equator":"o","stripe":"v","north":"s","mid":"d","alto":"d","arc":"o",
              "south":"^","cap":"P"}
    complete = cohort["fields"] if cohort else {}
    for ix in range(1,5):
        rows = [(name,row) for name,row in matrix["fields"].items() if row["batch"]==ix]
        for j,(name,r) in enumerate(rows):
            ax.scatter((r["ra"]+105)%360-105,r["dec"],color=f"C{ix-1}",
                       marker=marker.get(r["role"],"o"),s=30 if name in complete else 23,
                       alpha=1 if name in complete else .45,
                       label=f"Batch {ix}" if j==0 else None)
    older=[r["center_deg"] for name,r in complete.items() if name not in matrix["fields"]]
    if older:
        x,y=np.array(older).T
        ax.scatter((x+105)%360-105,y,c="0.35",s=16,label="v0.7")
    ax.set(xlabel="Right ascension (deg, wrapped)",ylabel="Declination (deg)",
           title="Frozen geometry and completed survey cores")
    ax.legend(ncol=5,frameon=False,loc="best")
    output(fig,folder,"footprint")


def historical(audits, history, folder):
    fig,ax=plt.subplots(2,2,figsize=(10,6.5),layout="constrained")
    for panel,field in zip(ax.flat,PRIORITY):
        row=audits[field]["zero"]
        for axis,label in enumerate(("East","North")):
            for j,mag in enumerate(BINS):
                item=row.get("magnitude__"+mag, {})
                if "median" not in item:
                    continue
                lo,hi=item["median95"][axis]
                xp=j+(axis-.5)*.21
                panel.vlines(xp,lo,hi,color=f"C{axis}",alpha=.75)
                panel.scatter(xp,item["median"][axis],color=f"C{axis}",s=25,
                              label=label if j==0 else None)
        panel.axhspan(-.5,.5,color="0.9")
        panel.axhline(0,color="0.5",lw=.7)
        panel.set(title=field,ylabel="Median proper motion (mas/yr)")
        panel.set_xticks(range(3),[f"{b[:2]}–{b[-2:]}\n n={row.get('magnitude__'+b,{}).get('n',0)}" for b in BINS])
    ax[0,0].legend(frameon=False,ncol=2)
    fig.supxlabel("Spectroscopic QSO reference r (mag)")
    output(fig,folder,"historical_zero")

    if history:
        fig,ax=plt.subplots(1,2,figsize=(9,3.6),layout="constrained",sharey=True)
        for a,truth,label in zip(ax,("qso","gaia"),("DESI QSO","Gaia DR3 stars")):
            for j,band in enumerate(BINS):
                points=[(float(x),v[band]) for x,v in history["morphology"][truth].items()
                        if band in v and v[band]["denominator"]]
                points.sort()
                if not points:
                    continue
                x, f = zip(*points)
                m=np.array([p["value"] for p in f])
                lo,hi=np.array([p["wilson95"] for p in f]).T
                a.plot(x,m,"o-",color=f"C{j}",label=f"{band[:2]}–{band[-2:]}  n={f[0]['denominator']}")
                a.fill_between(x,lo,hi,color=f"C{j}",alpha=.12)
            a.set(title=label,xlabel="NSC r median CLASS_STAR cut",ylim=(0,1.01))
            a.axvline(.8,ls=":",c="0.4")
            a.legend(frameon=False)
        ax[0].set_ylabel("Correctly recovered fraction (95% Wilson)")
        output(fig,folder,"morphology")


def performance(cohort, folder):
    fig,ax=plt.subplots(1,2,figsize=(10,4.1),layout="constrained")
    pooled=cohort["pooled"]
    for i,tag in enumerate(BINS):
        stats=pooled.get(tag,{}).get("W0",{}).get("statistics",{})
        for offset,model,label,color in [(-.19,"published_nsc","NSC DR2","0.6"),
                                         (.19,"W0","SDSS+NSC, W0","C0")]:
            measures=stats.get("methods",{}).get(model)
            if measures:
                ax[0].bar(i+offset,measures["radial_abs68"],width=.37,color=color,label=label if i==0 else None)
        n=stats.get("n",0)
        ax[0].text(i,0,f"n={n}",ha="center",va="bottom",fontsize=8)
    ax[0].set_xticks(range(3),["20–21","21–22","22–23"])
    ax[0].set(xlabel="QSO reference r (mag)",ylabel="Paired radial P68 (mas/yr)",
              title=f"{cohort['area_sqdeg']:.1f} deg² completed")
    ax[0].legend(frameon=False)
    fields=sorted(cohort["fields"])
    for i,name in enumerate(fields):
        interval=cohort["fields"][name]["zero95"]
        if interval is None:
            continue
        for axis in range(2):
            x=i+(axis-.5)*.30
            lo,hi=interval["zero95_masyr"][axis]
            ax[1].vlines(x,lo,hi,color=f"C{axis}",alpha=.66)
            values=cohort["fields"][name]["faint_precision"]["methods"]["W0"]["median"]
            ax[1].scatter(x,values[axis],color=f"C{axis}",s=12)
    ax[1].axhspan(-.5,.5,color="0.9")
    ax[1].set(xlabel="Completed cores (alphabetical)",ylabel="Zero point and object CI (mas/yr)",
              title="r = 22–23 by core")
    if len(fields)<=28:
        ax[1].set_xticks(range(len(fields)),fields,rotation=90,fontsize=6)
    output(fig,folder,"precision")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries",type=Path,default=Path("results/summary"))
    parser.add_argument("--figures",type=Path,default=Path("figures"))
    parser.add_argument("--matrix",type=Path,default=Path("configs/spatial-v08-fields.toml"))
    parser.add_argument("--cohort",type=Path)
    parser.add_argument("--history",type=Path)
    args=parser.parse_args()
    audits={}
    for f in sorted(args.summaries.glob("v08_*_spatial_audit_v08_*_summary.json")):
        row=json.loads(f.read_text())
        audits[row["region"]]=row
    hist=None if args.history is None else json.loads(args.history.read_text())
    cohort=None if args.cohort is None else json.loads(args.cohort.read_text())
    args.figures.mkdir(exist_ok=True,parents=True)
    if all(k in audits for k in PRIORITY):
        historical(audits,hist,args.figures)
    if args.matrix.is_file():
        context(tomllib.loads(args.matrix.read_text()),cohort,args.figures)
    if cohort:
        performance(cohort,args.figures)


if __name__=="__main__":
    main()
