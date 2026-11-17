"""Render v1.0 result figures from validated public aggregate receipts."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NAMES = {
    "repeat_rich": "Repeat rich",
    "repeat_sparse": "Original sparse",
    "sparse_north_inner": "North inner",
    "sparse_north_outer": "North outer",
    "sparse_south_east": "South east",
    "sparse_south_east_ready": "South east (east half)",
}
COLORS = ["#4479a4", "#008b70"]


def comparison(ax, data, kind, cohort, heading):
    ticks=[]
    for i, (field, output) in enumerate(data.items()):
        stats=output[kind]["by_cohort"].get("c0_c1",{}).get(cohort,{})
        ticks.append(f"{NAMES[field]}\nn={stats.get('n',0):,}")
        for j,key in enumerate(("C0","C1")):
            if key in stats.get("methods",{}):
                ax.bar(i+(j-.5)*.34,stats["methods"][key]["radial_abs68"],width=.32,color=COLORS[j])
    for label,color in zip(("C0","C1"),COLORS):
        ax.bar([],[],color=color,label=label)
    ax.set_ylabel("Radial proper-motion P68 [mas/yr]")
    ax.set_xticks(range(len(ticks)),ticks,fontsize=8)
    ax.set_title(heading)
    ax.legend(frameon=False)


def motion(data, output):
    fig, ax=plt.subplots(2,1,figsize=(9,7),layout="constrained")
    comparison(ax[0],data,"qso","strict__all__21_to_23","Strict spectroscopic QSOs, 21 ≤ r < 23")
    comparison(ax[1],data,"gaia","strict__all__all","Quality-selected Gaia stars, NSC r < 21")
    fig.savefig(output/"v10_motion_controls_summary.png",dpi=170)
    plt.close(fig)


def quality(data, output):
    names=("passed","formal_only","residual_only","both")
    labels=("Passed","Formal error","Residual","Both")
    fig, ax=plt.subplots(1,figsize=(10,4.2),layout="constrained")
    pos=np.arange(len(data))
    lower=np.zeros(len(data))
    for reason,label,color in zip(names,labels,["#008b70","#ba7857","#54799a","#655283"]):
        values=[d["dark_quality"]["C1"]["by_reason"][reason]["n"] for d in data.values()]
        ax.bar(pos,values,bottom=lower,label=label,color=color)
        lower+=values
    ax.set_xticks(pos,[f"{NAMES[name]}\nn={int(n):,}" for name,n in zip(data,lower)])
    ax.set(ylabel="Number with common time and calibration support",title="C1 outcomes for fixed r-band anchors, 22 ≤ r < 23")
    ax.legend(ncols=4,frameon=False)
    fig.savefig(output/"v10_dark_quality_summary.png",dpi=170)
    plt.close(fig)


def boundary(data, output):
    fig, ax=plt.subplots(1,figsize=(10,5),layout="constrained")
    types={"nsc_same_exposure_ccd":"NSC same exposure / CCD","sdss_adjacent_field":"SDSS adjacent field",
           "near_healpix_edge":"HEALPix near-edge","near_partition_seam":"Partition near seam"}
    tick=[]
    for field,d in data.items():
        for key,value in d["physical_boundaries"].items():
            if key in types and value.get("pair_count",0):
                tick.append((f"{NAMES[field]} · {types[key]}",value))
    for i,(name,v) in enumerate(tick):
        for j,color in enumerate(COLORS):
            ax.plot(v["radial_p68_mas"][j+1],i+.16*(j-.5),".",color=color)
    ax.set_yticks(range(len(tick)),[f"{key}, n={v['pair_count']:,}" for key,v in tick],fontsize=8)
    ax.set_xlabel("Radial positional difference P68 [mas]")
    ax.set_title("Independent held-galaxy boundary diagnostics")
    for label,color in zip(("C0","C1"),COLORS):
        ax.plot([],[],".",color=color,label=label)
    ax.legend(frameon=False)
    fig.savefig(output/"v10_boundary_summary.png",dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,default=Path("results/summary"))
    p.add_argument("--output",type=Path,default=Path("figures"))
    a=p.parse_args()
    data={}
    for field in NAMES:
        if field=="sparse_south_east_ready" and "sparse_south_east" in data:
            continue
        paths=list(a.summary.glob(f"v10_{field}_exposure_pilot_*.json"))
        if paths:
            if len(paths)!=1:
                p.error(f"Multiple receipts for {field}")
            data[field]=json.loads(paths[0].read_text())
            bright=list(a.summary.glob(f"v10_{field}_validation_bright_*.json"))
            if bright:
                if len(bright)!=1:
                    p.error(f"Multiple bright-only receipts for {field}")
                data[field]["gaia"]=json.loads(bright[0].read_text())["gaia"]
            edges=list(a.summary.glob(f"v10_{field}_validation_existing_*.json"))
            if edges:
                if len(edges)!=1:
                    p.error(f"Multiple physical-boundary receipts for {field}")
                checked=json.loads(edges[0].read_text())
                data[field]["physical_boundaries"]["sdss_adjacent_field"]=checked["sdss_adjacent_field"]
    if not data:
        p.error("No validated pilot results")
    motion(data,a.output)
    quality(data,a.output)
    boundary(data,a.output)


if __name__=="__main__":
    main()
