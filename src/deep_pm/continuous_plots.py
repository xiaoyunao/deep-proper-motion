"""Plot the measured C0 footprint and controls on the frozen 10° square."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np


def render(data, output):
    cores=data["cores"]
    boxes=data["covered_cores"]
    p68={}
    zero={}
    support={}
    count={}
    repeat={}
    errors={}
    for k,item in cores.items():
        q=item["qso"]["by_cohort"].get("c0",{}).get("strict__all__22_to_23",{})
        c=q.get("methods",{}).get("C0",{})
        p68[k]=(c.get("radial_abs68"),q.get("n",0))
        zero[k]=(c.get("median",[None])[0],q.get("n",0))
        ns=item["observations"]["NSC"]
        sd=item["observations"]["SDSS"]
        d=ns["n_rows"]+ns["unconstrained"]+sd["n_rows"]+sd["unconstrained"]
        support[k]=((ns["n_rows"]+sd["n_rows"])/d*100 if d else None,None)
        count[k]=(item["owned_w0"]/boxes[k]["area_sqdeg"],None)
        repeat[k]=(item["leverage"]["repeat_sdss"]["n"]/item["owned_w0"]*100 if item["owned_w0"] else None,None)
        err=item["formal_error_masyr"]
        errors[k]=(max(err[1]) if err is not None else None,None)
    fig, ax=plt.subplots(2,3,figsize=(12,8),layout="constrained")
    for a, (title, mapped, cmap, diverging) in zip(ax.ravel(),[
            ("W0 sources / deg²",count,"viridis",False),
            ("Supported r detections [%]",support,"viridis",False),
            ("W0 sources with repeated SDSS [%]",repeat,"viridis",False),
            ("QSO radial P68 [mas/yr], 22 ≤ r < 23",p68,"viridis",False),
            ("QSO east zero [mas/yr], 22 ≤ r < 23",zero,"RdBu_r",True),
            ("Median formal axis error [mas/yr]",errors,"viridis",False)]):
        values=[v for v,_ in mapped.values() if v is not None]
        if diverging:
            vmax=max(abs(np.asarray(values)),default=1) or 1
            norm=TwoSlopeNorm(0,vmin=-vmax,vmax=vmax)
        else:
            vmax=max(values,default=1)
            norm=Normalize(0,vmax or 1)
        colors=plt.get_cmap(cmap)
        a.add_patch(Rectangle((345,-5),10,10,facecolor="#eeeeee",edgecolor="none"))
        for k,(v,n) in mapped.items():
            box=boxes[k]
            lo, hi=box["ra_bounds_deg"]
            south,north=box["dec_bounds_deg"]
            a.add_patch(Rectangle((lo,south),hi-lo,north-south,
                                  facecolor=colors(norm(v)) if v is not None else "#d4d4d4",
                                  edgecolor="white",linewidth=1))
            if n is not None:
                a.text((lo+hi)/2,(south+north)/2,str(n),ha="center",va="center",fontsize=8,
                       bbox=dict(facecolor="white",alpha=.85,edgecolor="none",pad=1))
        a.set(xlim=(355,345),ylim=(-5,5),xticks=(345,347,349,351,353,355),
              yticks=(-5,-3,-1,1,3,5),aspect="equal",xlabel="RA [deg]",ylabel="Dec [deg]",title=title)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=colors),ax=a,shrink=.65)
    fig.suptitle("Continuous C field trial · gray = missing acquisition or metric · labels = QSO counts",fontsize=12)
    fig.savefig(output/"v12_continuous_coverage_summary.png",dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary",type=Path)
    p.add_argument("figures",type=Path)
    a=p.parse_args()
    a.figures.mkdir(parents=True,exist_ok=True)
    render(json.loads(a.summary.read_text()),a.figures)


if __name__=="__main__":
    main()
