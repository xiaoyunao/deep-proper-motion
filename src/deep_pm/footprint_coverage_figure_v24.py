"""F1. Native survey intersection and the fraction with long temporal leverage."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(path, output):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    if data["columns"] != ["ra_low","dec_low","intersection_sqdeg","time_sqdeg","time_and_15_year_sqdeg"]:
        raise ValueError("Unexpected sky integration columns")
    cells = np.asarray(data["cells"])
    ra, dec = cells[:, :2].T.astype(int)
    # Reversing RA places zero at the centre and east to the left. Columns
    # describe lower bin edges in the native spherical integration.
    col = (179-ra) % 360
    row = dec+90
    area = np.rad2deg(np.sin(np.deg2rad(dec+1))-np.sin(np.deg2rad(dec)))
    geometric, temporal, long = cells[:, 2:].T
    np.testing.assert_allclose(cells[:, 2:].sum(axis=0),data["total_sqdeg"],rtol=1e-9)
    x = np.deg2rad(np.arange(-180,181))
    y = np.deg2rad(np.arange(-90,91))
    panels = [(geometric,area,"Shared survey footprint","magma_r","Fraction of sky cell"),
              (temporal,geometric,"Time support for proper motion","viridis","Fraction of survey intersection"),
              (long,temporal,"Joint baseline of at least 15 yr","cividis","Fraction of time-supported area")]
    fig = plt.figure(figsize=(8.2,9.2),constrained_layout=True)
    grid = fig.add_gridspec(3,2,width_ratios=(1,.035))
    for i,(numerator,denominator,title,cmap,label) in enumerate(panels):
        keep = denominator>0
        ratio = numerator[keep]/denominator[keep]
        if np.any(ratio > 1+1.e-8) or np.any(ratio < 0):raise ValueError("Invalid opportunity fraction")
        screen = np.full((180,360),np.nan)
        screen[row[keep],col[keep]] = ratio
        ax = fig.add_subplot(grid[i,0],projection="aitoff")
        mesh = ax.pcolormesh(x,y,screen,vmin=0,vmax=1,cmap=cmap,shading="flat",rasterized=True)
        ax.set_facecolor("#eeeeeb")
        ticks = np.arange(-150,151,30)
        ax.set_xticks(np.deg2rad(ticks))
        ax.set_xticklabels([f"{v}°" for v in (-ticks)%360],fontsize=7)
        ax.tick_params(axis="y",labelsize=7)
        ax.grid(lw=.4,color="0.45",alpha=.45)
        ax.set_title(f"{title}    {numerator.sum():,.1f} deg²",fontsize=11,pad=5)
        bar = fig.colorbar(mesh,cax=fig.add_subplot(grid[i,1]),ticks=[0,.25,.5,.75,1])
        bar.set_label(label,fontsize=9)
    output.parent.mkdir(parents=True,exist_ok=True)
    for ext in (".png",".pdf"):
        fig.savefig(str(output)+ext,dpi=175,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary",type=Path)
    p.add_argument("prefix",type=Path)
    a=p.parse_args()
    draw(a.summary,a.prefix)
