"""F1. Native survey intersection and the fraction with long temporal leverage."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(path, output, source_summary=None, reference_area=None):
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
    if (source_summary is None) != (reference_area is None):
        raise ValueError("Final density requires both source counts and supported area")
    if source_summary is not None:
        from matplotlib.colors import LogNorm
        from matplotlib.ticker import LogFormatterMathtext
        src = json.loads(source_summary.read_text())
        native = json.loads(reference_area.read_text())
        if not src.get("reference_admission_sha256") or src["reference_admission_sha256"] != native["reference_admission_sha256"]:
            raise ValueError("Source catalogue and reference support have different physical graph versions")
        if abs(native["resolution_deg"]-.025)>1e-10:
            raise ValueError("Final map requires the registered fine geometry integration")
        coverage = {(int(a),int(d)):value for a,d,_,value in native["cells"]}
        counts = {(int(a),int(d)):n for a,d,n in src["sky_source_counts"]}
        if sum(counts.values()) != src["catalogue_count"]:
            raise ValueError("Mapped source counts do not close")
        native_area = np.array([coverage.get((a,d),0.) for a,d in zip(ra,dec)])
        output_n = np.array([counts.get((a,d),0) for a,d in zip(ra,dec)])
        panels.append((output_n,native_area,"Blind catalogue surface density","magma","Sources per supported deg²"))
    fig = plt.figure(figsize=(8.2,3.0*len(panels)),constrained_layout=True)
    grid = fig.add_gridspec(len(panels),2,width_ratios=(1,.035))
    for i,(numerator,denominator,title,cmap,label) in enumerate(panels):
        keep = denominator>0
        ratio = numerator[keep]/denominator[keep]
        density = i == 3
        if (not density and np.any(ratio > 1+1.e-8)) or np.any(ratio < 0):raise ValueError("Invalid opportunity fraction")
        screen = np.full((180,360),np.nan)
        screen[row[keep],col[keep]] = ratio
        ax = fig.add_subplot(grid[i,0],projection="aitoff")
        style = {"norm": LogNorm(vmin=10,vmax=100000)} if density else {"vmin":0,"vmax":1}
        mesh = ax.pcolormesh(x,y,screen,cmap=cmap,shading="flat",rasterized=True,**style)
        ax.set_facecolor("#eeeeeb")
        ticks = np.arange(-150,151,30)
        ax.set_xticks(np.deg2rad(ticks))
        ax.set_xticklabels([f"{v}°" for v in (-ticks)%360],fontsize=7)
        ax.tick_params(axis="y",labelsize=7)
        ax.grid(lw=.4,color="0.45",alpha=.45)
        subtitle = f"{numerator[keep].sum():,} sources in mapped cells" if density else f"{numerator.sum():,.1f} deg²"
        ax.set_title(f"{title}    {subtitle}",fontsize=11,pad=5)
        options = {"format":LogFormatterMathtext(),"extend":"both"} if density else {"ticks":[0,.25,.5,.75,1]}
        bar = fig.colorbar(mesh,cax=fig.add_subplot(grid[i,1]),**options)
        bar.set_label(label,fontsize=9)
    output.parent.mkdir(parents=True,exist_ok=True)
    for ext in (".png",".pdf"):
        fig.savefig(str(output)+ext,dpi=175,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary",type=Path)
    p.add_argument("prefix",type=Path)
    p.add_argument("--source-summary",type=Path)
    p.add_argument("--reference-area",type=Path)
    a=p.parse_args()
    draw(a.summary,a.prefix,a.source_summary,a.reference_area)
