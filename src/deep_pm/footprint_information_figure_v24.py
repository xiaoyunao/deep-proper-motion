"""F6. Observed source counts, conditional precision and leverage on identical bins."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
import numpy as np


def draw(source,output):
    summary=json.loads(source.read_text())
    rows=summary["figure_bins"]
    magnitudes=np.arange(16,24.01,.5)
    spans=np.arange(0,32.01,2.)
    layers=np.full((2,3,len(spans)-1,len(magnitudes)-1),np.nan)
    for row in rows:
        i,j=int(not row["single_run"]),int((row["r_low"]-16)*2)
        k=int(row["year_low"]/2)
        layers[i,0,k,j]=row["n"]
        if row["n"]>=20:
            layers[i,1,k,j]=row["vector_standard_error"]
            layers[i,2,k,j]=row["leverage"]
    limits=[LogNorm(1,max(2,np.nanmax(layers[:,0]))),
            Normalize(0,max(1,np.nanmax(layers[:,1]))),Normalize(0,1)]
    titles=["Accepted sources","Formal motion uncertainty","Largest group leverage"]
    labels=["Sources per bin","Median √(σ²ₑ + σ²ₙ) (mas yr⁻¹)","Median maximum axis fraction"]
    fig,axes=plt.subplots(2,3,figsize=(11,6.4),constrained_layout=True,sharex=True,sharey=True)
    for column,(title,norm,cmap) in enumerate(zip(titles,limits,["viridis","magma","cividis"])):
        for row in range(2):
            ax=axes[row,column]
            mesh=ax.pcolormesh(magnitudes,spans,np.ma.masked_invalid(layers[row,column]),cmap=cmap,
                               norm=norm,shading="flat",rasterized=True)
            ax.set_facecolor("#eceeea")
            if row==0:ax.set_title(title,fontsize=11)
            else:ax.set_xlabel("Observed r magnitude")
            if column==0:
                ax.set_ylabel(("One SDSS run","Multiple SDSS runs")[row]+"\nJoint time baseline (yr)")
            ax.set_yticks([0,5,10,15,20,25,30])
            ax.set_yticklabels(["0","5","10","15","20","25","≥30"])
        fig.colorbar(mesh,ax=axes[:,column],orientation="horizontal",pad=.04,shrink=.9,label=labels[column])
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in (".pdf",".png"):
        fig.savefig(str(output)+suffix,bbox_inches="tight",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("summary",type=Path);p.add_argument("prefix",type=Path)
    a=p.parse_args();draw(a.summary,a.prefix)
