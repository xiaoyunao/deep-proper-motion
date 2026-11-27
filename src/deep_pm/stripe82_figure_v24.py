"""Render the common-star diagnostic from one immutable aggregate."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

METHODS=(("ours","SDSS + NSC","#23669d"),("xiao","Xiao et al. (2025)","#bf611d"),("nsc","NSC DR2","#168478"))


def draw(summary,prefix):
    t=json.loads(summary.read_text())
    data=t["density"]
    edges=data["edges_masyr"];extent=(edges[0],edges[-1])*2
    magnitude=t["magnitude"]
    bins=[(key,(float(key.split("-")[0])+float(key.split("-")[1]))/2)
          for key in magnitude if magnitude[key]["common"]>=20]
    if not bins:raise ValueError("Insufficient common HLC stars for the density and magnitude figure")
    with plt.rc_context({"pdf.fonttype":42,"font.size":10,"axes.spines.top":False,"axes.spines.right":False}):
        fig,axes=plt.subplots(2,3,figsize=(13.4,7.7),constrained_layout=True)
        maxn=max(np.max(data["counts"]["ours_"+axis]) for axis in ("e","n"))
        for k,(axis,tex) in enumerate((("e",r"$\mu_{\alpha *}$"),("n",r"$\mu_{\delta}$"))):
            ax=axes[0,k]
            counts=np.array(data["counts"]["ours_"+axis]).T
            image=ax.imshow(np.ma.masked_equal(counts,0),origin="lower",extent=extent,
                            cmap="magma_r",norm=LogNorm(vmin=1,vmax=max(2,maxn)),interpolation="nearest")
            ax.plot(edges[::len(edges)-1],edges[::len(edges)-1],color=".45",lw=.8,ls="--")
            ax.set(xlabel="Stripe 82 "+tex+" (mas yr⁻¹)",ylabel="SDSS + NSC "+tex+" (mas yr⁻¹)")
        fig.colorbar(image,ax=list(axes[0,:2]),label="Common stars per bin",shrink=.78)
        x=[v for key,v in bins]
        for method,label,color in METHODS:
            stats=[magnitude[key]["delta_to_hlc"][method] for key,v in bins]
            axes[0,2].plot(x,[r["radial_p68"] for r in stats],".-",label=label,c=color)
            for k in range(2):
                axes[1,k].plot(x,[r["median"][k] for r in stats],".-",label=label,c=color)
        axes[0,2].legend(frameon=False,fontsize=9)
        axes[0,2].set_ylabel("Radial difference P68 (mas yr⁻¹)")
        for k,tex in enumerate((r"$\mu_{\alpha *}$",r"$\mu_{\delta}$")):
            axes[1,k].axhline(0,c=".6",lw=.7,ls="--")
            axes[1,k].set_ylabel("Median Δ"+tex+" (mas yr⁻¹)")
        ax=axes[1,2]
        for field,label,col in (("parent","HLC parent",".50"),("common","Four-way common","#23669d")):
            count_bins=[(key,np.mean([float(v) for v in key.split("-")])) for key in magnitude if magnitude[key][field]>0]
            ax.plot([v for key,v in count_bins],[magnitude[key][field] for key,v in count_bins],".-",label=label,c=col)
        ax.set_yscale("log");ax.set_ylabel("Stars in magnitude bin");ax.legend(frameon=False,fontsize=9)
        for ax in (axes[0,2],*axes[1]):
            ax.set_xlabel("Stripe 82 clipped PSF r (mag)")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
            ax.grid(axis="y",lw=.5,alpha=.25)
        fig.suptitle(f"Proper motions of {data['sample']:,} common Stripe 82 stars",fontsize=15)
        prefix.parent.mkdir(exist_ok=True,parents=True)
        for extension in (".pdf",".png"):
            fig.savefig(str(prefix)+extension,dpi=180,bbox_inches="tight")
        plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("summary",type=Path);p.add_argument("prefix",type=Path)
    a=p.parse_args();draw(a.summary,a.prefix)
