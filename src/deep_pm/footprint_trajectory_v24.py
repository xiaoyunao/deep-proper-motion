"""Render preregistered real member histories directly from a closed catalogue."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .astrometry import project,years
from .continuous_reference import read_offsets,verify_directory
from .coordinate_metric import forward_jacobian,projected_covariance,native_covariance
from .io import digest,save_json
from .stream_json import array_items

BANDS={"u":"#5c4b99","g":"#37936c","r":"#a7353e","i":"#dc801e","z":"#77522d","y":"#ae4e87","vr":"#337998"}


def run(root,output):
    import duckdb
    database=root/"catalogue/catalogue.duckdb"
    closure=root/"catalogue/v22_catalogue_ownership_summary.json"
    if not closure.exists():raise ValueError("Catalogue must be closed before representative selection")
    plan=json.loads((root/"registration/private_full_coordinate_plan.json").read_text())
    sources={row["core"]:Path(row["source"]) for row in plan["cores"]}
    db=duckdb.connect(str(database),read_only=True)
    selected=[]
    for low,condition in ((21,"sdss_runs=1"),(22,"sdss_runs=1"),(22,"sdss_runs>=2")):
        item=db.execute(f"""SELECT core,slot,sdss_runs,member_sha256 FROM accepted
               WHERE magnitude>=? AND magnitude<? AND {condition}
               ORDER BY md5(member_sha256 || 'v24-representative') LIMIT 1""",[low,low+1]).fetchone()
        if item:selected.append((low,*item))
    if not selected:raise ValueError("No supported trajectory stratum")
    fig,axes=plt.subplots(2,len(selected),figsize=(4.1*len(selected),6.4),constrained_layout=True,squeeze=False)
    private=[]
    for col,(low,core,slot,runs,key) in enumerate(selected):
        source=sources[core]
        folder=root/"sources"/core
        path=folder/"private_standard_motion.json.gz"
        proof=json.loads((folder/"v22_source_core_summary.json").read_text())
        if digest(path)!=proof["private_output_sha256"][path.name]:raise ValueError("Source version mismatch")
        row=next(r for j,r in enumerate(array_items(path)) if j==slot)
        verify_directory(source,("detections.csv",))
        ids=row["detection_ids"]
        keep=["detection_id","observation_time","sigma_east_mas","sigma_north_mas","ra_deg","dec_deg","survey","band"]
        frame=pd.concat([part[part.detection_id.isin(ids)] for part in pd.read_csv(source/"detections.csv",usecols=keep,chunksize=250000)])
        if frame.detection_id.duplicated().any() or len(frame)!=len(ids):raise ValueError("Member identity did not reproduce")
        frame=frame.set_index("detection_id").loc[ids]
        graph=root/"reference"/f"component-{int(row['reference_component']):04d}"/"private_c0_unfiltered.npz"
        if digest(graph)!=row["reference_model_sha256"]:raise ValueError("Reference graph mismatch")
        model=read_offsets(graph);center=row["projection_center_deg"];a,d=frame.ra_deg,frame.dec_deg
        corrected=project(a,d,center)-np.array([model[g][0] for g in row["physical_groups"]])
        state=np.array(row["tangent_state"]).reshape(2,2)
        ra,dec=row["position_deg"]
        inverse=np.linalg.inv(forward_jacobian(np.array([ra]),np.array([dec]),center)[0])
        p=(corrected-state[:,0])@inverse.T
        sigma=np.hypot(frame[["sigma_east_mas","sigma_north_mas"]],50.)
        cov=projected_covariance(a,d,center,native_covariance(np.asarray(sigma)))
        err=np.sqrt(np.diagonal(inverse@cov@inverse.T,axis1=1,axis2=2))
        t=years(frame.observation_time.to_numpy());line=np.array([t.min(),t.max()])
        slope=inverse@state[:,1]
        for k in (0,1):
            ax=axes[k,col];ax.plot(line,(line-row["reference_epoch"])*slope[k],color=".22",lw=1.4,zorder=1)
            for (survey,band),group in frame.groupby(["survey","band"]):
                ix=frame.index.get_indexer(group.index)
                color="#1a365b" if survey=="SDSS" else BANDS.get(band.lower(),".4")
                ax.errorbar(t[ix],p[ix,k],yerr=err[ix,k],fmt="s" if survey=="SDSS" else "o",color=color,
                            ecolor=color,alpha=.7,elinewidth=.65,ms=4,label=f"{survey} {band}")
            ax.grid(axis="y",alpha=.22)
            if col==0:ax.set_ylabel(("East","North")[k]+" offset at J2015.5 (mas)")
            if k:ax.set_xlabel("Observation epoch (Julian year, TT)")
        axes[0,col].set_title(f"{low} ≤ r < {low+1}   ·   {runs} SDSS run"+("s" if runs>1 else ""))
        axes[1,col].legend(frameon=False,fontsize=7,ncol=2)
        private.append(dict(core=core,slot=slot,member_sha256=key,reference_sha256=digest(graph)))
    output.mkdir(exist_ok=True,parents=True)
    for ext in ("pdf","png"):fig.savefig(output/f"F2_trajectories.{ext}",dpi=180)
    plt.close(fig)
    save_json(root/"catalogue/private_f2_examples.json",dict(closure=digest(closure),selected=private))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use a Slurm CPU node")
    run(a.root,a.output)
