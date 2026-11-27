"""Exact native scan/CCD and time opportunity at moving HLC parent positions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .footprint_raster_v24 import chip_geometry,sdss_geometry,sdss_window_mask
from .io import digest,save_json


def partition(job):
    base,kind,number,files,bindings=job
    root=base/"v2.4";out=root/"s82-opportunity"
    paths=sorted((root/"s82").glob("private_hlc_*.parquet"))
    tag=f"{kind}-{number:02d}"
    ticket=dict(geometry=bindings,hlc=[json.loads(p.with_suffix('.json').read_text())["sha256"] for p in paths],
                implementation=digest(Path(__file__)))
    proof=out/(tag+".json");datafile=out/(tag+".npz")
    if proof.exists():
        r=json.loads(proof.read_text())
        if r["ticket"]!=ticket or r["sha256"]!=digest(datafile):raise ValueError("Changed S82 footprint shard")
        return tag
    pieces=[]
    for p,sha in zip(paths,ticket["hlc"]):
        if digest(p)!=sha:raise ValueError("Changed HLC geometry parent")
        pieces.append(pd.read_parquet(p,columns=["identity","ra","dec","pm_e","pm_n"],filters=[("parent","=",True)]))
    s=pd.concat(pieces,ignore_index=True)
    a,d=np.deg2rad(s.ra.to_numpy()),np.deg2rad(s.dec.to_numpy())
    ca,sa=np.cos(a),np.sin(a);cd,sd=np.cos(d),np.sin(d)
    sky=np.column_stack((ca*cd,sa*cd,sd))
    east=np.column_stack((-sa,ca,np.zeros(len(a))));north=np.column_stack((-ca*sd,-sa*sd,cd))
    velocity=(east*s.pm_e.to_numpy()[:,None]+north*s.pm_n.to_numpy()[:,None])*np.deg2rad(1/3600000.)
    speed=np.linalg.norm(velocity,axis=1).max()
    tree=cKDTree(sky)
    lo=np.full(len(s),np.inf);hi=np.full(len(s),-np.inf)
    count=np.zeros(len(s),dtype=np.int32)
    max_lat=max(abs(d.min()),abs(d.max()))
    def apply(g):
        c,edges,cuts,radius,time,ok=g
        dt=(time-(51544.5+15.5*365.25))/365.25
        # Banker's rounding of catalogue RA and TT/HJD minutes do not determine inclusion.
        pad=radius+speed*abs(dt)+2.e-10
        use=np.flatnonzero(ok&(abs(np.arcsin(np.clip(c[:,2],-1,1)))<=max_lat+pad))
        for j in use:
            ix=np.asarray(tree.query_ball_point(c[j],2*np.sin(min(np.pi,pad[j])/2)),dtype=int)
            if not len(ix):continue
            xy=sky[ix]+dt[j]*velocity[ix]
            xy/=np.linalg.norm(xy,axis=1)[:,None]
            ix=ix[np.all(xy@edges[j].T>=cuts[j]-1.e-12,axis=1)]
            lo[ix]=np.minimum(lo[ix],time[j]);hi[ix]=np.maximum(hi[ix],time[j])
            count[ix]+=1
    if kind=="early":
        with fits.open(files[0],memmap=True) as hdus:
            t=hdus[1].data
            for start in range(0,len(t),20000):
                block=t[start:start+20000]
                apply(sdss_geometry(block[sdss_window_mask(block,301)]))
    else:
        for p in files:apply(chip_geometry(pd.read_parquet(p)))
    np.savez_compressed(datafile,identity=s.identity.to_numpy(),first=lo,last=hi,count=count)
    save_json(proof,dict(ticket=ticket,sha256=digest(datafile)))
    return tag


def run(base,workers):
    root=base/"v2.4";out=root/"s82-opportunity";out.mkdir(exist_ok=True,parents=True)
    records=json.loads((root/"geometry/private_late_40.json").read_text())["directories"]
    late=[]
    for d in records:
        receipt=json.loads((Path(d)/"receipt.json").read_text())
        late.append((receipt["ticket"]["files"],receipt["ticket"]["inputs"]))
    sdss=base/"cache/window_flist.fits"
    jobs=[(base,"early",0,[sdss],digest(sdss))]
    jobs.extend((base,"late",i,f,sig) for i,(f,sig) in enumerate(late))
    with ProcessPoolExecutor(max_workers=workers) as pool:tags=list(pool.map(partition,jobs))
    combined={}
    ident=None
    for tag in tags:
        with np.load(out/(tag+".npz")) as part:
            key=tag.split("-")[0]
            if ident is None:ident=part["identity"]
            elif not np.array_equal(ident,part["identity"]):raise ValueError("HLC source order changed")
            if key not in combined:combined[key]=[part["first"],part["last"],part["count"]]
            else:
                lo,hi,n=combined[key]
                np.minimum(lo,part["first"],out=lo);np.maximum(hi,part["last"],out=hi);n+=part["count"]
    early,late=combined["early"],combined["late"]
    joint=(early[2]>0)&(late[2]>0)
    time=joint&((late[1]-late[0])>=.6*365.25)&((np.maximum(early[1],late[1])-np.minimum(early[0],late[0]))>=5*365.25)
    frame=pd.DataFrame(dict(identity=ident,geometry=joint,time_opportunity=time))
    f=out/"private_hlc_opportunity.parquet";frame.to_parquet(f,index=False)
    save_json(out/"v24_hlc_geometric_opportunity_summary.json",dict(parent=len(ident),geometric=int(joint.sum()),time=int(time.sum()),
              sha256=digest(f),scope="Full HLC parent propagated to each actual SDSS field/NSC CCD epoch, spherical native boundaries. Survey geometry before per-source detection and common-reference support."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("--workers",type=int,default=2);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):p.error("Use assigned Slurm CPUs")
    run(a.base,a.workers)
