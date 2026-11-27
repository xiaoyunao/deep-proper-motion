"""Native SDSS great-circle and NSC spherical polygon quadrature for the full sky."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
from numba import njit

from .footprints import sdss_window_mask
from .io import digest, save_json

RAD = np.pi/180


def unit(a, d):
    a, d = a*RAD, d*RAD
    return np.column_stack((np.cos(a)*np.cos(d), np.sin(a)*np.cos(d), np.sin(d)))


def sdss_geometry(table):
    node = table["NODE"].astype(float)*RAD
    inc = table["INCL"].astype(float)*RAD
    a = np.column_stack((np.cos(node),np.sin(node),np.zeros(len(node))))
    b = np.column_stack((-np.sin(node)*np.cos(inc),np.cos(node)*np.cos(inc),np.sin(inc)))
    c = np.cross(a,b)
    mu = (table["MU_START"].astype(float)*RAD-node)
    width = ((table["MU_END"].astype(float)-table["MU_START"])%360)*RAD
    end = mu+width
    bottom = np.minimum(table["NU_START"],table["NU_END"]).astype(float)*RAD
    top = np.maximum(table["NU_START"],table["NU_END"]).astype(float)*RAD
    def combine(x,u,y,v):
        return x[:,None]*u+y[:,None]*v
    mid = (mu+end)/2
    lat = (top+bottom)/2
    center = combine(np.cos(lat),combine(np.cos(mid),a,np.sin(mid),b),np.sin(lat),c)
    normals=np.stack((combine(-np.sin(mu),a,np.cos(mu),b),
                      combine(np.sin(end),a,-np.cos(end),b),c,-c),axis=1)
    cuts=np.column_stack((np.zeros(len(mu)),np.zeros(len(mu)),np.sin(bottom),-np.sin(top)))
    radius=(width+top-bottom)/2
    t=table["TAI"].astype(float)
    if t.ndim==2:
        t=t[:,2]
    ok=(radius>0)&(radius<np.pi/2)&(t>=4e9)&(t<6e9)
    # TAI seconds since MJD 0; TT = TAI + 32.184 s.
    return center,normals,cuts,radius,(t+32.184)/86400,ok


def chip_geometry(df):
    corners=np.stack([unit(df[f"vertex_ra{i}"].to_numpy(),df[f"vertex_dec{i}"].to_numpy())
                      for i in range(1,5)],axis=1)
    ra=df[[f"vertex_ra{i}" for i in range(1,5)]].to_numpy()
    dec=df[[f"vertex_dec{i}" for i in range(1,5)]].to_numpy()
    ok=np.isfinite(corners).all(axis=(1,2))&(abs(ra)<=720).all(axis=1)&(abs(dec)<=90).all(axis=1)
    center=corners.sum(axis=1)
    center/=np.maximum(np.linalg.norm(center,axis=1)[:,None],1.e-30)
    normals=np.cross(corners,np.roll(corners,-1,axis=1))
    scale=np.linalg.norm(normals,axis=2)
    normals/=np.maximum(scale[:,:,None],1.e-30)
    facing=np.einsum("nij,nj->ni",normals,center)
    orientation=np.sign(facing[:,0])
    normals*=orientation[:,None,None]
    # Every vertex must be inside the four directed great-circle edges.
    convex=np.einsum("nij,nkj->nik",normals,corners).min(axis=(1,2))>=-1.e-10
    radius=np.arccos(np.clip(np.einsum("nij,nj->ni",corners,center).min(axis=1),-1,1))
    t=df.mjd.to_numpy(float)
    ok &= (scale.min(axis=1)>1.e-12)&convex&(radius>0)&(radius<np.pi/2)&(t>=50000)&(t<=62000)
    # NSC MJD is UTC. Applicable 2012-2019 leap-second steps.
    delta=32.184+np.where(t>=57754,37,np.where(t>=57204,36,np.where(t>=56109,35,34)))
    return center,normals,np.zeros((len(df),4)),radius,t+delta/86400,ok


@njit(cache=True)
def paint(first,last,possible_first,possible_last,center,edges,cuts,radii,times,ok):
    ny,nx=first.shape
    bins=nx//360
    lon=(np.arange(nx)+.5)*RAD/bins
    lat=((np.arange(ny)+.5)/bins-90)*RAD
    cl=np.cos(lon); sl=np.sin(lon); cd=np.cos(lat); sd=np.sin(lat)
    for row in range(len(times)):
        if not ok[row]:
            continue
        v=center[row]; ep=times[row]; r=radii[row]+1.e-10
        a=(np.arctan2(v[1],v[0])/RAD)%360
        d=np.arcsin(min(1.,max(-1.,v[2])))
        dx=180. if abs(d)+r>=np.pi/2 else np.arcsin(min(1.,np.sin(r)/np.cos(d)))/RAD
        low=max(0,int(np.floor((d/RAD-r/RAD+90)*bins)))
        high=min(ny-1,int(np.floor((d/RAD+r/RAD+90)*bins)))
        left=int(np.floor((a-dx)*bins)); right=int(np.floor((a+dx)*bins))
        # Conservative 1-degree scheduling extent retains sub-grid slivers.
        for iy in range(low//bins,high//bins+1):
            for ix in range(left//bins,right//bins+1):
                cell=ix%360
                possible_first[iy,cell]=min(possible_first[iy,cell],ep)
                possible_last[iy,cell]=max(possible_last[iy,cell],ep)
        for i in range(low,high+1):
            for jj in range(left,right+1):
                j=jj%nx; x=cd[i]*cl[j]; y=cd[i]*sl[j]; z=sd[i]
                inside=True
                for k in range(4):
                    if edges[row,k,0]*x+edges[row,k,1]*y+edges[row,k,2]*z<cuts[row,k]-1.e-12:
                        inside=False
                        break
                if inside:
                    first[i,j]=min(first[i,j],ep)
                    last[i,j]=max(last[i,j],ep)


def raster(job):
    kind, names, out, index, bins, binding = job
    directory=Path(out)/f"{kind}-{bins}-{index:02d}"
    directory.mkdir(parents=True,exist_ok=True)
    receipt=directory/"receipt.json"
    key=dict(inputs=binding,files=names,bins=bins,implementation=digest(Path(__file__)))
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old["ticket"]!=key:
            raise ValueError("Changed raster inputs")
        for name,sha in old["files"].items():
            if digest(directory/name)!=sha:
                raise ValueError("Changed raster")
        return str(directory)
    shape=(180*bins,360*bins)
    first=np.full(shape,np.inf,np.float64); last=np.full(shape,-np.inf,np.float64)
    low=np.full((180,360),np.inf);high=np.full((180,360),-np.inf)
    rows=0; invalid=0
    def add(args):
        nonlocal rows,invalid
        ok=args[-1];rows+=len(ok);invalid+=int((~ok).sum())
        paint(first,last,low,high,*args)
    if kind=="early":
        with fits.open(names[0],memmap=True) as hdus:
            t=hdus[1].data
            for start in range(0,len(t),20000):
                batch=t[start:start+20000]
                chosen=sdss_window_mask(batch,301)
                add(sdss_geometry(batch[chosen]))
    else:
        for path in names:
            add(chip_geometry(pd.read_parquet(path)))
    # Only occupied sample centres on disk. Area is integrated from boundaries.
    ids=np.flatnonzero(np.isfinite(first))
    np.savez_compressed(directory/"samples.npz",ids=ids.astype(np.int32),first=first.ravel()[ids],
                        last=last.ravel()[ids],low=low,high=high)
    save_json(receipt,dict(ticket=key,rows=rows,invalid=invalid,files={"samples.npz":digest(directory/"samples.npz")}))
    return str(directory)


def run(kind, base, output, workers, bins):
    work=base/"v2.4"
    if kind=="early":
        f=base/"cache/window_flist.fits"
        jobs=[(kind,[str(f)],str(output),0,bins,digest(f))]
    else:
        root=work/"chips"; manifest=root/"private_chip_manifest.json"; m=json.loads(manifest.read_text())
        paths=[]
        for part in m["parts"]:
            name="_".join(f"{x:g}" for x in part["cell"])+".parquet"
            paths.append((part["rows"],str(root/name),part["sha256"]))
        if m["noncanonical_centres"]:
            paths.append((m["noncanonical_centres"],str(root/"private_noncanonical_centres.parquet"),m["noncanonical_sha256"]))
        buckets=[[] for _ in range(workers)];sizes=np.zeros(workers,dtype=int)
        for rows,path,sha in sorted(paths,reverse=True):
            if digest(Path(path))!=sha:
                raise ValueError("CCD inventory changed")
            i=int(sizes.argmin());buckets[i].append(path);sizes[i]+=rows
        jobs=[(kind,files,str(output),i,bins,digest(manifest)) for i,files in enumerate(buckets)]
    with ProcessPoolExecutor(max_workers=min(len(jobs),workers)) as pool:
        dirs=list(pool.map(raster,jobs))
    save_json(output/f"private_{kind}_{bins}.json",dict(directories=dirs,bins=bins))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind",choices=["early","late"])
    p.add_argument("base",type=Path);p.add_argument("output",type=Path)
    p.add_argument("--bins",type=int,choices=[10,20,40],default=20)
    p.add_argument("--workers",type=int,default=1)
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):
        p.error("Use assigned Slurm CPU resources")
    run(**vars(a))
