"""Integrate native survey and time opportunity within a single supported physical graph."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from numba import njit,types
from numba.typed import Dict

from .astrometry import basis
from .footprint_raster_v24 import chip_geometry,sdss_geometry,sdss_window_mask
from .io import digest,save_json


@njit(cache=True)
def accepted_direction(a,d,frame,cov):
    # Inverse gnomonic Jacobian in native angular units:
    # d(local tangent angle)/d(plane coord) = (u·centre) (local basis·chart basis).
    ca,sa,cd,sd=np.cos(a),np.sin(a),np.cos(d),np.sin(d)
    dot=ca*cd*frame[0,0]+sa*cd*frame[0,1]+sd*frame[0,2]
    if dot<=0:return False
    for axis in range(2):
        x,y,z=(-sa,ca,0.) if axis==0 else (-ca*sd,-sa*sd,cd)
        u=dot*(x*frame[1,0]+y*frame[1,1]+z*frame[1,2])
        v=dot*(x*frame[2,0]+y*frame[2,1]+z*frame[2,2])
        if u*u*cov[0,0]+2*u*v*cov[0,1]+v*v*cov[1,1]>144.:return False
    return True


@njit(cache=True)
def mark(tiles,bins,components,frames,covs,group,centers,edges,cuts,radii,times,valid):
    rad=np.pi/180
    for k in range(len(group)):
        ix=group[k]
        if ix<0 or not valid[k]:continue
        comp=components[ix];center=centers[k];r=radii[k]+1.e-10
        a=(np.arctan2(center[1],center[0])/rad)%360
        d=np.arcsin(min(1.,max(-1.,center[2])))
        dx=180. if abs(d)+r>=np.pi/2 else np.arcsin(min(1.,np.sin(r)/np.cos(d)))/rad
        south=max(0,int(np.floor((d/rad-r/rad+90)*bins)))
        north=min(180*bins-1,int(np.floor((d/rad+r/rad+90)*bins)))
        for i in range(south,north+1):
            delta=((i+.5)/bins-90)*rad;cd,sd=np.cos(delta),np.sin(delta)
            for jj in range(int(np.floor((a-dx)*bins)),int(np.floor((a+dx)*bins))+1):
                j=jj%(360*bins);alpha=(j+.5)/bins*rad
                x,y,z=cd*np.cos(alpha),cd*np.sin(alpha),sd
                inside=True
                for h in range(4):
                    if edges[k,h,0]*x+edges[k,h,1]*y+edges[k,h,2]*z<cuts[k,h]-1.e-12:
                        inside=False;break
                if not inside or not accepted_direction(alpha,delta,frames[comp],covs[ix]):continue
                tile=comp*64800+(i//bins)*360+j//bins
                if tile not in tiles:
                    arr=np.empty((2,bins,bins));arr[0,:,:]=np.inf;arr[1,:,:]=-np.inf
                    tiles[tile]=arr
                values=tiles[tile];u,v=i%bins,j%bins
                values[0,u,v]=min(values[0,u,v],times[k])
                values[1,u,v]=max(values[1,u,v],times[k])


def models(joint):
    evidence=json.loads((joint/"admission/v24_reference_admission_summary.json").read_text())
    frames=np.zeros((max(int(c) for c in evidence["graphs"])+1,3,3))
    groups=[];components=[];variance=[]
    for name,sha in evidence["graphs"].items():
        c=int(name);p=joint/"reference"/f"component-{c:04d}"/"private_c0_unfiltered.npz"
        if digest(p)!=sha:raise ValueError("Changed admitted physical group model")
        with np.load(p) as m:
            keys=m["names"].astype(str);draws=m["draws"];draws-=draws.mean(axis=1,keepdims=True)
            groups.extend(keys);components.extend([c]*len(keys));variance.append(draws.transpose(0,2,1)@draws/47)
            frames[c]=basis(*m["projection_center_deg"])
    idx=pd.Index(groups)
    if not idx.is_unique:raise ValueError("One offset assigned to incompatible physical components")
    return idx,np.asarray(components),frames,np.concatenate(variance)


def produce(base,kind,section,bins):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=base/"v2.4";joint=root/"joint";out=joint/"native-area";out.mkdir(exist_ok=True,parents=True)
    registry=root/"geometry"/f"private_{kind}_{bins}.json"
    directory=Path(json.loads(registry.read_text())["directories"][section])
    meta=directory/"receipt.json";spec=json.loads(meta.read_text())["ticket"]
    ticket=dict(geometry=digest(meta),graph=digest(joint/"admission/v24_reference_admission_summary.json"),
                bins=bins,implementation=digest(Path(__file__)))
    marker=out/f"{kind}-{bins}-{section:03d}.json";file=marker.with_suffix(".parquet")
    if marker.exists():
        old=json.loads(marker.read_text())
        if old["ticket"]!=ticket or digest(file)!=old["sha256"]:raise ValueError("Changed supported opportunity shard")
        return
    index,components,frames,cov=models(joint)
    tiles=Dict.empty(key_type=types.int64,value_type=types.float64[:,:,::1])
    def add(labels,geometry):
        mark(tiles,bins,components,frames,cov,index.get_indexer(labels),*geometry)
    for path in spec["files"]:
        if kind=="late":
            df=pd.read_parquet(path)
            add("NSC:"+df.exposure.astype(str),chip_geometry(df))
        else:
            with fits.open(path,memmap=True) as hdus:
                data=hdus[1].data
                for start in range(0,len(data),20000):
                    f=data[start:start+20000];f=f[sdss_window_mask(f,301)]
                    labels=np.char.add(np.char.add(np.char.add("SDSS:",f["RUN"].astype(str)),":"),f["CAMCOL"].astype(str))
                    add(labels,sdss_geometry(f))
    schema=pa.schema([("id",pa.int64()),("first",pa.float64()),("last",pa.float64())])
    temp=file.with_suffix(".pending.parquet")
    with pq.ParquetWriter(temp,schema,compression="zstd") as writer:
        for tile in sorted(tiles):
            lo,hi=tiles[tile].reshape(2,-1)
            keep=np.flatnonzero(np.isfinite(lo))
            writer.write_table(pa.Table.from_arrays([keep+np.int64(tile)*bins**2,lo[keep],hi[keep]],schema=schema))
    temp.replace(file);save_json(marker,dict(ticket=ticket,sha256=digest(file),tiles=len(tiles)))


def integrate(base,bins):
    import duckdb
    root=base/"v2.4";joint=root/"joint";out=joint/"native-area"
    proofs=[]
    for kind in ("early","late"):
        registry=root/"geometry"/f"private_{kind}_{bins}.json"
        for i,d in enumerate(json.loads(registry.read_text())["directories"]):
            p=out/f"{kind}-{bins}-{i:03d}.json";r=json.loads(p.read_text())
            if r["ticket"]["geometry"]!=digest(Path(d)/"receipt.json") or digest(p.with_suffix(".parquet"))!=r["sha256"]:
                raise ValueError("Missing or changed opportunity geometry binding")
            if r["ticket"]["graph"]!=digest(joint/"admission/v24_reference_admission_summary.json"):
                raise ValueError("Mixed reference graph in opportunity mosaic")
            proofs.append(digest(p))
    db=duckdb.connect();db.execute("SET memory_limit='5GB'")
    db.execute(f"SET threads={min(8,int(os.environ['SLURM_CPUS_PER_TASK']))}")
    db.execute(f"SET temp_directory='{out}/spill'")
    for kind in ("early","late"):
        db.execute(f"""CREATE TEMP TABLE {kind} AS SELECT id,min(first) t0,max(last) t1
             FROM read_parquet('{out}/{kind}-{bins}-*.parquet') GROUP BY id""")
    stride=bins**2;period=64800*stride
    # Join on both the physical component and spherical point. Only after the
    # time gates are applied may overlapping admissible components form a union.
    query=f"""WITH physical AS (SELECT e.id%{period} pid,
              l.t1-l.t0>=.6*365.25 AND greatest(e.t1,l.t1)-least(e.t0,l.t0)>=5*365.25 ok
              FROM early e JOIN late l USING(id)),
              united AS (SELECT pid,bool_or(ok) ok FROM physical GROUP BY pid)
              SELECT pid//{stride} tile,(pid%{stride})//{bins} latrow,
                     count(*) joint,count(*) FILTER(WHERE ok) usable FROM united GROUP BY ALL"""
    df=db.execute(query).fetchdf()
    south=df.tile.to_numpy()//360-90+df.latrow.to_numpy()/bins
    weight=np.rad2deg(np.sin(np.deg2rad(south+1/bins))-np.sin(np.deg2rad(south)))/bins
    df["reference_area"]=df.joint.to_numpy()*weight;df["time_reference_area"]=df.usable.to_numpy()*weight
    bytile=df.groupby("tile")[["reference_area","time_reference_area"]].sum()
    unsigned=json.loads((root/"registration/v24_footprint_inventory_summary.json").read_text())["grids"][str(bins)]["areas_sqdeg"]
    if bytile.reference_area.sum()>unsigned["intersection"]+1e-5 or bytile.time_reference_area.sum()>unsigned["time"]+1e-5:
        raise ValueError("Native calibrated area exceeds uncalibrated geometry")
    save_json(out/f"v24_native_reference_area_{bins}_summary.json",dict(resolution_deg=1/bins,reference_admission_sha256=digest(joint/"admission/v24_reference_admission_summary.json"),
              receipt_set=proofs,reference_overlap_sqdeg=bytile.reference_area.sum(),reference_and_time_sqdeg=bytile.time_reference_area.sum(),
              cells=[[int(tile%360),int(tile//360)-90,*row] for tile,row in zip(bytile.index,bytile.to_numpy())],
              scope="Joint native field and CCD footprint supported at that sky position by the same physical component; "
                    "each native-axis reference SE <=12 mas. Union after component-specific time tests. No count-to-area conversion."))
    db.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("base",type=Path);p.add_argument("kind",choices=("early","late","union"))
    p.add_argument("--section",type=int,default=0);p.add_argument("--bins",type=int,choices=(20,40),default=40);a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID"):p.error("Use a Slurm compute node")
    if a.kind=="union":integrate(a.base,a.bins)
    else:produce(a.base,a.kind,a.section,a.bins)
