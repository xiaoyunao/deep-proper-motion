"""Independent epoch-labelled S82 comparison inputs, cached before catalogue pairing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from . import access
from .io import digest, save_json


def pack(ids,ra,dec,pm_e,pm_n,se,sn,epoch,valid):
    ra,dec,pm_e,pm_n,se,sn,epoch=(np.asarray(v,dtype=float) for v in (ra,dec,pm_e,pm_n,se,sn,epoch))
    pos=(ra>=0)&(ra<=360)&(abs(dec)<90)
    motion=np.isfinite(pm_e)&np.isfinite(pm_n)&(abs(pm_e)<1e5)&(abs(pm_n)<1e5)
    good=pos&motion&(se>0)&(sn>0)&np.isfinite(se)&np.isfinite(sn)&(epoch>=1990)&(epoch<=2040)&valid
    dt=np.where(good,2015.5-epoch,0.)
    de=np.deg2rad(np.where(good,pm_e,0)*dt/3600000)
    dn=np.deg2rad(np.where(good,pm_n,0)*dt/3600000)
    a=np.deg2rad(ra);d=np.deg2rad(dec)
    x=(np.cos(d)-dn*np.sin(d))*np.cos(a)-de*np.sin(a)
    y=(np.cos(d)-dn*np.sin(d))*np.sin(a)+de*np.cos(a)
    z=np.sin(d)+dn*np.cos(d)
    return pd.DataFrame(dict(external=ids,ra=np.rad2deg(np.arctan2(y,x))%360,
                             dec=np.rad2deg(np.arctan2(z,np.hypot(x,y))),valid=good,
                             pm_e=pm_e,pm_n=pm_n,se_e=se,se_n=sn))[pos]


def xiao(path,out):
    receipt=out/"v24_xiao_epoch_input_summary.json"
    fingerprint=[path.stat().st_size,path.stat().st_mtime_ns]
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved.get("implementation")!=digest(Path(__file__)) or saved["input_stat"]!=fingerprint or any(digest(out/n)!=s for n,s in saved["outputs"].items()):
            raise ValueError("Xiao input or comparison cache changed")
        return
    files={};total=0;agreed=0;n_epoch=0
    with fits.open(path,memmap=True,character_as_bytes=True) as f:
        t=f[1].data
        for start in range(0,len(t),1000000):
            block=t[start:start+1000000]
            ra=block["RA"];dec=block["DEC"]
            ix=np.flatnonzero((abs(dec)<=2.2)&((ra<53)|(ra>=307))&(ra>=0)&(ra<360))
            ix=ix[np.char.strip(np.asarray(block["PM_SOURCE"][ix], dtype="U8"))=="DESI"]
            if not len(ix):continue
            epoch=block["MJD_DESI"][ix].astype(float)
            mjd=np.where(epoch>2400000,epoch-2400000.5,epoch)
            days=(mjd>53000)&(mjd<65000)
            epoch[days]=Time(mjd[days],format="mjd",scale="utc").tt.jyear
            dd=(epoch>=2000)&(epoch<=2040)
            stated=block["TIME"][ix]
            sdss=block["MJD_SDSS"][ix]
            agreed+=int(np.sum(dd&(abs(epoch-(2000+(sdss-51544.5)/365.25)-stated)<.02)))
            n_epoch+=int(dd.sum())
            data=pack(ix+start,ra[ix],dec[ix],block["PMRA"][ix],block["PMDEC"][ix],
                      block["PMRA_ERR"][ix],block["PMDEC_ERR"][ix],epoch,dd)
            name=f"private_xiao_{start:09d}.parquet"
            data.to_parquet(out/name,index=False,compression="zstd")
            files[name]=digest(out/name);total+=len(data)
    if fingerprint!=[path.stat().st_size,path.stat().st_mtime_ns]:
        raise ValueError("Xiao catalogue changed during read")
    if total == 0 or n_epoch == 0:
        raise ValueError("No validated Xiao DESI epochs in full Stripe 82 neighbourhood")
    save_json(receipt,dict(implementation=digest(Path(__file__)),input_stat=fingerprint,outputs=files,rows=total,
                          epoch_valid=n_epoch,span_consistent_0p02_years=agreed))


def nsc(out,workers):
    cols="id,ra,dec,mjd,pmra,pmdec,pmraerr,pmdecerr,deltamjd"
    def one(a):
        path=out/f"private_nsc_{a:03d}.parquet"; proof=path.with_suffix(".json")
        if proof.exists():
            saved=json.loads(proof.read_text())
            if digest(path)!=saved["sha256"]:
                raise ValueError("Changed official NSC input")
            return saved
        cache=access.Cache(out/"query-cache")
        where=(f"q3c_radial_query(ra,dec,{a+.5},0,2.3) AND ra>={a} AND ra<{a+1}")
        df=access.checked_spatial(cache,"nsc",cols,"nsc_dr2.object",where,-2.2,2.2)
        if df.id.isna().any() or df.id.duplicated().any():
            raise ValueError("Duplicate official NSC ID")
        epoch=np.full(len(df),np.nan)
        good=df.mjd.between(50000,62000).to_numpy()
        epoch[good]=Time(df.mjd[good].to_numpy(),format="mjd",scale="utc").tt.jyear
        packed=pack(df.id,df.ra,df.dec,df.pmra,df.pmdec,df.pmraerr,df.pmdecerr,
                    epoch,df.deltamjd.to_numpy()>0)
        packed.to_parquet(path,index=False,compression="zstd")
        result=dict(rows=len(df),usable=int(packed.valid.sum()),file=path.name,sha256=digest(path))
        save_json(proof,result)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts=list(pool.map(one,list(range(53))+list(range(307,360))))
    save_json(out/"v24_nsc_epoch_input_summary.json",dict(rows=sum(p["rows"] for p in parts),parts=parts))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["nsc","xiao"]);p.add_argument("output",type=Path)
    p.add_argument("--xiao",type=Path);p.add_argument("--workers",type=int,default=2)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):
        p.error("Use assigned Slurm CPUs")
    if a.mode=="xiao":xiao(a.xiao,a.output)
    else:nsc(a.output,a.workers)
