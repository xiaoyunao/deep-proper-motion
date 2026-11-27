"""Freeze the published HLC motion/photometry parent, independent of ground PMs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd

from .footprint_assets_v24 import COUNTS
from .io import digest, save_json


def shard(args):
    h, root, output, signature = args
    name = f"HLC.RA_{h:02d}_to_{h+1:02d}.fits.gz"
    path = root/name
    proof = json.loads((root/(name+".json")).read_text())
    tablefile = output/f"private_hlc_{h:02d}.parquet"
    done = tablefile.with_suffix(".json")
    ticket = dict(input=proof["sha256"], protocol=signature, implementation=digest(Path(__file__)))
    if done.is_file():
        old = json.loads(done.read_text())
        if old["ticket"] != ticket or old["sha256"] != digest(tablefile):
            raise ValueError("Changed HLC parent inputs or output")
        return old
    if digest(path) != proof["sha256"]:
        raise ValueError("HLC changed after verification")
    with fits.open(path, memmap=False) as hdus:
        t = hdus[1].data
        def v(key):
            return np.asarray(t[key], dtype=float)
        ra, dec, epoch = (v(k) for k in ("RA_MEAN_CLIP","DEC_MEAN_CLIP","T0_CLIP"))
        east, north, se, sn = [1000*v(k) for k in ("RA_PM_CLIP","DEC_PM_CLIP",
                                                      "RA_PM_ERR_CLIP","DEC_PM_ERR_CLIP")]
        n, dt = v("N_POS_EPOCHS_CLIP"), v("TIME_SPAN")
        phot = v("MEAN_PSFMAG_CLIP")
        r = phot[:,2]
        er = v("MEAN_PSFMAG_ERR_CLIP")[:,2]
        rpoints = v("N_GOOD_EPOCHS_PSF_CLIP")[:,2]
        point = v("MEAN_OBJECT_TYPE") >= 5.5
        photo = point & (r>=16) & (r<24) & (er>0) & (er<=.15) & (rpoints>=5)
        pos = (ra>=0) & (ra<=360) & (abs(dec)<=90)
        motion = pos & (epoch>2400000) & (epoch<2500000) & (se>0) & (sn>0)
        motion &= np.isfinite(east) & np.isfinite(north) & (n>=3) & (abs(east)<1e6) & (abs(north)<1e6)
        years = np.where(motion, (2451545.+15.5*365.25-epoch)/365.25, 0.)
        a = np.where(motion, ra+east*years/(3600000*np.cos(np.deg2rad(dec))), v("RA_MEAN"))
        d = np.where(motion, dec+north*years/3600000, v("DEC_MEAN"))
        has_pos = (a>=-1) & (a<=361) & (abs(d)<=90)
        fit = motion & (se<=20) & (sn<=20) & (dt>=730.5) & (n>=10)
        for axis in ("RA","DEC"):
            chi = v(axis+"_CHISQ_LIN_CLIP")
            fit &= (chi>=0) & (chi<=5*(n-2))
        ok = photo & fit & has_pos
        colour = np.where((phot[:,1]>0)&(phot[:,1]<35)&(phot[:,3]>0)&(phot[:,3]<35),
                          phot[:,1]-phot[:,3], np.nan)
        frame = pd.DataFrame(dict(identity=(np.int64(h)<<32)+np.arange(len(r)),ra=a%360,dec=d,
                                  has_position=has_pos, propagated=motion, stellar_photo=photo,
                                  parent=ok,r=r,colour=colour,pm_e=east,pm_n=north,se_e=se,se_n=sn))
        frame.to_parquet(tablefile,index=False,compression="zstd")
        bins = [16,18,19,20,21,21.5,22,22.5,23,24]
        counts = {}
        for lo,hi in zip(bins,bins[1:]):
            ix=(r>=lo)&(r<hi)
            counts[f"{lo:g}-{hi:g}"] = dict(rows=int(ix.sum()), stellar_photo=int((ix&photo).sum()),
                                            parent=int((ix&ok).sum()))
        result=dict(ticket=ticket,rows=len(t),usable_positions=int(has_pos.sum()),parent=int(ok.sum()),
                    photometry=int(photo.sum()),bins=counts,sha256=digest(tablefile))
    save_json(done,result)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("root","protocol","output"):
        p.add_argument(key,type=Path)
    p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not 1<=a.workers<=int(os.environ["SLURM_CPUS_PER_TASK"]):
        p.error("Use assigned CPUs on Slurm")
    a.output.mkdir(exist_ok=True,parents=True)
    signature=digest(a.protocol)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        parts=list(pool.map(shard,[(h,a.root,a.output,signature) for h in COUNTS]))
    if sum(p["rows"] for p in parts)!=3700548:
        raise ValueError("HLC parent denominator did not close")
    save_json(a.output/"v24_s82_parent_summary.json",dict(rows=sum(p["rows"] for p in parts),
              parent=sum(p["parent"] for p in parts),protocol_sha256=signature,
              bins={b:{k:sum(p["bins"][b][k] for p in parts) for k in vals}
                    for b,vals in parts[0]["bins"].items()}))
