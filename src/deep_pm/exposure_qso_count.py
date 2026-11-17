"""Reconstruct geometric overlap with the DR16Q parent and Xiao2025 release."""
import argparse
from pathlib import Path
import os

from astropy.io import fits
import numpy as np
from scipy.spatial import cKDTree

from .exposure_catalogues import coordinates
from .io import save_json
from .provenance import run_stage

RADII=(.5,1.,2.)
FLAGS=(b"DESI",b"Gaia")


def count(work,dr16,legacy,workers):
    rank=int(os.environ["SLURM_PROCID"])
    tasks=int(os.environ["SLURM_NTASKS"])
    with fits.open(dr16,memmap=True) as h:
        cat=h[1].data
        ra,dec=cat["RA"],cat["DEC"]
        if len(cat)!=750414 or not (np.isfinite(ra).all() and np.isfinite(dec).all()):
            raise ValueError("DR16Q rows failed parent sample checks")
        tree=cKDTree(coordinates(ra,dec))
    result=np.zeros((len(RADII),3,len(ra)),dtype=np.int32)
    bounds=2*np.sin(np.deg2rad(np.array(RADII)/3600)/2)
    with fits.open(legacy,memmap=True,character_as_bytes=True) as h:
        t=h[1].data
        lo=len(t)*rank//tasks
        hi=len(t)*(rank+1)//tasks
        for start in range(lo,hi,1_000_000):
            part=slice(start,min(start+1_000_000,hi))
            a,d=t["RA"][part],t["DEC"][part]
            good=np.isfinite(a)&np.isfinite(d)
            if not good.any():
                continue
            edges=tree.query_ball_point(coordinates(a[good],d[good]),bounds[-1],
                                       return_sorted=False,workers=workers)
            lengths=np.fromiter(map(len,edges),int,count=len(edges))
            rows=np.repeat(np.arange(len(edges)),lengths)
            if not len(rows):
                continue
            cols=np.fromiter((i for e in edges for i in e),int,count=len(rows))
            xyz=coordinates(a[good],d[good])
            distance=np.linalg.norm(xyz[rows]-tree.data[cols],axis=1)
            origin=np.char.strip(t["PM_SOURCE"][part][good])[rows]
            for r,bound in enumerate(bounds):
                enough=distance<=bound
                for flag,mask in enumerate((enough,*(enough&(origin==k) for k in FLAGS))):
                    result[r,flag]+=np.bincount(cols[mask],minlength=len(ra)).astype(result.dtype)
            if (start-lo)%20_000_000==0:
                print(f"rank {rank} finished {part.stop-lo:,}/{hi-lo:,}",flush=True)
    path=work/f"counts_{rank}.npy"
    partial=path.with_suffix(".part")
    with partial.open("wb") as target:
        np.save(target,result)
    partial.replace(path)


def merge(work,tasks):
    tables=[np.load(work/f"counts_{i}.npy",allow_pickle=False) for i in range(tasks)]
    total=np.sum(tables,axis=0)
    results={}
    for radius,(all_rows,ground,gaia) in zip(RADII,total):
        results[str(radius)]={ "with_any_match":int((all_rows>0).sum()),
            "with_one_match":int((all_rows==1).sum()),
            "one_ground":int(((all_rows==1)&(ground==1)).sum()),
            "one_gaia":int(((all_rows==1)&(gaia==1)).sum()),
            "ambiguous":int((all_rows>1).sum())}
    save_json(work/"qso_count_summary.json",dict(parent_rows=int(total.shape[-1]),
        by_radius_arcsec=results,paper_sample=734012,input_shards=tasks))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["count","merge"])
    p.add_argument("root",type=Path)
    p.add_argument("config",type=Path)
    p.add_argument("--dr16",type=Path)
    p.add_argument("--legacy",type=Path)
    p.add_argument("--tasks",type=int,default=4)
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run on a Slurm node")
    a.root.mkdir(parents=True,exist_ok=True)
    if a.mode=="count":
        rank=int(os.environ["SLURM_PROCID"])
        # Receipt directory per data shard avoids four processes sharing a lock.
        target=a.root/f"shard_{rank}"
        target.mkdir(exist_ok=True)
        with run_stage(target,a.config,f"shard_{rank}","exposure-qso-count-shard-v09",
                       outputs=[f"counts_{rank}.npy"]) as go:
            if go:
                count(target,a.dr16,a.legacy,a.workers)
        link=a.root/f"counts_{rank}.npy"
        if not link.exists():
            link.symlink_to(target.resolve()/link.name)
    else:
        files=[f"counts_{i}.npy" for i in range(a.tasks)]
        if not all((a.root/name).is_file() for name in files):
            raise ValueError("Reconstruction requires every shard")
        with run_stage(a.root,a.config,"legacy_qso","exposure-qso-count-v09",
                       inputs=files,outputs=["qso_count_summary.json"]) as go:
            if go:
                merge(a.root,a.tasks)


if __name__=="__main__":
    main()
