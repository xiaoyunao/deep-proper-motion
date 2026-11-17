"""Inspect rectangular SDSS and NSC footprints before any motion fitting."""
import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path

from astropy.io import fits
from matplotlib.path import Path as Polygon
import healpy as hp
import numpy as np
from scipy.spatial import cKDTree

from . import access
from .__main__ import tomllib
from .astrometry import project
from .exposure_catalogues import inside
from .footprints import sdss_visits
from .io import save_json
from .provenance import run_stage
from .regional_preflight import CHIP_VERTICES, span_update


def subboxes(box):
    for i in range(5):
        for j in range(5):
            a,d=box["ra"][0]+2*i,box["dec"][0]+2*j
            yield dict(ra=[a,a+2],dec=[d,d+2])


def where_box(box, halo=0):
    ra,dec=box["ra"],box["dec"]
    return (f"ra >= {ra[0]-halo:.6f} AND ra < {ra[1]+halo:.6f} "
            f"AND dec >= {dec[0]-halo:.6f} AND dec < {dec[1]+halo:.6f}")


def sample(rows, n, seed):
    rng=np.random.default_rng(seed)
    return np.concatenate([np.column_stack((rng.uniform(*b["ra"],n),
        np.rad2deg(np.arcsin(rng.uniform(*np.sin(np.deg2rad(b["dec"])),n))))) for b in rows])


def qtile(values):
    a=np.asarray(values)
    return np.quantile(a,[.16,.5,.84]).tolist() if len(a) else None


def preflight(work, box, settings, window):
    tiles=list(subboxes(box))
    count=settings["positions_per_tile"]
    sky=sample(tiles,count,settings["seed"])
    center=(np.mean(box["ra"]),np.mean(box["dec"]))
    xy=project(*sky.T,center)
    tree=cKDTree(xy)
    nsamp=len(sky)
    early=np.zeros(nsamp,int)
    first=np.full(nsamp,np.inf)
    last=np.full(nsamp,-np.inf)
    visits=[]
    for scan in sdss_visits(window,center,7.3,rerun=301,vectorized=True):
        mask=scan.coverage(xy,sky)
        if mask.any():
            early[mask]+=1
            span_update(mask,scan.year,first,last)
            visits.append(scan.year)
    with fits.open(window,memmap=True) as h:
        t=h[1].data
        inds=np.flatnonzero(inside(t["RA"],t["DEC"],box,.18))
        keys=set(zip(t["RUN"][inds],t["CAMCOL"][inds],t["FIELD"][inds]))
    early_span=np.maximum(0,last-first)
    first.fill(np.inf)
    last.fill(-np.inf)
    late=np.zeros(nsamp,int)
    cache=access.Cache(work/"cache")
    columns="exposure,ccdnum,instrument,filter,mjd,"+",".join(CHIP_VERTICES)
    chips=access.checked_spatial(cache,"nsc",columns,"nsc_dr2.chip",where_box(box,.25),
                                 box["dec"][0]-.25001,box["dec"][1]+.25001)
    vertices=chips[CHIP_VERTICES].to_numpy().reshape(-1,2,4)
    ok=np.isfinite(vertices).all(axis=(1,2)) & chips.mjd.between(50000,62000).to_numpy()
    ids=np.flatnonzero(ok)
    points=project(vertices[ids,0,:].ravel(),vertices[ids,1,:].ravel(),center).reshape(-1,4,2)
    masks=defaultdict(set)
    seen=Counter()
    for i,corners in zip(ids,points):
        mid=corners.mean(axis=0)
        close=np.asarray(tree.query_ball_point(mid,np.linalg.norm(corners-mid,axis=1).max()),dtype=int)
        valid=close[Polygon(corners).contains_points(xy[close])]
        if len(valid):
            row=chips.iloc[i]
            seen[f"{row.instrument}:{row['filter']}"]+=1
            if row["filter"]=="r":
                masks[row.exposure].update(valid.tolist())
                span_update(valid,row.mjd/365.25,first,last)
    for samples in masks.values():
        late[list(samples)]+=1
    span=np.maximum(0,last-first)
    joint=(early>0)&(late>0)
    joint_time=joint&(span>=.6)
    stats=[]
    archive=0
    estimate=0
    for i,region in enumerate(tiles):
        where=where_box(region)+" AND ndetr>0 AND (rmag<24 OR nphotr=0 OR rmag IS NULL)"
        rows=cache.query("nsc","SELECT count(*) AS n, sum(ndet) AS detections FROM nsc_dr2.object WHERE "+where,expected=1)
        objects=int(rows.iloc[0,0])
        detections=int(rows.iloc[0,1]) if objects else 0
        archive+=objects
        estimate+=detections
        q=slice(i*count,(i+1)*count)
        stats.append(dict(region=region,joint=float(joint[q].mean()),joint_time=float(joint_time[q].mean()),
                          long_sdss_repeat=float((early_span[q]>=2).mean()),
                          sdss_visits=qtile(early[q]),nsc_r_visits=qtile(late[q]),
                          objects=objects,approximate_nsc_measurements=detections))
    pixels=hp.ang2pix(settings["nside"],*sky.T,lonlat=True,nest=False)
    sky_cells=len(set(pixels))
    area=np.degrees(np.sin(np.deg2rad(box["dec"][1]))-np.sin(np.deg2rad(box["dec"][0])))*(box["ra"][1]-box["ra"][0])
    save_json(work/"geometry_summary.json",dict(box=box,area_sqdeg=area,tiles=stats,
        sampled_positions=nsamp,sampled_ring_pixels=sky_cells,physical_sdss_runs=len(visits),
        sdss_scan_centres_buffered=len(keys),sdss_year_span=float(np.ptp(visits)) if len(visits)>1 else None,
        geometry_sdss_nights=len({int(x*365.25) for x in visits}),sdss_visits=qtile(early),nsc_r_visits=qtile(late),
        nsc_r_exposures=len(masks),ccd_coverage_by_instrument_band=seen,
        joint_fraction=float(joint.mean()),joint_timed_fraction=float(joint_time.mean()),
        long_sdss_repeat_fraction=float((early_span>=2).mean()),objects=archive,
        approximate_nsc_measurements=estimate))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("window",type=Path)
    p.add_argument("config",type=Path)
    p.add_argument("boxes",nargs="+")
    args=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Use a Slurm compute node")
    names=args.boxes
    if "SLURM_PROCID" in os.environ and len(names)>1:
        names=names[int(os.environ["SLURM_PROCID"])::int(os.environ["SLURM_NTASKS"])]
    settings=tomllib.loads(args.config.read_text())
    for name in names:
        work=args.root/name
        work.mkdir(parents=True,exist_ok=True)
        with run_stage(work,args.config,name,"exposure-preflight-v09",outputs=["geometry_summary.json"]) as go:
            if go:
                preflight(work,settings["boxes"][name],settings,args.window)


if __name__=="__main__":
    main()
