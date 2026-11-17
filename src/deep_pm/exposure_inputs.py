"""Acquire a frozen rectangular core and halo while retaining physical detector IDs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import resource
from time import monotonic

import numpy as np
import pandas as pd

from . import access
from .__main__ import standardize, tomllib
from .exposure_catalogues import inside
from .exposure_preflight import where_box
from .io import save_json
from .provenance import run_stage
from .regional_inputs import GAIA_COLUMNS
from .sdss_sas import read_sdss_sas


def tap_rectangle(box, halo=0., alias=""):
    west,east=box["ra"]
    south,north=box["dec"]
    return (f"{alias}ra >= {west-halo:.9f} AND {alias}ra < {east+halo:.9f} "
            f"AND {alias}dec >= {south-halo:.9f} AND {alias}dec < {north+halo:.9f}")


def tap_join(cache, archive, box, halo, service="tap", workers=3, native_cap=30000, native_window=False):
    """Partition on archive positions, then verify every joined measurement count."""
    table="nsc_dr2.meas m JOIN nsc_dr2.object o ON m.objectid=o.id"
    select=",".join("m."+k for k in access.MEAS.split(","))
    quality=" AND o.ndetr>0 AND (o.rmag<24 OR o.nphotr=0 OR o.rmag IS NULL)"
    lo,hi=np.asarray(box["ra"])+[-halo,halo]
    south,north=np.asarray(box["dec"])+[-halo,halo]
    partitions=[]
    target_rows = 80000 if service == "tap" else int(.85*native_cap)
    answer_cap = 100000 if service == "tap" else native_cap

    def split(low,high,data,finish):
        if finish and float(data.ndet.sum())<=target_rows:
            partitions.append((low,high))
            return
        axis=int(np.argmax(high-low))
        mid=round((low[axis]+high[axis])/2,9)
        hi_l=high.copy()
        hi_l[axis]=mid
        lo_r=low.copy()
        lo_r[axis]=mid
        west=data.iloc[:,axis]<mid
        split(low,hi_l,data.loc[west],True)
        split(lo_r,high,data.loc[~west],True)

    def receive(bounds):
        low,high=bounds
        where=tap_rectangle(dict(ra=[low[0],high[0]],dec=[low[1],high[1]]),alias="o.")+quality
        if service == "nsc":
            mid = (low+high)/2
            radius = np.linalg.norm(high-low)/2+.003
            where = (f"q3c_radial_query(o.ra,o.dec,{mid[0]:.10f},{mid[1]:.10f},{radius:.10f}) AND "+where)
        try:
            if service == "nsc" and native_window:
                return cache.checked_native_window(select,table,where,cap=answer_cap)
            return cache.checked(service,select,table,where,cap=answer_cap)
        except access.PartitionTooLarge:
            axis=int(np.argmax(high-low))
            cut=round((low[axis]+high[axis])/2,9)
            rc,lc=low.copy(),high.copy()
            rc[axis]=lc[axis]=cut
            return pd.concat([receive((low,lc)),receive((rc,high))],ignore_index=True)

    split(np.array([lo,south]),np.array([hi,north]),archive[["ra","dec","ndet"]],True)
    print(f"{service} expected measurement partitions: {len(partitions)}",flush=True)
    parts=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,part in enumerate(pool.map(receive,partitions),1):
            parts.append(part)
            print(f"{service} verified measurements partition {i}/{len(partitions)}: {len(part):,}",flush=True)
    result=pd.concat(parts,ignore_index=True)
    if result.measid.duplicated().any():
        raise RuntimeError("Duplicate measurement IDs in disjoint archive bins")
    return result


def acquire(work, spec, settings, catalogues, sas, workers):
    start=monotonic()
    halo=settings["halo_deg"]
    center=tuple(np.mean(spec[k]) for k in ("ra","dec"))
    radius=float(np.hypot(np.ptp(spec["ra"])/2,np.ptp(spec["dec"])/2))
    cache=access.Cache(work/"cache")
    data,nfiles=read_sdss_sas(sas,center,radius+2*halo,workers=workers)
    data=data.loc[inside(data.ra,data.dec,spec,halo)]
    data.to_csv(work/"sdss.csv",index=False)
    print(f"{len(data):,} SDSS detections in {nfiles} inspected files",flush=True)
    use_tap=settings.get("tap_join",False)
    native_join=settings.get("native_join",False)
    native_window=settings.get("native_count_window",False) and native_join and not use_tap
    route="tap" if use_tap else "nsc"
    footprint=tap_rectangle if use_tap else where_box
    base=footprint(spec,halo)
    south,north=spec["dec"]
    lo,hi=south-halo-1e-7,north+halo+1e-7
    query=base+" AND ndetr>0 AND (rmag<24 OR nphotr=0 OR rmag IS NULL)"
    columns="id AS objectid,ra,dec,pmra,pmdec,pmraerr,pmdecerr,deltamjd"+(" ,ndet" if use_tap or native_join else "")
    archive=access.checked_spatial(cache,route,columns,
                                   "nsc_dr2.object",query,lo,hi)
    ids=archive.objectid.astype(str).sort_values()
    if ids.duplicated().any():
        raise ValueError("Duplicate NSC object keys in disjoint archive partitions")
    print(f"{len(ids):,} NSC archive objects",flush=True)
    archive.rename(columns={"objectid":"id"}).to_csv(work/"official_nsc.csv",index=False)
    if use_tap or native_join:
        nsc=tap_join(cache,archive,spec,halo,service=route,workers=settings.get("download_workers",3),
                     native_cap=settings.get("native_row_cap",30000),native_window=native_window)
    else:
        nsc=access.fetch_measurements_for_objects(cache,ids,batch_size=250)
    returned=set(nsc.objectid.astype(str))
    requested=set(ids)
    if returned-requested:
        raise ValueError("NSC returned measurements outside the requested archive IDs")
    missing=sorted(requested-returned)
    for start in range(0,len(missing),250):
        check="objectid IN ("+",".join(map(access.literal,missing[start:start+250]))+")"
        if len(cache.checked(route,"objectid","nsc_dr2.meas",check)):
            raise ValueError("Object IDs absent from batches have published measurement rows")
    print(f"{len(missing):,} archive objects have no matching rows in meas",flush=True)
    nsc.loc[inside(nsc.ra,nsc.dec,spec,halo)].to_csv(work/"nsc_object.csv",index=False)
    chip_pad=halo+.35
    chips=access.checked_spatial(cache,route,"exposure,ccdnum,instrument,airmass",
                                 "nsc_dr2.chip",footprint(spec,chip_pad),
                                 south-chip_pad-1e-7,north+chip_pad+1e-7)
    chips=chips.drop_duplicates()
    if chips.duplicated(["exposure","ccdnum"]).any():
        raise ValueError("Conflicting NSC chip metadata for one physical detector")
    chips.to_csv(work/"chips.csv",index=False)
    gaia_columns=GAIA_COLUMNS+(",visibility_periods_used" if use_tap or native_join else "")
    gaia=access.checked_spatial(cache,route,gaia_columns,"gaia_dr3.gaia_source",base,lo,hi)
    (work/"validation").mkdir(exist_ok=True)
    gaia.to_csv(work/"validation/gaia.csv",index=False)
    for stem in ("spectra_union","spectra_veto"):
        qso=pd.read_csv(catalogues/f"{stem}.csv",usecols=["ra","dec","reference_r","origin"])
        qso=qso.loc[inside(qso.ra,qso.dec,spec,halo)].copy()
        qso["in_core"]=inside(qso.ra,qso.dec,spec)
        qso.to_csv(work/f"{stem}.csv",index=False)
    standardize(work,{},center,radius,"object")
    method = ("tap_archive_join" if use_tap else "datalab_native_archive_join" if native_join
              else "datalab_object_index")
    save_json(work/"acquisition_summary.json",dict(objects=len(ids),archive_objects_without_measures=len(missing),
        measurement_route=method,native_count_window=native_window,
        downloaded_nsc_measures=len(nsc),chip_metadata_entries=len(chips),
        sdss_measures=len(data),sdss_fields=nfiles,gaia_positions=len(gaia),elapsed_seconds=monotonic()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("region")
    p.add_argument("catalogues",type=Path)
    p.add_argument("sas",type=Path)
    p.add_argument("config",type=Path)
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        p.error("Run on a Slurm compute node")
    settings=tomllib.loads(a.config.read_text())
    work=a.root/a.region
    work.mkdir(parents=True,exist_ok=True)
    outputs=("detections.csv","external_labels.csv","official_nsc.csv","spectra_union.csv","spectra_veto.csv",
             "validation/gaia.csv","acquisition_summary.json","sdss.csv","nsc_object.csv","chips.csv")
    with run_stage(work,a.config,a.region,"exposure-inputs-v09",outputs=outputs) as go:
        if go:
            acquire(work,settings["fields"][a.region],settings,a.catalogues,a.sas,a.workers)


if __name__=="__main__":
    main()
