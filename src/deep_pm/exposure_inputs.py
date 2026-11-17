"""Acquire a frozen rectangular core and halo while retaining physical detector IDs."""
import argparse
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
    base=where_box(spec,halo)
    south,north=spec["dec"]
    lo,hi=south-halo-1e-7,north+halo+1e-7
    query=base+" AND ndetr>0 AND (rmag<24 OR nphotr=0 OR rmag IS NULL)"
    archive=access.checked_spatial(cache,"nsc","id AS objectid,ra,dec,pmra,pmdec,pmraerr,pmdecerr,deltamjd",
                                   "nsc_dr2.object",query,lo,hi)
    ids=archive.objectid.astype(str).sort_values()
    if ids.duplicated().any():
        raise ValueError("Duplicate NSC object keys in disjoint archive partitions")
    print(f"{len(ids):,} NSC archive objects",flush=True)
    archive.rename(columns={"objectid":"id"}).to_csv(work/"official_nsc.csv",index=False)
    nsc=access.fetch_measurements_for_objects(cache,ids,batch_size=250)
    if set(nsc.objectid.astype(str))!=set(ids):
        raise ValueError("Archive object and detection partition disagree")
    nsc.loc[inside(nsc.ra,nsc.dec,spec,halo)].to_csv(work/"nsc_object.csv",index=False)
    gaia=access.checked_spatial(cache,"nsc",GAIA_COLUMNS,"gaia_dr3.gaia_source",base,lo,hi)
    (work/"validation").mkdir(exist_ok=True)
    gaia.to_csv(work/"validation/gaia.csv",index=False)
    for stem in ("spectra_union","spectra_veto"):
        qso=pd.read_csv(catalogues/f"{stem}.csv",usecols=["ra","dec","reference_r","origin"])
        qso=qso.loc[inside(qso.ra,qso.dec,spec,halo)].copy()
        qso["in_core"]=inside(qso.ra,qso.dec,spec)
        qso.to_csv(work/f"{stem}.csv",index=False)
    standardize(work,{},center,radius,"object")
    save_json(work/"acquisition_summary.json",dict(objects=len(ids),downloaded_nsc_measures=len(nsc),
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
             "validation/gaia.csv","acquisition_summary.json","sdss.csv","nsc_object.csv")
    with run_stage(work,a.config,a.region,"exposure-inputs-v09",outputs=outputs) as go:
        if go:
            acquire(work,settings["fields"][a.region],settings,a.catalogues,a.sas,a.workers)


if __name__=="__main__":
    main()
