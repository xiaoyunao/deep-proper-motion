"""Acquire the frozen regional NSC/SDSS overlap and separate validation tables."""
import argparse
import os
from pathlib import Path
import resource
from time import monotonic

import pandas as pd

from . import access
from .__main__ import standardize
from .io import save_json
from .nsc_comparison import configuration
from .provenance import run_stage
from .qso_fields import spectra
from .sdss_sas import read_sdss_sas

GAIA_COLUMNS = ("source_id,ra,dec,ref_epoch,pmra,pmdec,pmra_error,pmdec_error,parallax,"
                "ruwe,phot_g_mean_mag,phot_rp_mean_mag,parallax_error,radial_velocity,astrometric_params_solved")


def acquire(work, name, spec, settings, catalogue, sas):
    start = monotonic()
    center = spec["ra"],spec["dec"]
    radius = spec["radius"]
    buffer = settings["calibration_buffer_deg"]
    outer = radius+buffer+.003
    cache = access.Cache(work/"cache")
    qso = spectra(catalogue,center,radius,buffer)
    qso.to_csv(work/"spectroscopic_qso.csv",index=False)
    sdss, n_fields = read_sdss_sas(sas,center,radius+buffer)
    sdss.to_csv(work/"sdss.csv",index=False)
    print(f"{name}: {len(sdss):,} SDSS measurements",flush=True)
    selection = access.q3c(*center,outer)+" AND ndetr>0 AND (rmag<24 OR nphotr=0 OR rmag IS NULL)"
    archive = access.checked_spatial(cache,"nsc","id AS objectid,ra,dec,pmra,pmdec,pmraerr,pmdecerr,deltamjd",
                                     "nsc_dr2.object",selection,
                                     center[1]-outer-1e-8,center[1]+outer+1e-8)
    ids = archive.objectid.astype(str).sort_values()
    if len(ids)>settings["regional"]["max_archive_candidates"] or ids.duplicated().any():
        raise ValueError("Archive object count exceeds the frozen regional bound or is nonunique.")
    archive.rename(columns={"objectid":"id"}).to_csv(work/"official_nsc.csv",index=False)
    print(f"{name}: {len(ids):,} NSC archive objects",flush=True)
    nsc = access.fetch_measurements_for_objects(cache,ids,batch_size=250)
    if set(nsc.objectid.astype(str))!=set(ids):
        raise ValueError("Archive index/returned measurement object mismatch.")
    nsc.to_csv(work/"nsc_object.csv",index=False)
    gaia_r = outer+.001
    gaia = access.checked_spatial(cache,"nsc",GAIA_COLUMNS,"gaia_dr3.gaia_source",
                                  access.q3c(*center,gaia_r),center[1]-gaia_r-1e-8,center[1]+gaia_r+1e-8)
    (work/"validation").mkdir(exist_ok=True)
    gaia.to_csv(work/"validation"/"gaia.csv",index=False)
    standardize(work,{},center,radius,"object")
    save_json(work/"acquisition_summary.json",dict(objects=len(ids),measurements=len(nsc),sdss_measurements=len(sdss),
              sdss_fields=n_fields,gaia_positions=len(gaia),spectra=len(qso),wall_seconds=monotonic()-start,
              peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              raw_bytes=sum((work/f).stat().st_size for f in ("nsc_object.csv","sdss.csv","official_nsc.csv"))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("region")
    p.add_argument("catalogue",type=Path)
    p.add_argument("sas",type=Path)
    p.add_argument("--config",type=Path,default=Path("configs/regional-v07.toml"))
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Use a Slurm compute node.")
    _,settings=configuration(args.config)
    args.work.mkdir(parents=True,exist_ok=True)
    outputs=["detections.csv","external_labels.csv","official_nsc.csv","spectroscopic_qso.csv",
             "validation/gaia.csv","sdss.csv","nsc_object.csv","acquisition_summary.json"]
    stage = "regional-inputs-" + settings.get("run_series", "v07")
    with run_stage(args.work,args.config,args.region,stage,outputs=outputs) as go:
        if go:
            acquire(args.work,args.region,settings["fields"][args.region],settings,args.catalogue,args.sas)


if __name__=="__main__":
    main()
