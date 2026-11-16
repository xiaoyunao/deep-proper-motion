"""Select independent QSO fields from geometry before reading their proper motions."""
import argparse
from collections import defaultdict
from pathlib import Path
import os

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import access, normalize
from .__main__ import in_cone, standardize
from .astrometry import project
from .io import save_json
from .nsc_comparison import configuration
from .provenance import run_stage


def spectra(file, center, radius, buffer):
    with fits.open(file, memmap=True) as handle:
        table = handle["QSO_CAT"].data
        ra, dec = table["TARGET_RA"], table["TARGET_DEC"]
        delta_ra = (ra - center[0] + 180) % 360 - 180
        outer = radius + buffer
        close = (abs(delta_ra) < outer/np.cos(np.deg2rad(abs(center[1])+outer))) & (abs(dec-center[1]) < outer)
        subset = table[close]
        good = ((subset["Z"] > .5) & (subset["Z"] < 6) & (subset["ZWARN"] == 0)
                & (subset["FLUX_R"] > 0) & (subset["FLUX_IVAR_R"] > 0))
        data = subset[good]
        frame = pd.DataFrame(dict(target_id=data["TARGETID"].astype(np.int64),
                                  ra=data["TARGET_RA"].astype(float), dec=data["TARGET_DEC"].astype(float),
                                  z=data["Z"].astype(float),
                                  reference_r=22.5-2.5*np.log10(data["FLUX_R"].astype(float)),
                                  snr=data["TSNR2_QSO"].astype(float)))
    frame = frame.sort_values("snr", ascending=False).drop_duplicates("target_id")
    frame = in_cone(frame, center, outer)
    frame["in_core"] = frame.index.isin(in_cone(frame, center, radius).index)
    return frame


def object_index(cache, center, radius):
    return access.checked_spatial(cache, "nsc", "id AS objectid,ra,dec", "nsc_dr2.object",
                                  access.q3c(*center, radius),
                                  center[1]-radius-1e-8, center[1]+radius+1e-8)


def inventory(work, spec, settings, catalog):
    center, radius = (spec["ra"],spec["dec"]), spec["radius"]
    buffer = settings["calibration_buffer_deg"]
    qso = spectra(catalog, center, radius, buffer)
    qso.to_csv(work/"spectroscopic_qso.csv", index=False)
    cache = access.Cache(work/"cache")
    sdss = access.fetch_sdss(cache, *center, radius+buffer)
    sdss.to_csv(work/"sdss.csv", index=False)
    objects = object_index(cache, center, radius+buffer+.003)
    objects.to_csv(work/"nsc_object_index.csv", index=False)
    sd = normalize.sdss(sdss)
    sd = sd.loc[sd.observation_time.notna() & sd.magnitude.notna()]
    points = project(qso.ra, qso.dec, center)
    d_sdss = cKDTree(project(sd.ra_deg, sd.dec_deg, center)).query(points)[0]
    d_nsc = cKDTree(project(objects.ra, objects.dec, center)).query(points)[0]
    candidate = (d_sdss <= 500) & (d_nsc <= 1000) & qso.in_core
    bright_candidate = (d_sdss <= 500) & qso.in_core
    qso["coverage_candidate"] = candidate
    qso.to_csv(work/"spectroscopic_qso.csv", index=False)
    statistics = dict(area_sqdeg=np.pi*radius**2, spectra_core=int(qso.in_core.sum()),
                      spectra_faint_core=int((qso.in_core & qso.reference_r.between(21,23,inclusive="left")).sum()),
                      sdss_detection_rows=len(sd), sdss_run_count=sd.exposure_id.nunique(),
                      sdss_date_count=sd.observation_time.astype(int).nunique(), nsc_object_rows=len(objects),
                      candidates_with_sdss=int(bright_candidate.sum()),
                      candidates_with_both=int(candidate.sum()),
                      faint_candidates_with_both=int((candidate&qso.reference_r.between(21,23,inclusive="left")).sum()))
    save_json(work/"qso_coverage_summary.json", statistics)


def targeted_measurements(work, spec, settings):
    center, radius = (spec["ra"],spec["dec"]), spec["radius"]
    buffer = settings["calibration_buffer_deg"]
    cache = access.Cache(work/"cache")
    qso = pd.read_csv(work/"spectroscopic_qso.csv")
    objects = pd.read_csv(work/"nsc_object_index.csv", dtype={"objectid":str})
    sd = normalize.sdss(pd.read_csv(work/"sdss.csv"))
    ref = sd.morphology.eq(3) & sd.nchild.eq(0) & sd.magnitude.between(16,22) & sd.magnitude_error.between(0,.15)
    references = sd.loc[ref & sd.observation_time.notna(), ["ra_deg","dec_deg"]].drop_duplicates()
    known_xy = np.vstack([project(references.ra_deg,references.dec_deg,center),
                          project(qso.ra,qso.dec,center)])
    tree = cKDTree(project(objects.ra,objects.dec,center))
    neighborhoods = tree.query_ball_point(known_xy, 3100)
    selected = sorted({i for hit in neighborhoods for i in hit})
    print(f"{len(selected)} NSC object-index candidates within 3.1 arcsec of SDSS references or QSO", flush=True)
    measurements = access.fetch_measurements_for_objects(cache, objects.objectid.iloc[selected])
    measurements.to_csv(work/"nsc_object.csv", index=False)
    gaia_columns = ("source_id,ra,dec,ref_epoch,pmra,pmdec,pmra_error,pmdec_error,parallax,"
                    "ruwe,phot_g_mean_mag,phot_rp_mean_mag,parallax_error,radial_velocity,astrometric_params_solved")
    region = radius+buffer+.003
    gaia = access.checked_spatial(cache,"nsc",gaia_columns,"gaia_dr3.gaia_source",access.q3c(*center,region),
                                  center[1]-region-1e-8,center[1]+region+1e-8)
    (work/"validation").mkdir(exist_ok=True)
    gaia.to_csv(work/"validation"/"gaia.csv", index=False)
    standardize(work,{},center,radius,"object")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("region")
    p.add_argument("stage",choices=("inventory","fetch"))
    p.add_argument("--catalogue",type=Path)
    p.add_argument("--config",type=Path,default=Path("configs/precision-v04.toml"))
    args=p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Run on a Slurm compute node.")
    args.work.mkdir(parents=True,exist_ok=True)
    _, settings = configuration(args.config)
    spec=settings["fields"][args.region]
    if args.stage=="inventory":
        if args.catalogue is None or not args.catalogue.is_file():
            p.error("Select the verified DESI DR1 QSO FITS.")
        # Receipt hashes the regional selection; the common FITS lives on NERSC.
        with run_stage(args.work,args.config,args.region,"qso-coverage-v04",
                       outputs=["spectroscopic_qso.csv","sdss.csv","nsc_object_index.csv",
                                "qso_coverage_summary.json"]) as go:
            if go:
                inventory(args.work,spec,settings,args.catalogue)
    else:
        with run_stage(args.work,args.config,args.region,"qso-fetch-v04",
                       inputs=["spectroscopic_qso.csv","sdss.csv","nsc_object_index.csv"],
                       outputs=["nsc_object.csv","validation/gaia.csv","detections.csv","external_labels.csv"]) as go:
            if go:
                targeted_measurements(args.work,spec,settings)


if __name__=="__main__":
    main()
