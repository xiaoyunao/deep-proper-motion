"""Convert verified survey columns to the shared detection format."""
import numpy as np
import pandas as pd

from .astrometry import sdss_tai_to_tt, utc_to_tt


def base(survey, release, ids, exposure, detector, ra, dec, mjd, east, north,
         band, magnitude, magerr, shape, flags, objectid, provenance):
    result = pd.DataFrame(dict(
        survey=survey, release=release, detection_id=ids, exposure_id=exposure,
        detector_id=detector, ra_deg=ra, dec_deg=dec, observation_time=mjd,
        time_scale="TT", sigma_east_mas=east, sigma_north_mas=north,
        band=band, magnitude=magnitude, magnitude_error=magerr, morphology=shape,
        quality_flags=flags, original_object_id=objectid, provenance=provenance,
    ))
    result = result.replace([np.inf, -np.inf], np.nan)
    for column in ("sigma_east_mas", "sigma_north_mas"):
        result.loc[~result[column].between(1e-8, 5000), column] = np.nan
    result.loc[~result.magnitude.between(-10, 50), "magnitude"] = np.nan
    result.loc[~result.magnitude_error.between(0, 20), "magnitude_error"] = np.nan
    return result


def nsc(table):
    t = table
    mjd = t.mjd.where(t.mjd.between(50000, 62000))
    out = base("NSC", "DR2", "NSC:" + t.measid.astype(str), "NSC:" + t.exposure.astype(str),
               t.ccdnum, t.ra, t.dec, utc_to_tt(mjd.fillna(55000)).astype(float),
               t.raerr * 1000, t.decerr * 1000, t["filter"], t.mag_auto, t.magerr_auto,
               t.class_star, t["flags"], t.objectid, "nsc_dr2.meas")
    out.loc[mjd.isna(), "observation_time"] = np.nan
    for column in ("fwhm", "x", "y", "airmass", "instrument", "color"):
        if column in t:
            out[column] = t[column]
    return out.drop_duplicates("detection_id")


def sdss(table):
    t = table.rename(columns=str.lower).copy()
    # Use the r epoch only when both r-to-reference offsets are zero.
    tai = t.tai_r.where(t.tai_r.between(4e9, 6e9))
    has_r = t.offsetra_r.eq(0) & t.offsetdec_r.eq(0)
    tai = tai.where(has_r)
    physical_id = t.run.astype(str) + ":" + t.camcol.astype(str)
    out = base("SDSS", "DR17", "SDSS:" + t.objid.astype(str), "SDSS:" + t.run.astype(str),
               t.camcol, t.ra, t.dec, sdss_tai_to_tt(tai.fillna(5e9)), t.raerr * 1000,
               t.decerr * 1000, "r", t.psfmag_r, t.psfmagerr_r, t["type"], t["flags"],
               t.objid.astype(str), "SDSS DR17 single scan")
    out.loc[tai.isna(), "observation_time"] = np.nan
    out["time_band_status"] = np.where(has_r, "r_reference", "reference_band_unverified")
    for source, target in [("airmass_r", "airmass"), ("psffwhm_r", "fwhm"), ("nchild", "nchild")]:
        out[target] = t[source]
    out["color"] = t.psfmag_g - t.psfmag_i if {"psfmag_g", "psfmag_i"} <= set(t) else np.nan
    out["scan_detector"] = physical_id
    if "resolvestatus" in t:
        # RUN_PRIMARY = 0x1. Preserve runs and discard overlap duplicates.
        out = out[(t.resolvestatus.fillna(0).astype("int64") & 1) != 0]
    return out.drop_duplicates("detection_id")
