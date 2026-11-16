"""Read regional SDSS DR17 single-epoch data from the shared SAS mirror."""
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd

from .access import SDSS_COLUMNS
from .astrometry import project


def read_sdss_sas(root, center, radius):
    index_path = root/"resolve"/"2013-07-29"/"window_flist.fits"
    parts = []
    # The frame size is about 13.5 x 9 arcmin. The circumscribed margin
    # includes field edges and adjacent scans before the final spherical cut.
    margin = radius + .18
    with fits.open(index_path, memmap=True) as hdus:
        fields = hdus[1].data
        dra = (fields["RA"] - center[0] + 180) % 360 - 180
        box = (abs(dra) < margin/np.cos(np.deg2rad(abs(center[1])+margin)))
        box &= abs(fields["DEC"] - center[1]) < margin
        locations = fields[box]
        in_radius = np.linalg.norm(project(locations["RA"], locations["DEC"], center), axis=1)
        locations = locations[in_radius <= np.tan(np.deg2rad(margin))*180/np.pi*3_600_000]
        field_keys = sorted(set((int(row["RUN"]), int(row["RERUN"]), int(row["CAMCOL"]),
                                 int(row["FIELD"])) for row in locations if int(row["RERUN"]) == 301))
    limit = np.tan(np.deg2rad(radius))*180/np.pi*3_600_000
    for run, rerun, camera, field in field_keys:
        path = root/"photoObj"/str(rerun)/str(run)/str(camera)/f"photoObj-{run:06d}-{camera}-{field:04d}.fits"
        with fits.open(path, memmap=True) as hf:
            source = hf[1].data
            keep = (source["RESOLVE_STATUS"] & 1) != 0
            block = source[keep]
            inside = np.linalg.norm(project(block["RA"],block["DEC"],center),axis=1) <= limit
            a = block[inside]
            if not len(a):
                continue
            frame = dict(objID=[f"{run}:{camera}:{field}:{int(i)}" for i in a["ID"]],
                         run=run,rerun=rerun,camcol=camera,field=field)
            scalar = dict(ra="RA",dec="DEC",raErr="RAERR",decErr="DECERR",resolveStatus="RESOLVE_STATUS",
                          type="OBJC_TYPE",flags="OBJC_FLAGS",nChild="NCHILD",mjd="MJD")
            vector = dict(TAI_r=("TAI",2),psfMag_r=("PSFMAG",2),psfMagErr_r=("PSFMAGERR",2),
                          psfMag_g=("PSFMAG",1),psfMag_i=("PSFMAG",3),offsetRa_r=("OFFSETRA",2),
                          offsetDec_r=("OFFSETDEC",2),psffwhm_r=("PSF_FWHM",2))
            for col,key in scalar.items():
                frame[col] = np.asarray(a[key], dtype=float)
            for col,(key,idx) in vector.items():
                frame[col] = np.asarray(a[key][:,idx], dtype=float)
            frame["airmass_r"] = np.nan
            parts.append(pd.DataFrame(frame, columns=SDSS_COLUMNS.split(",")))
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SDSS_COLUMNS.split(","))
    return result, len(field_keys)
