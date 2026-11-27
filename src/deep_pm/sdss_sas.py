"""Read regional SDSS DR17 single-epoch data from the shared SAS mirror."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from astropy.io import fits
import numpy as np
import pandas as pd

from .access import SDSS_COLUMNS
from .astrometry import project


def select_sdss_field(root, center, limit, key, run_primary=True):
    run, rerun, camera, field = key
    path = root/"photoObj"/str(rerun)/str(run)/str(camera)/f"photoObj-{run:06d}-{camera}-{field:04d}.fits"
    with fits.open(path, memmap=True) as hf:
        source = hf[1].data
        if source is None and hf[1].header.get("NAXIS2", 0)==0:
            return None
        selected = (np.flatnonzero((source["RESOLVE_STATUS"] & 1) != 0)
                    if run_primary else np.arange(len(source)))
        inside = np.linalg.norm(project(source["RA"][selected],source["DEC"][selected],center),axis=1) <= limit
        selected = selected[inside]
        if not len(selected):
            return None
        frame = dict(objID=[f"{run}:{camera}:{field}:{int(i)}" for i in source["ID"][selected]],
                     run=run,rerun=rerun,camcol=camera,field=field)
        scalar = dict(ra="RA",dec="DEC",raErr="RAERR",decErr="DECERR",resolveStatus="RESOLVE_STATUS",
                      type="OBJC_TYPE",flags="OBJC_FLAGS",nChild="NCHILD",mjd="MJD")
        vector = dict(TAI_r=("TAI",2),psfMag_r=("PSFMAG",2),psfMagErr_r=("PSFMAGERR",2),
                      psfMag_g=("PSFMAG",1),psfMag_i=("PSFMAG",3),offsetRa_r=("OFFSETRA",2),
                      offsetDec_r=("OFFSETDEC",2),psffwhm_r=("PSF_FWHM",2))
        for col,name in scalar.items():
            frame[col] = np.asarray(source[name][selected],dtype=float)
        for col,(name,idx) in vector.items():
            frame[col] = np.asarray(source[name][selected,idx],dtype=float)
        frame["airmass_r"] = np.nan
    return pd.DataFrame(frame,columns=SDSS_COLUMNS.split(","))


def read_sdss_sas(root, center, radius, workers=1):
    index_path = root/"resolve"/"2013-07-29"/"window_flist.fits"
    # The frame size is about 13.5 x 9 arcmin. The circumscribed margin
    # includes field edges and adjacent scans before the final spherical cut.
    margin = radius + .18
    with fits.open(index_path, memmap=True) as hdus:
        fields = hdus[1].data
        dra = (fields["RA"] - center[0] + 180) % 360 - 180
        max_lat = abs(center[1])+margin
        box = (np.ones(len(dra),dtype=bool) if max_lat >= 90 else
               abs(dra) < margin/np.cos(np.deg2rad(max_lat)))
        box &= abs(fields["DEC"] - center[1]) < margin
        locations = fields[box]
        in_radius = np.linalg.norm(project(locations["RA"], locations["DEC"], center), axis=1)
        locations = locations[in_radius <= np.tan(np.deg2rad(margin))*180/np.pi*3_600_000]
        field_keys = sorted(set((int(row["RUN"]), int(row["RERUN"]), int(row["CAMCOL"]),
                                 int(row["FIELD"])) for row in locations if int(row["RERUN"]) == 301))
    limit = np.tan(np.deg2rad(radius))*180/np.pi*3_600_000
    read = partial(select_sdss_field,root,center,limit)
    if workers>1 and len(field_keys)>1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts=[part for part in pool.map(read,field_keys) if part is not None]
    else:
        parts=[part for part in map(read,field_keys) if part is not None]
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SDSS_COLUMNS.split(","))
    return result, len(field_keys)
