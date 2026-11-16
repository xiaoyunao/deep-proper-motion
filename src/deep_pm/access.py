"""Small, count-checked public catalogue queries and restartable chunk cache."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from matplotlib.path import Path as Polygon

from .astrometry import project

TAP = "https://datalab.noirlab.edu/tap/sync"
DATALAB = "https://datalab.noirlab.edu/query/query"
SDSS = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
MEAS = "measid,objectid,exposure,ccdnum,mjd,ra,dec,raerr,decerr,filter,mag_auto,magerr_auto,class_star,flags,fwhm,x,y"
SDSS_COLUMNS = ("objID,run,rerun,camcol,field,ra,dec,raErr,decErr,TAI_r,psfMag_r,psfMagErr_r,"
                "psfMag_g,psfMag_i,resolveStatus,type,flags,nChild")


def request(service, query):
    if service == "nsc":
        url, params = DATALAB, dict(sql=query, ofmt="csv", **{"async": "False"})
        headers = {"X-DL-AuthToken": "anonymous.0.0.anon_access"}
    elif service == "tap":
        url, params = TAP, dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=query)
        headers = {}
    else:
        url, params = SDSS, dict(cmd=query, format="csv")
        headers = {}
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=(12, 100))
            response.raise_for_status()
            data = response.content
            if data.startswith(b"#Table"):
                data = data.split(b"\n", 1)[1]
            prefix = data.lstrip()[:120].lower()
            if (not prefix or prefix.startswith((b"<", b"error", b"msg ", b"query failed"))
                    or b"query_status" in data[:2048].lower()):
                raise ValueError(f"Unexpected response body from {service}")
            frame = pd.read_csv(BytesIO(data), keep_default_na=True)
            return frame, data
        except (requests.RequestException, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"{service} query failed: {query[:140]}") from exc
            time.sleep(2 ** attempt)


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def query(self, service, query, expected=None):
        key = sha256((service + "\n" + query).encode()).hexdigest()
        csv = self.path / (key + ".csv")
        meta = csv.with_suffix(".json")
        if csv.exists() and meta.exists():
            saved = json.loads(meta.read_text())
            data = csv.read_bytes()
            if sha256(data).hexdigest() == saved["sha256"]:
                frame = pd.read_csv(BytesIO(data))
                if len(frame) == saved["rows"] and (expected is None or len(frame) == expected):
                    return frame
        frame, data = request(service, query)
        if expected is not None and expected != len(frame):
            raise RuntimeError(f"{service}: expected {expected} rows, got {len(frame)} for {key}")
        temporary = csv.with_suffix(".part")
        temporary.write_bytes(data)
        temporary.replace(csv)
        metadata = dict(service=service, query=query, rows=len(frame), sha256=sha256(data).hexdigest())
        temporary = meta.with_suffix(".part")
        temporary.write_text(json.dumps(metadata))
        temporary.replace(meta)
        return frame

    def checked(self, service, columns, source, where, cap=30000):
        suffix = " FROM " + source + " WHERE " + where
        expected = int(self.query(service, "SELECT COUNT(*) AS n" + suffix, expected=1).iloc[0, 0])
        if expected > cap:
            raise RuntimeError(f"{service}: partition has {expected} rows above explicit cap {cap}")
        return self.query(service, "SELECT " + columns + suffix, expected=expected)


def bbox(ra, dec, width):
    dra = width / np.cos(np.deg2rad(min(89.99, abs(dec) + width)))
    lo, hi = ra - dra, ra + dra
    if lo < 0 or hi >= 360:
        ra_clause = f"(ra >= {lo % 360:.10f} OR ra <= {hi % 360:.10f})"
    else:
        ra_clause = f"ra BETWEEN {lo:.10f} AND {hi:.10f}"
    return f"({ra_clause}) AND dec BETWEEN {dec-width:.10f} AND {dec+width:.10f}"


def q3c(ra, dec, radius):
    return f"q3c_radial_query(ra,dec,{ra:.10f},{dec:.10f},{radius:.10f})"


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def fetch_object_index(cache, ra, dec, radius, progress=print):
    objects = cache.checked("nsc", "id AS objectid,ra,dec", "nsc_dr2.object", q3c(ra, dec, radius))
    parts = []
    for start in range(0, len(objects), 150):
        group = objects.objectid.iloc[start:start + 150]
        where = "objectid IN (" + ",".join(map(literal, group)) + ")"
        parts.append(cache.checked("nsc", MEAS, "nsc_dr2.meas", where, cap=15000))
        progress(f"NSC objects {start + len(group)}/{len(objects)}")
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=MEAS.split(","))
    if result.measid.duplicated().any():
        raise RuntimeError("Duplicate measurement IDs in object-index chunks")
    return result, objects


def covering_chips(cache, ra, dec, radius, chip_search):
    corners = ",".join(f"vertex_{axis}{i}" for axis in ("ra", "dec") for i in range(1, 5))
    columns = "exposure,ccdnum,ra,dec,instrument,airmass,mjd,filter," + corners
    chips = cache.checked("nsc", columns, "nsc_dr2.chip", q3c(ra, dec, radius + chip_search))
    selected = []
    for _, ccd in chips.iterrows():
        polygon = project([ccd[f"vertex_ra{i}"] for i in range(1, 5)],
                          [ccd[f"vertex_dec{i}"] for i in range(1, 5)], (ra, dec)) / 3600000
        if not np.isfinite(polygon).all():
            continue
        boundary = Polygon(np.vstack([polygon, polygon[0]]), closed=True)
        # Test a circumscribed square, then trim actual detections by spherical distance.
        square = Polygon([(-radius, -radius), (radius, -radius), (radius, radius),
                          (-radius, radius), (-radius, -radius)], closed=True)
        if boundary.intersects_path(square, filled=True):
            selected.append(ccd)
    return pd.DataFrame(selected, columns=chips.columns)


def fetch_exposure_index(cache, chips, ra, dec, radius, progress=print):
    parts = []
    for i, (exposure, group) in enumerate(chips.groupby("exposure")):
        ccds = ",".join(str(int(value)) for value in sorted(set(group.ccdnum)))
        where = f"exposure={literal(exposure)} AND ccdnum IN ({ccds}) AND ({bbox(ra, dec, radius)})"
        parts.append(cache.checked("nsc", MEAS, "nsc_dr2.meas", where))
        if i % 10 == 0:
            progress(f"NSC exposure {i + 1}/{chips.exposure.nunique()}")
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=MEAS.split(","))
    if frame.measid.duplicated().any():
        raise RuntimeError("Duplicate measurement IDs in exposure chunks")
    return frame


def fetch_sdss(cache, ra, dec, radius):
    return cache.checked("sdss", SDSS_COLUMNS, "PhotoObjAll", bbox(ra, dec, radius))


def read_sdss_field(path):
    """Read the SDSS r-band quantities for a single independently processed field."""
    with fits.open(path, memmap=True) as hdus:
        table = hdus[1].data
        scalar = ["ra", "dec", "raerr", "decerr", "type", "nchild"]
        out = {key: np.asarray(table[key], dtype=float) for key in scalar}
        out["tai_r"] = np.asarray(table["tai"][:, 2], dtype=float)
        out["psfmag_r"] = np.asarray(table["psfmag"][:, 2], dtype=float)
        out["psfmagerr_r"] = np.asarray(table["psfmagerr"][:, 2], dtype=float)
        return pd.DataFrame(out)
