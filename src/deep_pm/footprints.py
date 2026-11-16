"""Catalogue exposure geometry, independent of actual source detections."""
from dataclasses import dataclass

from astropy.io import fits
from matplotlib.path import Path as Polygon
import numpy as np

from .astrometry import project, sdss_tai_to_tt, utc_to_tt, years


def wrap(angle):
    return (angle + 180) % 360 - 180


def sdss_great_circle(ra, dec, node, incl):
    angle = np.deg2rad(wrap(ra - node))
    decl = np.deg2rad(dec)
    i = np.deg2rad(incl)
    x = np.cos(angle) * np.cos(decl)
    y = np.sin(angle) * np.cos(decl)
    z = np.sin(decl)
    mu = node + np.rad2deg(np.arctan2(y * np.cos(i) + z * np.sin(i), x))
    nu = np.rad2deg(np.arcsin(np.clip(z * np.cos(i) - y * np.sin(i), -1, 1)))
    return mu, nu


@dataclass
class Visit:
    exposure: str
    survey: str
    year: float
    band: str
    coverage: object


def nsc_visits(chips, center):
    for exposure, group in chips.groupby("exposure"):
        polygons = []
        for row in group.itertuples():
            corners = np.array([(getattr(row, f"vertex_ra{i}"), getattr(row, f"vertex_dec{i}"))
                                for i in range(1, 5)])
            if np.isfinite(corners).all():
                points = project(corners[:, 0], corners[:, 1], center)
                polygons.append(Polygon(np.vstack([points, points[0]]), closed=True))
        if not polygons:
            continue

        def covered(xy, _sky, polygons=tuple(polygons)):
            return any(boundary.contains_point(xy, radius=-1) for boundary in polygons)

        row = group.iloc[0]
        yield Visit("NSC:" + str(exposure), "NSC", float(years(utc_to_tt(row.mjd))),
                    row["filter"], covered)


def sdss_visits(path, center, radius, rerun=None, vectorized=False):
    with fits.open(path, memmap=True) as hdus:
        table = hdus[1].data
        # Field center metadata preselection only; tests below use the native scan bounds.
        preselect = ((np.abs(table["DEC"] - center[1]) <= radius + 1)
                     & (np.abs(wrap(table["RA"] - center[0]))
                        <= (radius + 1) / np.cos(np.deg2rad(center[1]))))
        if rerun is not None:
            preselect &= np.char.strip(np.asarray(table["RERUN"], dtype=str)) == str(rerun)
        selected = table[np.flatnonzero(preselect)]
        keys = ["RUN", "TAI", "NODE", "INCL", "MU_START", "MU_END", "NU_START", "NU_END"]
        columns = {key: np.asarray(selected[key]).astype(float) for key in keys}
    for run in np.unique(columns["RUN"]):
        fields = columns["RUN"] == run
        mid = columns["TAI"][fields]
        if mid.ndim == 2:
            mid = mid[:, 2]
        mid = mid[np.isfinite(mid) & (mid >= 4e9) & (mid < 6e9)]
        if not len(mid):
            continue
        boundaries = [columns[key][fields] for key in keys[2:]]

        def covered(_xy, sky, limits=boundaries):
            node, incl, a, b, c, d = limits
            width = (b - a) % 360
            if vectorized:
                mu, nu = sdss_great_circle(sky[:, 0, None], sky[:, 1, None], node, incl)
            else:
                mu, nu = sdss_great_circle(*sky, node, incl)
            result = np.any(((mu - a) % 360 <= width) & (nu >= np.minimum(c, d))
                            & (nu <= np.maximum(c, d)), axis=-1)
            return result if vectorized else bool(result)

        yield Visit(f"SDSS:{int(run)}", "SDSS", float(years(sdss_tai_to_tt(np.median(mid)))),
                    "r", covered)
