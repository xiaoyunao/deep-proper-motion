"""Low-dimensional residual colour and directional DCR on fixed detections."""
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils import iers
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr


SITES = {"sdss": (32.78036, -105.82042), "c4d": (-30.1690, -70.8065),
         "k4m": (31.9584, -111.5986), "ksb": (31.9629, -111.6005)}


def zenith_vector(lat, hour_angle, dec):
    east = np.cos(lat) * np.sin(hour_angle)
    north = np.sin(lat) * np.cos(dec) - np.cos(lat) * np.sin(dec) * np.cos(hour_angle)
    cos_z = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour_angle)
    return np.column_stack((east, north)), cos_z


def directions(frame, metadata):
    nsc = frame.survey.eq("NSC")
    lookup = metadata.set_index("exposure", verify_integrity=True)
    inst = frame.exposure_id.str.replace(r"^NSC:", "", regex=True).map(lookup.instrument)
    inst.loc[~nsc] = "sdss"
    airmass = frame.exposure_id.str.replace(r"^NSC:", "", regex=True).map(lookup.airmass)
    airmass.loc[~nsc] = frame.loc[~nsc, "airmass"]
    out = np.full((len(frame), 2), np.nan)
    for name, (lat, lon) in SITES.items():
        idx = np.flatnonzero(inst.eq(name))
        if not len(idx):
            continue
        rows = frame.iloc[idx]
        site = EarthLocation.from_geodetic(lon*u.deg, lat*u.deg)
        with iers.conf.set_temp("auto_download", False):
            moment = Time(rows.observation_time.to_numpy(), format="mjd", scale="tt", location=site)
            lst = moment.sidereal_time("mean").rad
        vector, cos_z = zenith_vector(np.deg2rad(lat), lst-np.deg2rad(rows.ra_deg.to_numpy()),
                                      np.deg2rad(rows.dec_deg.to_numpy()))
        a = airmass.iloc[idx].to_numpy()
        good = (cos_z >= .4) & np.isfinite(a) & (a >= .98) & (abs(1/np.maximum(cos_z, .4)-a) <= .22)
        out[idx[good]] = vector[good] / cos_z[good, None]
    return inst.fillna("unknown") + "/" + frame.band.astype(str), out


def stable_color(frame, group):
    g = frame.loc[frame.survey.eq("SDSS") & frame.color.between(-.7, 4.5)]
    stats = g.assign(source=group.loc[g.index]).groupby("source").color
    center = stats.median()
    width = stats.quantile(.75) - stats.quantile(.25)
    return center.loc[(stats.size() >= 2) & (width <= .30)]


def templates(keys, direction, colors, levels, include_dcr):
    values = []
    names = []
    for name in levels:
        color = np.where(keys == name, colors, 0)
        if name != "sdss/r":
            values.extend([np.column_stack([color, np.zeros(len(color))]),
                           np.column_stack([np.zeros(len(color)), color])])
            names.extend([name + "/E", name + "/N"])
        if include_dcr:
            values.append(color[:, None] * direction)
            names.append(name + "/DCR")
    return np.stack(values, axis=2), names


def chromatic_fit(refs, groups, xy, sigma, feature, draws, seed):
    ri, sources = pd.factorize(refs)
    gi, visits = pd.factorize(groups)
    n = len(ri)
    idx = np.arange(n)
    design = coo_matrix((np.ones(2*n), (np.r_[idx, idx], np.r_[ri, len(sources)+gi])),
                        shape=(n, len(sources)+len(visits))).tocsr()
    # Residualize each column on object and visit intercepts before solving
    # for joint coefficients. The fitted projections have no gauge freedom.
    results = []
    for axis in range(2):
        noise = sigma[:, axis]
        mat = design.multiply((1/noise)[:, None]).tocsr()
        y = np.column_stack([xy[:, axis], feature[:, axis]]) / noise[:, None]
        x = []
        for j in range(y.shape[1]):
            best = lsmr(mat, y[:, j], atol=1e-10, btol=1e-10, maxiter=2000)
            if best[1] not in (0, 1, 2, 4, 5):
                raise RuntimeError("Colour design residualization did not converge")
            x.append(y[:, j] - mat @ best[0])
        results.append(np.column_stack(x))
    data = np.vstack(results)
    response, model = data[:, 0], data[:, 1:]
    fit, _, rank, scale = np.linalg.lstsq(model, response, rcond=1e-6)
    raw_scale = np.linalg.norm(feature / sigma[:, :, None], axis=(0, 1))
    if (rank != model.shape[1] or np.any(np.linalg.norm(model, axis=0) <= 1e-7 * raw_scale)
            or scale[0]/scale[-1] > 1e5):
        raise ValueError("Colour/DCR is confounded with source and visit intercepts")
    for _ in range(4):
        w = np.sqrt(np.minimum(1, 2.5 / np.maximum(2.5, abs(response-model@fit))))
        fit = np.linalg.lstsq(model*w[:, None], response*w, rcond=1e-6)[0]
    weights = np.tile(ri, 2)
    rng = np.random.default_rng(seed)
    drawn = []
    for _ in range(draws):
        wb = w * np.sqrt(rng.exponential(size=len(sources))[weights])
        drawn.append(np.linalg.lstsq(model*wb[:, None], response*wb, rcond=1e-6)[0])
    return fit, np.array(drawn), float(scale[0]/scale[-1])
