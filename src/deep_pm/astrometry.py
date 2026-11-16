"""Local gnomonic coordinates and weighted motion models."""
from dataclasses import dataclass

import numpy as np
from astropy.time import Time

MAS_PER_RADIAN = 180 * 3600000 / np.pi


def basis(ra, dec):
    a, d = np.deg2rad([ra, dec])
    center = [np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)]
    east = [-np.sin(a), np.cos(a), 0]
    north = [-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)]
    return np.array([center, east, north])


def project(ra, dec, center):
    a, d = np.deg2rad([ra, dec])
    vectors = np.column_stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    local = vectors @ basis(*center).T
    if np.any(local[:, 0] <= 0):
        raise ValueError("Gnomonic projection requires one hemisphere")
    return local[:, 1:] / local[:, :1] * MAS_PER_RADIAN


def unproject(xy, center):
    xy = np.atleast_2d(xy)
    xyz = np.column_stack([np.ones(len(xy)), xy / MAS_PER_RADIAN]) @ basis(*center)
    return np.column_stack([
        np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360,
        np.rad2deg(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))),
    ])


def sdss_tai_to_tt(seconds):
    return Time(np.asarray(seconds) / 86400, format="mjd", scale="tai").tt.mjd


def utc_to_tt(mjd):
    return Time(mjd, format="mjd", scale="utc").tt.mjd


def years(mjd_tt):
    return 2000 + (np.asarray(mjd_tt) - 51544.5) / 365.25


def seasons(time, gap_days):
    time = np.asarray(time)
    if not len(time):
        return np.empty(0, dtype=int)
    order = np.argsort(time)
    result = np.empty(len(time), dtype=int)
    result[order] = np.cumsum(np.r_[False, np.diff(time[order]) > gap_days / 365.25])
    return result


@dataclass
class Motion:
    coeff: np.ndarray
    cov: np.ndarray
    epoch: float
    residual: np.ndarray
    weight: np.ndarray
    chi2: float
    span: float

    def predict(self, times):
        return self.coeff[0] + (np.asarray(times)[..., None] - self.epoch) * self.coeff[1]


def fit_motion(xy, time, sigma, epoch, robust=True):
    time = np.asarray(time, dtype=float)
    xy = np.asarray(xy, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if len(time) < 2 or np.ptp(time) <= 0 or np.any(~np.isfinite(sigma) | (sigma <= 0)):
        raise ValueError("Linear fit needs distinct epochs and positive finite errors")
    design = np.column_stack([np.ones(len(time)), time - time.mean()])
    transform = np.array([[1, epoch - time.mean()], [0, 1]])
    weight = np.ones(len(time))
    for iteration in range(8 if robust and len(time) >= 3 else 1):
        coeff = []
        cov = []
        for axis in range(2):
            root = np.sqrt(weight) / sigma[:, axis]
            matrix = design * root[:, None]
            normal = matrix.T @ matrix
            coeff.append(np.linalg.lstsq(matrix, xy[:, axis] * root, rcond=None)[0])
            cov.append(transform @ np.linalg.inv(normal) @ transform.T)
        coeff = np.array(coeff).T
        residual = xy - design @ coeff
        distance2 = np.sum((residual / sigma) ** 2, axis=1)
        if iteration < 7 and robust and len(time) >= 3:
            weight = np.minimum(1, 6 / (4 + distance2))
    return Motion(transform @ coeff, np.array(cov), epoch, residual, weight,
                  float(distance2.sum()), float(np.ptp(time)))
