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
    a, d = np.deg2rad(np.asarray([ra, dec], dtype=float))
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


def apparent_positions(ra, dec, pm_east, pm_north, parallax, radial_velocity, dt, observer, center):
    """Rectilinear ICRS propagation and annual parallax for validation only."""
    pm_east, pm_north, parallax, radial_velocity, dt = map(
        np.asarray, (pm_east, pm_north, parallax, radial_velocity, dt))
    alpha, delta = np.deg2rad([ra, dec])
    ca, sa = np.cos(alpha), np.sin(alpha)
    cd, sd = np.cos(delta), np.sin(delta)
    direction = np.stack([cd * ca, cd * sa, sd], axis=-1)
    east = np.stack([-sa, ca, np.zeros_like(sa)], axis=-1)
    north = np.stack([-sd * ca, -sd * sa, cd], axis=-1)
    angular_velocity = (pm_east[..., None] * east + pm_north[..., None] * north) / MAS_PER_RADIAN
    # km/s converted to AU/year, then relative to the Gaia distance.
    radial_fraction = radial_velocity * (365.25 * 86400 / 149597870.7) * parallax / MAS_PER_RADIAN
    xyz = (direction + dt[..., None] * (angular_velocity + radial_fraction[..., None] * direction)
           - observer * (parallax[..., None] / MAS_PER_RADIAN))
    local = xyz @ basis(*center).T
    return local[..., 1:] / local[..., :1] * MAS_PER_RADIAN


def epoch_halo_deg(config, mu=None):
    """Pad a survey query by the largest reference-to-endpoint displacement."""
    settings = config["access"]
    if "epoch_min" not in settings:
        return settings["halo_deg"]
    epoch = config["reference_epoch"]
    span = max(abs(epoch - settings[name]) for name in ("epoch_min", "epoch_max"))
    rate = config["mu_max"] if mu is None else mu
    padding = settings.get("edge_pad_mas", 1000) + 2 * settings.get("max_parallax_mas", 0)
    return max(settings.get("halo_deg", 0), (rate * span + padding) / 3600000)


def sdss_tai_to_tt(seconds):
    return Time(np.asarray(seconds, dtype=float) / 86400, format="mjd", scale="tai").tt.mjd


def utc_to_tt(mjd):
    return Time(np.asarray(mjd, dtype=float), format="mjd", scale="utc").tt.mjd


def years(mjd_tt):
    return 2000 + (np.asarray(mjd_tt, dtype=float) - 51544.5) / 365.25


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
