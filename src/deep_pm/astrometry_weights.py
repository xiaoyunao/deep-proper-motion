"""Empirical positional variance, correlated errors, and identifiable parallax."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .astrometry import basis


def variance_model(delta, left, right, minimum):
    """Estimate marginal variance from differences of disjoint galaxy epochs."""
    if len(delta) < minimum:
        return np.array([1., 50.]), False
    difference = np.asarray(delta)
    formal = np.asarray(left)**2 + np.asarray(right)**2

    def objective(parameters):
        scale, floor = np.exp(parameters)
        variance = scale**2 * formal + 2 * floor**2
        # Student likelihood with finite variance to limit the impact of mismatches.
        return np.sum(.5*np.log(variance) + 3*np.log1p(difference**2/(3*variance)))

    fit = minimize(objective, np.log([1., 35.]), method="L-BFGS-B",
                   bounds=np.log([[.25, 4.], [5., 160.]]))
    return (np.exp(fit.x), True) if fit.success else (np.array([1., 50.]), False)


def parallax_factors(observer, center, local_position):
    """Derivative of gnomonic position with respect to geocentric parallax."""
    from .astrometry import MAS_PER_RADIAN

    local = np.asarray(local_position) / MAS_PER_RADIAN
    direction_norm = np.sqrt(1 + local @ local)
    projected = observer @ basis(*center).T
    return (projected[:, :1]*local - projected[:, 1:])*direction_norm


@dataclass
class WeightedMotion:
    params: np.ndarray
    covariance: np.ndarray
    residual: np.ndarray
    weights: np.ndarray
    influence: np.ndarray
    parallax_information: float
    chi2: float

    @property
    def motion(self):
        return self.params[[1, 3]]

    @property
    def error(self):
        return np.sqrt(np.diag(self.covariance)[[1, 3]])


def correlated_fit(xy, time, sigma, groups, shared, epoch, samples=None, factors=None):
    """Fit with a rank-one common error per group and axis plus robust weights."""
    n = len(time)
    if n < 2 or np.ptp(time) <= 0:
        raise ValueError("Motion requires distinct epochs")
    design = np.zeros((n, 2, 4+(factors is not None)))
    for axis in range(2):
        design[:, axis, 2*axis] = 1
        design[:, axis, 2*axis+1] = time - epoch
    if factors is not None:
        design[:, :, -1] = factors
    members = {}
    for j, key in enumerate(groups):
        if np.any(shared[j]):
            members.setdefault(key, []).append(j)
    weights = np.ones(n)
    for iteration in range(8 if n >= 3 else 1):
        precision = weights[:, None] / sigma**2
        weighted = precision[:, :, None] * design
        for ix in members.values():
            w = precision[ix]
            common = shared[ix[0]]**2
            weighted[ix] -= (w * (common/(1+common*w.sum(axis=0))))[:, :, None]*weighted[ix].sum(axis=0)
        matrix = design.reshape(2*n, -1)
        inverse_design = weighted.reshape(matrix.shape).T
        covariance = np.linalg.inv(inverse_design @ matrix)
        influence = (covariance @ inverse_design).reshape(-1, n, 2)
        params = influence.reshape(-1,2*n) @ xy.ravel()
        residual = xy - design @ params
        if iteration < 7 and n >= 3:
            distance = np.sum(residual**2 / (sigma**2 + shared**2), axis=1)
            weights = np.minimum(1, 6/(4+distance))
    information = 1/covariance[-1, -1] if factors is not None else 0.
    chi2 = float(np.sum(residual**2 / (sigma**2 + shared**2)))
    if samples is not None:
        draws = np.einsum("pna,dna->dp", influence, samples)
        covariance += np.cov(draws, rowvar=False)
    return WeightedMotion(params, covariance, residual, weights, influence, information, chi2)
