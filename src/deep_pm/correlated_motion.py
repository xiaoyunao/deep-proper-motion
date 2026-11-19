"""Experimental four-parameter IRLS with explicit additional night covariance.

The native marginal-error decomposition is empirical, not a claim that published
catalogue errors are pure centroids. Reference-bootstrap covariance stays a
separate synchronous term. Error-parameter uncertainty is not included here.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor,cho_solve

from .astrometry import Motion


@dataclass
class CorrelatedMotion:
    model: Motion
    covariance: np.ndarray
    independent_covariance: np.ndarray
    night_covariance: np.ndarray
    reference_covariance: np.ndarray
    response: np.ndarray


def fit_correlated_motion(position,time,independent_sigma,night,night_sigma,epoch,offsets,robust=True):
    time=np.asarray(time,float);position=np.asarray(position,float);sigma=np.asarray(independent_sigma,float)
    night=np.asarray(night);extra=np.asarray(night_sigma,float)
    if len(time)<2 or np.ptp(time)<=0 or position.shape!=sigma.shape or sigma.shape!=extra.shape or sigma.shape!=(len(time),2):
        raise ValueError('Need matching two-axis positions/errors and distinct times')
    if not np.isfinite(position).all() or not np.isfinite(sigma).all() or (sigma<=0).any() or not np.isfinite(extra).all() or (extra<0).any():
        raise ValueError('Need finite positions, positive independent and nonnegative common errors')
    # Night labels include survey/instrument so unrelated observatories do not share a term.
    design=np.column_stack([np.ones(len(time)),time-epoch]);weight=np.ones(len(time))
    response=np.zeros((4,len(time),2));coeff=np.empty((2,2));cov=np.zeros((4,4))
    night_blocks=[(night[:,None]==night[None,:])*np.outer(extra[:,a],extra[:,a]) for a in range(2)]
    for iteration in range(8 if robust and len(time)>=3 else 1):
        for axis in range(2):
            matrix=np.diag(sigma[:,axis]**2/weight)+night_blocks[axis]
            factor=cho_factor(matrix,lower=True,check_finite=False)
            inv_design=cho_solve(factor,design,check_finite=False)
            inv_normal=np.linalg.inv(design.T@inv_design)
            influence=inv_normal@inv_design.T
            coeff[:,axis]=influence@position[:,axis]
            sl=slice(2*axis,2*axis+2);cov[sl,sl]=inv_normal;response[sl,:,axis]=influence
        residual=position-design@coeff
        distance=(residual**2/(sigma**2+extra**2)).sum(axis=1)
        if iteration<7 and robust and len(time)>=3:weight=np.minimum(1,6/(4+distance))
    independent=np.zeros((4,4));common_night=np.zeros((4,4))
    for axis in range(2):
        sl=slice(2*axis,2*axis+2);influence=response[sl,:,axis]
        independent[sl,sl]=(influence*(sigma[:,axis]**2/weight))@influence.T
        common_night[sl,sl]=influence@night_blocks[axis]@influence.T
    draws=np.stack([o[1] for o in offsets],axis=1)
    propagated=np.einsum('pna,dna->dp',response,draws)
    common_reference=np.cov(propagated,rowvar=False)
    model=Motion(coeff,np.array([cov[:2,:2],cov[2:,2:]]),epoch,residual,weight,float(distance.sum()),float(np.ptp(time)))
    return CorrelatedMotion(model,independent+common_night+common_reference,independent,common_night,common_reference,response)
