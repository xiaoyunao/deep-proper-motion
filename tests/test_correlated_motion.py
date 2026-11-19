import numpy as np
from deep_pm.astrometry import fit_motion
from deep_pm.correlated_motion import fit_correlated_motion


def test_zero_common_term_recovers_frozen_irls_and_covariance():
    t=np.array([2000,2008,2010,2015.]);sigma=np.array([[40,80],[60,50],[100,20],[30,70.]])
    x=np.column_stack([t-2000,-2*(t-2000)])+np.array([[2,-2],[6,0],[100,10],[-1,2]])
    offsets=[(np.zeros(2),np.zeros((48,2)),100)]*4
    a=fit_motion(x,t,sigma,2015.5)
    b=fit_correlated_motion(x,t,sigma,np.arange(4),np.zeros_like(sigma),2015.5,offsets)
    np.testing.assert_allclose(a.coeff,b.model.coeff,atol=1e-10)
    np.testing.assert_allclose(a.weight,b.model.weight,atol=1e-10)
    np.testing.assert_allclose(b.covariance[:2,:2],a.cov[0],atol=1e-10)


def test_repeated_same_night_does_not_average_away_shared_error():
    t=np.repeat([2000.,2010.],20);x=np.column_stack([2*t,3*t]);sigma=np.full((40,2),10.)
    offsets=[(np.zeros(2),np.zeros((48,2)),100)]*40
    a=fit_correlated_motion(x,t,sigma,t.astype(str),np.full((40,2),50.),2015.5,offsets,robust=False)
    expected=2*(50**2+10**2/20)/10**2
    np.testing.assert_allclose(np.diag(a.covariance)[[1,3]],expected,rtol=1e-10)
    np.testing.assert_allclose(a.covariance,a.independent_covariance+a.night_covariance+a.reference_covariance)
