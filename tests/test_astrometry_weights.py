import unittest

import numpy as np

from deep_pm.astrometry import apparent_positions, fit_motion
from deep_pm.astrometry_weights import correlated_fit, parallax_factors, variance_model


class AstrometryWeightTests(unittest.TestCase):
    def setUp(self):
        self.t = np.array([0., 1., 2., 5., 9., 9.01])
        self.xy = np.column_stack([2 + 4*self.t, 3 - 6*self.t])
        self.sigma = np.full_like(self.xy, 12.)
        self.group = np.array(["a","b","c","d","e","e"])

    def test_independent_robust_model_matches_baseline(self):
        xy = self.xy.copy()
        xy[2] += 143.
        old = fit_motion(xy,self.t,self.sigma,3.)
        new = correlated_fit(xy,self.t,self.sigma,self.group,np.zeros_like(xy),3.)
        np.testing.assert_allclose(new.params,old.coeff.T.ravel())
        np.testing.assert_allclose(new.weights,old.weight)
        np.testing.assert_allclose(new.error,np.sqrt(old.cov[:,1,1]))

    def test_correlated_covariance_matches_dense_gls(self):
        shared = np.full_like(self.xy, 20.)
        model = correlated_fit(self.xy,self.t,self.sigma,self.group,shared,3.)
        design = np.column_stack([np.ones(len(self.t)),self.t-3.])
        counts = (self.group[:,None]==self.group).astype(float)
        covariance = np.diag(self.sigma[:,0]**2)+counts*20.**2
        expected = np.linalg.inv(design.T @ np.linalg.solve(covariance,design))
        np.testing.assert_allclose(model.covariance[:2,:2],expected)
        np.testing.assert_allclose(model.motion,[4,-6])

    def test_parallax_incidence_and_fit(self):
        phase = self.t*1.7
        observer = np.column_stack([np.cos(phase),np.sin(phase),np.sin(phase*2)*.4])
        sky = 359.99,38.
        factors = parallax_factors(observer,sky,np.zeros(2))
        exact = apparent_positions(*sky,0,0,10.,0,self.t,observer,sky)
        np.testing.assert_allclose(10*factors, exact, atol=1e-6)
        model = correlated_fit(self.xy+14*factors,self.t,self.sigma,self.group,
                               np.zeros_like(self.xy),3.,factors=factors)
        np.testing.assert_allclose(model.motion,[4,-6])
        self.assertAlmostEqual(model.params[-1],14.)
        self.assertGreater(model.parallax_information,0)

    def test_sparse_error_group_keeps_baseline(self):
        params,trained = variance_model(np.arange(5.),np.arange(5.),np.arange(5.),20)
        self.assertFalse(trained)
        np.testing.assert_array_equal(params,[1.,50.])
