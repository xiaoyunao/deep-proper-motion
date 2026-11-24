import unittest

import numpy as np

from deep_pm.allband_qso import evaluated_fit
from deep_pm.precision_v03 import fit_with_calibration
from deep_pm.standard_motion import calibrated_covariance, icrs_motion


class CatalogueFitReuseTest(unittest.TestCase):
    def test_full_covariance_preserves_old_solve_and_ordered_members(self):
        rng = np.random.default_rng(713)
        time = np.array([2004.1, 2008.3, 2013.4, 2016.4, 2017.6, 2018.2])
        sigma = rng.uniform(50, 120, size=(len(time), 2))
        xy = rng.normal(0, 30, size=(len(time), 2)) + [1200., -2030.]
        xy[-1,1] -= 800
        keys = np.array(['sdss_a', 'sdss_b', 'nsc_a', 'nsc_b', 'nsc_b', 'nsc_c'])
        common_draws = rng.normal(0, 4, size=(48, 2))
        offsets = {key: (rng.normal(0, 15, size=2), common_draws+rng.normal(0,3,size=(48,2)))
                   for key in sorted(set(keys))}
        ix = np.array([3, 4, 0, 5, 1, 2])
        corrections = [offsets[key] for key in keys[ix]]
        epoch, center = 2015.5, (30., -3.)
        model, _ = fit_with_calibration(xy[ix], time[ix], sigma[ix], epoch, corrections)
        covariance, independent, reference, _ = calibrated_covariance(model, time[ix], sigma[ix], corrections)
        legacy = icrs_motion(*model.coeff, center, covariance)
        trial = evaluated_fit(ix, None, xy, time, sigma, keys, offsets, center, epoch)
        cached = evaluated_fit(ix, None, xy, time, sigma, keys, offsets, center, epoch, model=model)
        for key in legacy:
            np.testing.assert_array_equal(trial[key], legacy[key])
        for key in trial:
            np.testing.assert_array_equal(trial[key], cached[key])
        np.testing.assert_array_equal(trial['robust_weight'], model.weight)
        np.testing.assert_array_equal(trial['independent_tangent_covariance'], independent)
        np.testing.assert_array_equal(trial['reference_tangent_covariance'], reference)
        self.assertLess(model.weight[3], 1.)
        self.assertGreater(abs(reference[0,3]), 0.)
