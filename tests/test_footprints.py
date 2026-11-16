import unittest
import numpy as np

from deep_pm.footprints import Visit, sdss_great_circle
from deep_pm.injection import injection_pass, simulation


class FootprintsTest(unittest.TestCase):
    def test_injection_recovery_requires_both_surveys_and_earliest_sdss(self):
        sdss, nsc = set(range(10)), set(range(10, 110))
        requirements = (sdss, nsc, {0}, [sdss, nsc])
        self.assertFalse(injection_pass(nsc, requirements, .8))
        self.assertFalse(injection_pass((sdss - {0}) | nsc, requirements, .8))
        self.assertTrue(injection_pass(sdss | nsc, requirements, .8))

    def test_sdss_scan_rotation_and_ra_wrap(self):
        mu, nu = sdss_great_circle(np.array([359.9, .1, 95.]), np.zeros(3), 95., 0)
        np.testing.assert_allclose(mu % 360, [359.9, .1, 95.])
        np.testing.assert_allclose(nu, 0)

    def test_coverage_and_detection_probability_are_distinct(self):
        visits = [Visit(name, "NSC", 2017., "r", lambda *args, covered=covered: covered)
                  for name, covered in [("outside", False), ("inside", True)]]
        seen, drawn = simulation(visits, (0., 0.), dict(reference_epoch=2015.,
                                  injection_detection_probability={"NSC": 0.}), np.zeros(2),
                                  np.ones(2), 19., {}, np.random.default_rng(1))
        self.assertEqual([v.exposure for v in seen], ["inside"])
        self.assertEqual(drawn, [])
