import unittest

import numpy as np
import pandas as pd

from deep_pm.astrometry import fit_motion, project, sdss_tai_to_tt, unproject
from deep_pm.normalize import sdss


class AstrometryTest(unittest.TestCase):
    def test_wrap_and_cosine(self):
        center = (0, 60)
        xy = project([359.999, 0.001], [60, 60], center)
        np.testing.assert_allclose(xy[:, 0], [-1800, 1800], rtol=1e-7)
        np.testing.assert_allclose(unproject(xy, center), [[359.999, 60], [0.001, 60]])

    def test_tai(self):
        self.assertAlmostEqual(sdss_tai_to_tt(53000 * 86400), 53000 + 32.184 / 86400, places=8)

    def test_fit_preserves_robust_shift(self):
        time = np.arange(2000, 2010)
        target = (time - 2005)[:, None] * np.array([[800, -200]])
        target = target.astype(float)
        target[-1] += 10000
        fit = fit_motion(target, time, np.full(target.shape, 25), 2005)
        np.testing.assert_allclose(fit.coeff[1], [800, -200], atol=1)
        self.assertLess(fit.weight[-1], 0.1)

    def test_same_run_dedup_and_no_second_cosine(self):
        table = pd.DataFrame(dict(
            objid=[11, 12, 13, 14], run=[1, 1, 2, 3], camcol=2, ra=1., dec=60.,
            raerr=.1, decerr=[.2, .2, .2, -9999], tai_r=53000 * 86400,
            resolvestatus=[257, 32, 1025, 1], offsetra_r=[0, 0, 0, .1],
            offsetdec_r=0., psfmag_r=19., psfmagerr_r=.01, type=6, flags=0,
            airmass_r=1.2, psffwhm_r=1.4, nchild=0,
        ))
        out = sdss(table)
        self.assertEqual(list(out.detection_id), ["SDSS:11", "SDSS:13", "SDSS:14"])
        np.testing.assert_array_equal(out.sigma_east_mas, [100, 100, 100])
        self.assertTrue(np.isnan(out.sigma_north_mas.iloc[-1]))
        self.assertTrue(np.isnan(out.observation_time.iloc[-1]))
