import unittest

import numpy as np
import pandas as pd

from deep_pm.matched_control import matched_motion


class MatchedControlTest(unittest.TestCase):
    def test_secure_comparison_restricts_both_fits_to_the_same_sources(self):
        label = np.array([0, 0, 1, 1])
        required = {0: ({0}, {1}, {0}, [{0}, {1}]),
                    1: ({2}, {3}, {2}, [{2}, {3}])}
        gaia = pd.DataFrame(dict(pm_tangent_x=[4., 10.], pm_tangent_y=[5., 20.]))
        fixed = pd.DataFrame(dict(east=[2., 3.], north=[4., 5.]))
        tracks = {"fixed": [], "nsc_like": [], "motion": [
            dict(indices=[0, 1], status="secure"), dict(indices=[2, 3], status="ambiguous")]}
        fits = [dict(method="motion", in_core=True, track=i, pm_x=x, pm_y=y)
                for i, (x, y) in enumerate([(7, 7), (19, 31)])]
        summaries = matched_motion(fits, tracks, label, required, .8, gaia, fixed)
        self.assertEqual(summaries["motion/all_selected"]["blind"]["n"], 2)
        secure = summaries["motion/secure"]
        self.assertEqual(secure["fixed_all_visits"]["n"], 1)
        np.testing.assert_equal(secure["blind_minus_fixed"]["median_masyr"], [1., -2.])
