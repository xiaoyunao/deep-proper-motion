import unittest

import numpy as np
import pandas as pd

from deep_pm.matched_control import conflict_review, matched_motion


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

    def test_conflict_cost_compares_with_all_blockers(self):
        labels = np.array([0, 0, 1, -1])
        required = {0: ({0}, {1}, {0}, [{0}, {1}])}
        candidates = [dict(indices=[0, 1], score=8), dict(indices=[0, 2], score=14),
                      dict(indices=[1, 3], score=6)]
        components = [dict(candidates=[0, 1, 2], selected=[1, 2], state="optimal")]
        result = conflict_review([], {}, candidates, components, labels, required, .8)
        self.assertEqual(result["missed_with_optimal_candidate"], 1)
        np.testing.assert_equal(result["replacement_delta_quantiles"], [-12, -12, -12])
        self.assertEqual(result["conflict_counts"]["multiple_blocking_tracks"], 1)
        self.assertEqual(result["conflict_counts"]["blocker_contains_other_gaia_source"], 1)
