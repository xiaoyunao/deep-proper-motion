import unittest
import numpy as np
from deep_pm.continuous_conditions import experiment_indices, time_information, summaries
from deep_pm.public_summary import check_aggregate


class ConditionsTest(unittest.TestCase):
    def test_nested_indices_use_actual_earliest_and_preserve_nsc(self):
        t = np.array([2004., 2017., 2001., 2015., 2018.])
        nsc = np.array([False, True, False, True, True])
        ix = np.array([3, 0, 2, 1, 4])
        arms = experiment_indices(ix, nsc, t)
        np.testing.assert_array_equal(arms['NSC_only'], [3, 1, 4])
        np.testing.assert_array_equal(arms['one_SDSS'], [3, 1, 4, 2])
        np.testing.assert_array_equal(arms['all_SDSS'], ix)

    def test_same_night_count_does_not_imply_independent_groups(self):
        t = np.array([2000., 2010., 2010., 2010.])
        out = time_information(t, np.full((4, 2), 50.), np.ones(4), np.array(['a','b','b','b']))
        self.assertEqual(out['independent_groups'], 2)
        self.assertTrue(np.all(out['effective_groups'] <= 2))
        self.assertEqual(out['span_years'], 10)
        check_aggregate(out)

    def test_empty_cohorts_are_explicit_and_publishable(self):
        out = summaries([], 9137)
        self.assertEqual(out['ordinary__r21_22__common_releases']['n'], 0)
        check_aggregate(out)
