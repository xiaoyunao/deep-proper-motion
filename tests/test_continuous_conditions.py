import unittest
import numpy as np
from deep_pm.continuous_conditions import experiment_indices, time_information, summaries
from deep_pm.public_summary import check_aggregate
from deep_pm.continuous_condition_pool import reproduce


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

    def test_freeze_comparison_rejects_changed_support_and_motion(self):
        records = [dict(magnitude=21.5,exposures=['a','b'],motion={'all_SDSS':[1.,2.]})]
        frozen = [dict(magnitude=21.5,exposures=['a','b'],motion={'C0':[1.,2.]})]
        self.assertEqual(reproduce(records,frozen)['maximum_absolute_motion_difference_masyr'],0.)
        frozen[0]['motion']['C0'] = [1.001,2.]
        with self.assertRaises(ValueError):
            reproduce(records,frozen)
        frozen[0]['exposures'] = ['a','c']
        with self.assertRaises(ValueError):
            reproduce(records,frozen)
