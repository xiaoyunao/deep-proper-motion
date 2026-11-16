import unittest
import numpy as np

from deep_pm.validation import matched_object, truth_requirements


class RecoveryDefinitionTest(unittest.TestCase):
    def test_many_nsc_visits_cannot_hide_a_wrong_early_sdss_connection(self):
        survey = np.array(["SDSS"] * 2 + ["NSC"] * 30)
        time = np.r_[2000., 2002., 2010 + np.arange(30) / 365.25]
        labels = np.zeros(len(time), dtype=int)
        req = truth_requirements({0: np.arange(len(time))}, survey, time, 100)
        self.assertIsNone(matched_object(np.arange(1, len(time)), labels, req, .8))
        self.assertEqual(matched_object(np.arange(len(time)), labels, req, .8), 0)
        labels[-1] = 991
        self.assertIsNone(matched_object(np.arange(len(time)), labels, req, .8))
