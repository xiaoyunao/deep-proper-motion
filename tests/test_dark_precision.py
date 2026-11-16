import unittest

import numpy as np
from scipy.spatial import cKDTree

from deep_pm.dark_precision import bin_name, query_unique


class DarkPrecisionTest(unittest.TestCase):
    def test_holdout_match_rejects_ambiguity_and_shared_detections(self):
        tree = cKDTree([[0, 0], [1000, 0], [1200, 0], [3000, 0]])
        good, closest = query_unique(tree, np.array([[0, 20], [0, -20], [1100, 0],
                                                      [2950, 0], [3600, 0]]), 400, 300)
        assert good.tolist() == [False, False, False, True, False]
        assert closest[3] == 3


    def test_reference_r_uses_configured_edges(self):
        edges = [20, 21, 22, 23]
        assert [bin_name(mag, edges) for mag in (19.9, 20, 21, 22, 23)] == [
            "lt20", "20_to_21", "21_to_22", "22_to_23", "ge23"]
