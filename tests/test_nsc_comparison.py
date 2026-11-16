import unittest

import numpy as np

from deep_pm.nsc_cohort import field_resample, ranking
from deep_pm.nsc_comparison import absolute68, comparison


SETTINGS = dict(seed=321, bootstrap_draws=100, min_publish_count=3, tail_masyr=5,
                min_candidates_with_sdss=8)


class SameSourcePrecisionTest(unittest.TestCase):
    def test_radial_quantile_uses_per_object_vector(self):
        np.testing.assert_allclose(absolute68(np.array([[3,4],[4,3]])), [3.68,3.68,5])

    def test_paired_ratio_of_scaled_motions(self):
        rows = [dict(M3=(np.array([j/2,j/2]), [1,1]),
                     published_nsc=(np.array([j,j]),[1,1])) for j in range(1,7)]
        result = comparison(rows, SETTINGS)
        np.testing.assert_allclose(result["paired"]["M3_vs_published_nsc"]["fraction95"], .5)
        block = field_resample([rows[:3],rows[3:]], SETTINGS)
        np.testing.assert_allclose(block["fraction95"], .5)
        self.assertEqual(comparison(rows[:2], SETTINGS), {"n":2})

    def test_ranking_excludes_low_coverage_and_prioritizes_faint(self):
        stats = {k:dict(candidates_with_both=n,faint_candidates_with_both=f)
                 for k,n,f in (("a",7,7),("b",15,2),("c",10,4))}
        self.assertEqual([k for k,v in ranking(stats, SETTINGS)],["c","b"])
