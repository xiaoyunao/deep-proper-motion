import unittest

import numpy as np

from deep_pm.regional_cohort import resampling
from deep_pm.regional_validation import first_exit


class RegionalDiagnosticTests(unittest.TestCase):
    def test_exit_traces_furthest_consistent_candidate(self):
        truth={0:np.array([0,1,2,3])}
        tracks=[np.array([0,1,2]),np.array([0,1,3,4])]
        survey=np.array([False,True,True,True,True])
        labels=np.array([0,0,0,0,1])
        support=np.ones(len(labels),dtype=bool)
        result=first_exit(0,[0,1],truth,labels,{0:"m3_support",1:"final"},tracks,survey,support,.6)
        self.assertEqual(result,"m3_support")

    def test_observation_blocks_resample_on_matched_object_pairs(self):
        group_a=[{"W0":([2,1],[1,1]),"published_nsc":([4,2],[1,1])} for _ in range(4)]
        group_b=[{"W0":([-1,2],[1,1]),"published_nsc":([-2,4],[1,1])} for _ in range(6)]
        settings=dict(seed=71,bootstrap_draws=100)
        self.assertIsNone(resampling([group_a],settings,"W0"))
        x=resampling([group_a,group_b],settings,"W0")
        np.testing.assert_allclose(x["p68_gain95"],[.5,.5])
        self.assertLess(x["zero95_masyr"][0,0],0)
        self.assertGreater(x["zero95_masyr"][0,1],0)
