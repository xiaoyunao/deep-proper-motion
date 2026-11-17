import unittest

import numpy as np
import pandas as pd

from deep_pm.continuous_reference import deduplicate
from deep_pm.continuous_summary import membership
from deep_pm.validation_audit import boundary_pairs


class ContinuousReferenceTest(unittest.TestCase):
    def test_common_reference_deduplicates_and_drops_cross_shard_identity_conflicts(self):
        rows = pd.DataFrame([
            ("a", "SDSS-1", "SDSS:5", False),
            ("a", "NSC-1", "NSC:4", False),
            ("b", "X", "SDSS:5", True)],
            columns=["reference","detection_id","exposure_id","is_holdout"])
        overlap = rows.iloc[[1,2]].copy()
        overlap.loc[2,"reference"]="c"
        result, stats=deduplicate([rows,overlap])
        self.assertEqual(list(result.detection_id),["SDSS-1","NSC-1"])
        self.assertEqual(stats["conflicting_match_rows"],2)
        self.assertEqual(stats["unique_rows"],2)

    def test_membership_shares_weights_for_reused_physical_groups(self):
        rows=[dict(exposures=["SDSS:1","NSC:7"],nsc_epoch_days=[1]),
              dict(exposures=["SDSS:1","SDSS:2","NSC:7"],nsc_epoch_days=[1])]
        np.testing.assert_equal(membership(rows,"sdss_run").toarray(),[[1,0],[.5,.5]])
        np.testing.assert_equal(membership(rows,"nsc_exposure").toarray(),[[1],[1]])

    def test_same_exposure_detector_overlap_supports_c0_alone(self):
        data = pd.DataFrame({"survey": ["NSC", "NSC"], "exposure_id": ["NSC:1", "NSC:1"],
                             "detector_id": [1, 2]})
        refs = pd.DataFrame({"reference": ["a"], "is_holdout": [True]})
        xy = np.array([[0., 0.], [3., 4.]])
        keys = np.array(["NSC:1", "NSC:1"])
        c0 = {"NSC:1": (np.array([17., 23.]),)}
        result = boundary_pairs(refs, data, xy, pd.Series(["a", "a"]), (keys,), (c0,),
                                (350., 0.), {}, include_geometric=False)
        overlap = result["nsc_same_exposure_ccd"]
        self.assertEqual((overlap["pair_count"], overlap["held_references"], overlap["held_exposures"]),
                         (1, 1, 1))
        np.testing.assert_allclose(overlap["radial_p68_mas"], [5., 5.])
