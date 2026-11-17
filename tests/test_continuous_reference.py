import unittest

import numpy as np
import pandas as pd

from deep_pm.continuous_reference import deduplicate
from deep_pm.continuous_summary import membership


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
