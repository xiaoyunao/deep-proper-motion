import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from deep_pm.continuous_reference import deduplicate
from deep_pm.continuous_instruments import instrument_bins, instrument_by_exposure
from deep_pm.continuous_summary import estimate, membership
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

    def test_paired_gain_uses_the_same_visit_weights_for_all_methods(self):
        rows=[
            dict(exposures=[f"SDSS:{i%3}",f"NSC:{i%4}"],nsc_epoch_days=[i%4],
                 motion={"C0":[i+1.,0.],"Xiao":[2.*(i+1),0.]})
            for i in range(12)
        ]
        result=estimate(rows,("Xiao","C0"),9137)["paired_radial_gain"]["Xiao"]
        self.assertAlmostEqual(result["relative_gain"],.5)
        for values in result["visit_weighted"].values():
            self.assertEqual(values["n"],12)
            self.assertEqual(values["valid_draws"],800)
            np.testing.assert_allclose(values["relative_gain95"],[.5,.5])

    def test_instrument_cohort_requires_metadata_for_every_used_nsc_visit(self):
        with TemporaryDirectory() as directory:
            chips=Path(directory)/"chips.csv"
            chips.write_text("exposure,instrument\n0007,DECam\n08,90Prime\n")
            metadata=instrument_by_exposure([chips])
        motion={m:[0.,0.] for m in ("NSC","Xiao","C0")}
        records=[dict(magnitude=22.1,motion=motion,exposures=visits) for visits in
                 (["SDSS:1","NSC:0007"],["NSC:0007","NSC:08"],["NSC:0007","NSC:09"])]
        groups=instrument_bins(records,metadata)["22_to_23"]
        self.assertEqual({key:len(rows) for key, rows in groups.items()},
                         {"DECam":1,"90Prime_and_DECam":1,"incomplete":1})
