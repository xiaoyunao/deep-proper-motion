import unittest

import pandas as pd

from deep_pm.regional_preflight import CHIP_VERTICES
from deep_pm.regional_production import r_chip_shapes
from deep_pm.spatial_cohort import zero_status
from deep_pm.spatial_history import matched_conditions
from deep_pm.spatial_selection import choose


class SpatialValidationTest(unittest.TestCase):
    def test_small_or_precise_sample_with_significant_bias(self):
        self.assertFalse(zero_status(8, None, 20, 1, 3, .5)["adequate_support"])
        zero = {"zero95_masyr": [[.2, 1.1], [-.2, .2]]}
        sample = zero_status(210, zero, 20, 3, 3, .5)
        self.assertEqual(sample["significant_axes"], ["east"])
        self.assertFalse(sample["certified"])
        self.assertTrue(zero_status(210, {"zero95_masyr": [[-.4,.3],[-.1,.2]]},
                                    20, 3, 3, .5)["certified"])

    def test_same_conditions_and_color_have_separate_denominators(self):
        settings = dict(reference_r_edges=[20,21,22,23], seed=2,
                        spatial={"min_independent_blocks_for_high_precision":3})
        rows = [(21.2,"scan","cam","short","early","blue",[1,2])]*3
        rows += [(22.2,"scan","cam","short","early","red",[3,3])]*3
        rows += [(22.2,"another_scan","cam","short","early","blue",[-40,1])]*4
        values = matched_conditions({"a":rows},[["a"]],settings)
        case = values["observing_conditions"]["22_to_23_minus_21_to_22"]
        self.assertEqual(case["faint_and_bright_counts"].tolist(),[3,3])
        self.assertEqual(case["median_delta_masyr"].tolist(),[2,1])
        self.assertNotIn("additionally_color_matched",values)
        self.assertIsNone(case["block95"])

    def test_geometry_contains_wrapped_ra_and_handles_empty_table(self):
        columns = ["exposure","mjd",*CHIP_VERTICES]
        self.assertEqual(r_chip_shapes(pd.DataFrame(columns=columns,dtype=float),(0,0)),[])
        chips = pd.DataFrame([["x",57400,359.9,.1,.1,359.9,-.1,-.1,.1,.1]],columns=columns)
        [[epoch, polygons]] = r_chip_shapes(chips,(0,0))
        self.assertTrue(2016 < epoch < 2017)
        self.assertTrue(polygons[0].contains_point((0,0)))

    def test_batch_prefers_new_physical_scans(self):
        row = dict(joint_nsc_span_median=2.,repeated_sdss_fraction=.45,
                   nscr_exposures_by_instrument={"cam":3})
        spec = dict(role="north")
        candidates = {name: ("a",spec,row,{run}) for name,run in
                      [("a","12"),("b","12"),("c","23")]}
        first,second=choose(candidates,(2,1),1)
        self.assertIn("c",{r["name"] for r in first})
        self.assertEqual(len(second),1)


if __name__ == "__main__":
    unittest.main()
