import unittest

import pandas as pd

from deep_pm.continuous_ownership import close_pairs


class OwnershipPairsTest(unittest.TestCase):
    def test_wrap_and_shared_core(self):
        data = pd.DataFrame(dict(ra=[359.999999, .000001, 359.999999], dec=[0., 0., 0.],
                                 pm_east=[1., 1., 20.], pm_north=[0., 0., 0.], core=["west", "east", "west"]))
        result = close_pairs(data)
        self.assertEqual(result["exact_position_duplicate_rows"], 2)
        for counts in result["cross_core_close_pairs_by_arcsec"].values():
            self.assertEqual(counts["cross_core_pairs"], 2)
            self.assertEqual(counts["motion_difference_below_1_masyr"], 1)

    def test_no_close_pair(self):
        data = pd.DataFrame(dict(ra=[350., 351.], dec=[0., 0.], pm_east=[0., 0.],
                                 pm_north=[0., 0.], core=["west", "east"]))
        result = close_pairs(data)
        self.assertEqual(result["exact_position_duplicate_rows"], 0)
        for counts in result["cross_core_close_pairs_by_arcsec"].values():
            self.assertEqual(counts["cross_core_pairs"], 0)
