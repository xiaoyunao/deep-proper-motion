import unittest

import pandas as pd
import numpy as np

from deep_pm.continuous_ownership import close_pairs
from deep_pm.public_summary import check_aggregate


class OwnershipPairsTest(unittest.TestCase):
    def test_empty_catalogue(self):
        data = pd.DataFrame({name: pd.Series(dtype=float) for name in ("ra", "dec", "pm_east", "pm_north")})
        data["core"] = pd.Series(dtype=str)
        result = close_pairs(data)
        check_aggregate(result)
        self.assertEqual(result["catalogue_count"], 0)
        self.assertEqual(result["cross_core_close_pairs_by_arcsec"]["0.2"]["cross_core_pairs"], 0)

    def test_wrap_and_shared_core(self):
        data = pd.DataFrame(dict(ra=[359.999999, .000001, 359.999999], dec=[0., 0., 0.],
                                 pm_east=[1., 1., 20.], pm_north=[0., 0., 0.], core=["west", "east", "west"]))
        result = close_pairs(data)
        check_aggregate(result)
        self.assertEqual(result["catalogue_count"], 3)
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

    def test_different_physical_planes_use_standard_icrs_velocity(self):
        from deep_pm.astrometry import project
        from deep_pm.standard_motion import icrs_motion
        sky=[188.,29.];velocity=[1000.,2000.];centers={'west':(183.,33.),'east':(193.,33.)}
        records=[]
        for core,center in centers.items():
            position=project([sky[0]],[sky[1]],center)[0]
            jac=icrs_motion(position,[0.,0.],center)['state_jacobian'][np.ix_([1,3],[1,3])]
            motion=np.linalg.solve(jac,velocity)
            records.append(dict(ra=sky[0],dec=sky[1],pm_east=motion[0],pm_north=motion[1],core=core))
        data=pd.DataFrame(records)
        result=close_pairs(data,centers)
        for counts in result['cross_core_close_pairs_by_arcsec'].values():
            self.assertEqual(counts['cross_core_pairs'],1)
            self.assertEqual(counts['motion_difference_below_1_masyr'],1)
