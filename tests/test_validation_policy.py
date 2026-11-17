import unittest

import numpy as np
import pandas as pd

from deep_pm.validation_audit import gaia_bright
from deep_pm.validation_blocks import connected
from deep_pm.exposure_inputs import tap_join


class ValidationPolicyTest(unittest.TestCase):
    def test_gaia_thresholds_are_strict_on_both_magnitudes_and_pm(self):
        table = pd.DataFrame(dict(phot_g_mean_mag=[19.,19.,20.,19.,19.,19.],
            visibility_periods_used=[9,9,9,8,9,9], pmra_error=[.1,.1,.1,.1,.5,.1],
            pmdec_error=[.1,.1,.1,.1,.1,.6]))
        values = gaia_bright(table,np.array([20.9,21.,20.,20.,20.,20.]))
        np.testing.assert_array_equal(values,[True,False,False,False,False,False])

    def test_shared_runs_and_exposures_form_transitive_blocks(self):
        sample = [dict(exposures=groups,nsc_epoch_days=[day]) for groups,day in [
            (["SDSS:1","NSC:1"],7),(["SDSS:1","NSC:2"],8),
            (["SDSS:2","NSC:2"],8),(["SDSS:3","NSC:3"],8)]]
        self.assertEqual(sorted(map(len,connected(sample))),[1,3])
        self.assertEqual(sorted(map(len,connected(sample,True))),[4])

    def test_tap_join_partitions_on_half_open_archive_coordinates(self):
        class Checked:
            def __init__(self):
                self.queries=[]

            def checked(self, service, columns, source, where, cap):
                self.queries.append((service,source,where,cap))
                return pd.DataFrame({"measid":[where]})

        cache=Checked()
        archive=pd.DataFrame(dict(ra=[350.1,351.0],dec=[0.1,0.2],ndet=[50000,50000]))
        result=tap_join(cache,archive,dict(ra=[350.,352.],dec=[0.,1.]),0.)
        self.assertEqual(len(result),2)
        self.assertTrue(any("o.ra < 351.000000000" in row[2] for row in cache.queries))
        self.assertTrue(any("o.ra >= 351.000000000" in row[2] for row in cache.queries))
        self.assertTrue(all(row[0]=="tap" and "m.objectid=o.id" in row[1] for row in cache.queries))


if __name__ == "__main__":
    unittest.main()
