from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from deep_pm.astrometry import project
from deep_pm.access import Cache, PartitionTooLarge
from deep_pm.validation_audit import anchored_bright_gaia, gaia_bright, leave_one_median
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

    def test_static_bright_anchors_ignore_faint_gaia_motion(self):
        center=(350.,0.)
        table=pd.DataFrame(dict(ra=[350.,350.01],dec=[0.,0.],phot_g_mean_mag=[19.,19.],
            visibility_periods_used=[12,12],pmra_error=[.1,.1],pmdec_error=[.1,.1],pmra=[0.,1e5]))
        xy=project(table.ra,table.dec,center)
        matched,counts=anchored_bright_gaia(table,[[0],[1]],xy,np.array([20.,21.]),center)
        self.assertEqual(list(matched.index),[0])
        self.assertEqual(counts["excluded_faint_anchors"],1)

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

    def test_native_window_accepts_a_complete_result_and_reuses_it(self):
        payload=b"measid,verified_partition_count\n1,2\n2,2\n"
        with TemporaryDirectory() as tmp, patch("deep_pm.access.request") as request:
            request.return_value=(pd.read_csv(BytesIO(payload)),payload)
            cache=Cache(Path(tmp))
            for _ in range(2):
                frame=cache.checked_native_window("measid","public_table","x>0",cap=10)
                self.assertEqual(frame.to_dict("list"),{"measid":[1,2]})
            request.assert_called_once()
            self.assertIn("LIMIT 11",request.call_args.args[1])

    def test_native_window_verifies_empty_and_truncated_answers_independently(self):
        for payload, count, result in (
                (b"measid,verified_partition_count\n",0,b"measid\n"),
                (b"measid,verified_partition_count\n1,2\n",2,b"measid\n1\n2\n")):
            responses=(payload,f"n\n{count}\n".encode(),result)
            with self.subTest(count=count), TemporaryDirectory() as tmp, patch("deep_pm.access.request") as request:
                request.side_effect=[(pd.read_csv(BytesIO(v)),v) for v in responses]
                cache=Cache(Path(tmp))
                frame=cache.checked_native_window("measid","public_table","x>0",cap=10)
                self.assertEqual(len(frame),count)
                self.assertEqual(request.call_count,3)
                self.assertEqual(len(list(Path(tmp).glob("*.json"))),2)

    def test_native_window_bisects_an_over_cap_answer(self):
        payload=b"measid,verified_partition_count\n1,12\n"
        with TemporaryDirectory() as tmp, patch("deep_pm.access.request") as request:
            request.return_value=(pd.read_csv(BytesIO(payload)),payload)
            with self.assertRaises(PartitionTooLarge):
                Cache(Path(tmp)).checked_native_window("measid","public_table","x>0",cap=10)
            request.assert_called_once()

    def test_residual_reference_excludes_the_exposure_being_tested(self):
        v=np.array([[0.,1.],[2.,4.],[6.,2.],[10.,8.],[9.,9.]])
        result=leave_one_median(v,np.array([0,1,0,0,1]))
        np.testing.assert_array_equal(result,[[8.,5.],[9.,9.],[5.,4.5],[3.,1.5],[2.,4.]])


if __name__ == "__main__":
    unittest.main()
