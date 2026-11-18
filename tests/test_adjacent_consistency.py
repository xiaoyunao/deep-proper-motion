import copy
import unittest
import numpy as np
from deep_pm.adjacent_consistency import difference_covariance
from deep_pm.standard_motion import icrs_motion

class AdjacentCovarianceTest(unittest.TestCase):
    def row(self):
        h=np.zeros((4,2,2));h[0,0,0]=1;h[1,1,0]=1;h[2,0,1]=1;h[3,1,1]=1
        return dict(icrs_motion([0.,0.],[10.,20.],(350.,0.)),projection_center_deg=[350.,0.],
                    detection_ids=['first','second'],measurement_sigma_mas=[[3.,4.],[5.,6.]],
                    response=h,reference_state_draws=np.random.default_rng(16).normal(size=(48,4)))

    def test_identical_physical_measurements_and_reference_draws_cancel(self):
        first=self.row()
        delta,ind,ref=difference_covariance(first,copy.deepcopy(first))
        np.testing.assert_allclose(delta,0,atol=1e-8)
        np.testing.assert_allclose(ind,0,atol=1e-20)
        np.testing.assert_allclose(ref,0,atol=1e-20)

    def test_shared_reference_does_not_make_disjoint_photon_errors_shared(self):
        first=self.row();second=copy.deepcopy(first)
        second['detection_ids']=['third','fourth']
        _,ind,ref=difference_covariance(first,second)
        np.testing.assert_allclose(np.diag(ind),[18.,50.,32.,72.],atol=1e-8)
        np.testing.assert_allclose(ref,0,atol=1e-20)

    def test_inconsistent_error_for_same_measurement_is_rejected(self):
        first=self.row();second=copy.deepcopy(first)
        second['measurement_sigma_mas'][0][0]=30.
        with self.assertRaises(ValueError):
            difference_covariance(first,second)
