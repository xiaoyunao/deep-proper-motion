import unittest
import numpy as np
from deep_pm.condition_residuals import spatial_mean
from deep_pm.public_summary import check_aggregate

class SpatialCovarianceTest(unittest.TestCase):
    def test_shared_reference_error_does_not_shrink_as_independent_sources(self):
        draw=np.random.default_rng(5).normal(size=(48,2))*3
        record=dict(motion=[1.,-2.],independent_pm_covariance=np.diag([4.,9.]),reference_pm_draws=draw)
        one=spatial_mean([record])
        many=spatial_mean([record]*20)
        np.testing.assert_allclose(many['weighted_mean_masyr'],[1.,-2.])
        np.testing.assert_allclose(many['independent_mean_covariance'],np.diag([4.,9.])/20)
        np.testing.assert_allclose(many['reference_mean_covariance'],one['reference_mean_covariance'])
        check_aggregate(many)

    def test_empty_spatial_cells_have_no_invented_uncertainty(self):
        self.assertEqual(spatial_mean([]),dict(n=0))
