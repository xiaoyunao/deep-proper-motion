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

class ErrorCoverageDiagnosticTest(unittest.TestCase):
    def test_source_split_changes_error_diagnostic_without_refitting_motion(self):
        from deep_pm.condition_spatial_pool import error_calibration
        random=np.random.default_rng(5)
        rows=[dict(core='a',qso_index=i,motion=random.normal(size=2)*2,error=np.array([4.,4.])) for i in range(600)]
        before=np.array([r['motion'].copy() for r in rows])
        result=error_calibration(rows)
        self.assertGreater(result['training_count'],50)
        self.assertGreater(result['test_count'],50)
        self.assertTrue(np.all((np.array(result['scale'])>.35)&(np.array(result['scale'])<.65)))
        np.testing.assert_array_equal(before,[r['motion'] for r in rows])
        check_aggregate(result)

    def test_sparse_split_does_not_claim_calibrated_errors(self):
        from deep_pm.condition_spatial_pool import error_calibration
        self.assertEqual(error_calibration([])['status'],'insufficient_support')
