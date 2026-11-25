import numpy as np
from deep_pm.cluster_motion import mean,support_predictions


def test_qso_support_transfer_uses_geometry_and_independent_training_only():
    rows=[dict(response={'group':np.eye(2)},row=dict(pm_icrs_masyr=[1.,-2.])) for _ in range(8)]
    rows.append(dict(response={'group':np.eye(2)},row=dict(pm_icrs_masyr=[-5.,-.8])))
    rows.append(dict(response={'other':np.eye(2)},row=dict(pm_icrs_masyr=[-5.,-.8])))
    original=support_predictions(rows,list(range(8)))
    np.testing.assert_array_equal(original[8]['predicted_zero'],[1.,-2.])
    assert 9 not in original
    rows[8]['row']['pm_icrs_masyr']=[100000.,-100000.]
    np.testing.assert_array_equal(support_predictions(rows,list(range(8)))[8]['predicted_zero'],original[8]['predicted_zero'])


def test_reference_draws_remain_common_in_population_mean():
    draws=np.random.default_rng(30).normal(size=(48,2))*3
    values=[dict(row=dict(pm_icrs_masyr=[1.,2.],pm_error_masyr=[3.,3.]),
        independent_icrs=np.eye(2),graph_sha256='same',reference_draws=draws) for _ in range(4)]
    result=mean(values)
    np.testing.assert_array_equal(result['weighted_mean'],[1.,2.])
    np.testing.assert_allclose(result['independent_covariance'],np.eye(2)/4)
    np.testing.assert_allclose(result['synchronous_reference_covariance'],np.cov(draws,rowvar=False))
