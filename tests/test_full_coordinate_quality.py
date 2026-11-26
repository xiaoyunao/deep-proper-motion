import numpy as np
from deep_pm.full_coordinate_quality import reference_common,unique_positions


def test_reference_common_keeps_source_correlations_and_independent_graphs():
    rng=np.random.default_rng(21);d=rng.normal(size=(48,4));d-=d.mean(axis=0)
    def row(graph):return dict(reference_model_sha256=graph,synchronous_reference_icrs_state_draws=d.tolist(),values=dict(new_all=([1.,2.],[2.,2.])))
    first=reference_common([row('a') for _ in range(20)])
    expected=np.std(d[:,[1,3]],axis=0,ddof=1)
    np.testing.assert_allclose(first['equal_mean_common_SE'],expected)
    independent=reference_common([row('a') for _ in range(10)]+[row('b') for _ in range(10)])
    np.testing.assert_allclose(independent['equal_mean_common_SE'],expected/np.sqrt(2))


def test_micro_identity_tolerance_handles_wrap_without_science_radius_merging():
    rows=[dict(identity=['Q',359.999999999,0]),dict(identity=['Q',0.,0]),dict(identity=['Q',.00001,0]),dict(identity=['OTHER',0.,0])]
    out,n=unique_positions(rows)
    assert n==2 and len(out)==2


def test_correlated_motion_ellipse_is_not_diagonal_axis_scaling():
    import numpy as np
    from deep_pm.full_coordinate_quality import ellipse_coverage
    points=np.array([[1.4,1.4]]*5+[[-1.4,1.4]]*5)
    covariance=np.repeat(np.array([[[1.,.9],[.9,1.]]]),10,axis=0)
    out=ellipse_coverage(points,covariance)
    assert out['raw']['coverage']['0.68']['n']==5
    diagonal=ellipse_coverage(points,np.repeat(np.eye(2)[None],10,axis=0))
    assert diagonal['raw']['coverage']['0.68']['n']==0
