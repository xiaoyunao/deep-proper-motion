import numpy as np
from deep_pm.coordinate_metric import fit_vector_motion,covariance_of_estimator
from deep_pm.faint_covariance_v23 import components


def test_survey_covariance_preserves_cross_axis_and_saved_robust_weights():
    t=np.array([2001.,2005.,2013.,2017.,2018.]);xy=np.array([3.,-5.])[None,:]*(t-2015.5)[:,None];xy[1]+=[70.,-30.]
    c=np.array([[[3000.+300*i,700.],[700.,6000.-400*i]] for i in range(len(t))])
    fit=fit_vector_motion(xy,t,c,2015.5);j=np.eye(4);j[1,3]=.4;j[3,1]=-.2
    tags=np.array(['SDSS','SDSS','NSC','NSC','NSC']);split,per=components(fit.influence,c,fit.weight,tags,j)
    expected=j@covariance_of_estimator(fit.influence,c,fit.weight)@j.T
    np.testing.assert_allclose(sum(split.values()),expected[np.ix_([1,3],[1,3])],rtol=1e-12)
    assert np.linalg.eigvalsh(per).min()>-1e-10
    assert abs(split['SDSS'][0,1])>1e-3
    perm=np.array([4,2,0,3,1]);other,_=components(fit.influence[:,perm],c[perm],fit.weight[perm],tags[perm],j)
    for key in split:np.testing.assert_allclose(split[key],other[key],rtol=1e-12)
