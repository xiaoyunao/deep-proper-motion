import numpy as np
import pandas as pd
from deep_pm.full_coordinate_sources import information
from deep_pm.coordinate_metric import fit_vector_motion
from deep_pm.standard_motion import icrs_motion


def test_full2d_fisher_time_leverage_and_true_independent_years():
    t=np.array([2001.,2001.01,2014.,2018.,2018.001]);n=len(t)
    cp=np.tile([[3600.,1200.],[1200.,6000.]],(n,1,1))
    model=fit_vector_motion(np.zeros((n,2)),t,cp,2015.5)
    fit=icrs_motion(model.coeff[0],model.coeff[1],(0.,0.),model.cov)
    fit.update(robust_weight=model.weight,independent_tangent_covariance=model.cov,reference_tangent_covariance=np.zeros((4,4)))
    data=pd.DataFrame({'survey':['SDSS','SDSS','NSC','NSC','NSC'],'exposure_id':['SDSS:1','SDSS:2','NSC:1','NSC:2','NSC:3']})
    result=information(data,t,np.array([52000,52001,56000,57000,57000]),np.array(['a','b','c','d','e']),model,fit,cp,np.array([2001,2001,2014,2018,2018]))
    assert result['sdss_physical_runs']==2 and result['sdss_independent_UTC_years']==1
    assert result['nsc_independent_UTC_years']==2
    np.testing.assert_allclose(result['full2D_profiled_ICRS_slope_information'],np.diag(np.linalg.inv(model.cov[np.ix_([1,3],[1,3])])),rtol=1e-12)
    assert np.all(np.asarray(result['largest_information_fraction']['physical_group'])<=1)
