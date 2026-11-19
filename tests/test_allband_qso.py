import numpy as np
import pandas as pd

from deep_pm.allband_qso import evaluated_fit, physical_candidates


def test_all_filters_require_native_errors_valid_flags_and_signal():
    data=pd.DataFrame(dict(survey=['NSC']*7+['SDSS'],band=['u','VR','Y','custom','g','i','z','g'],
        sigma_east_mas=[20,20,20,20,0,20,20,20],sigma_north_mas=[20]*8,
        ra_deg=[359.999]*8,dec_deg=[70]*8,observation_time=[57000]*8,
        magnitude=[22]*8,magnitude_error=[.1,.1,.1,.1,.1,.1,.5,.1],quality_flags=[0,2,4,.5,0,8,0,0]))
    np.testing.assert_array_equal(physical_candidates(data),[True,True,False,False,False,False,False,False])


def test_fixed_source_reference_offset_changes_motion_and_common_error_not_sqrt_n():
    time=np.array([2005.,2005.,2015.,2015.]);sigma=np.full((4,2),50.)
    data=pd.DataFrame(dict(exposure_id=['SDSS:1','SDSS:2','NSC:1','NSC:2']))
    xy=np.column_stack([10+2*(time-2015.5),20-3*(time-2015.5)])
    samples=np.array([[-10.,0.],[10.,0.],[-10.,0.],[10.,0.]])
    old={name:(np.zeros(2),np.zeros_like(samples),100) for name in data.exposure_id}
    new={name:(np.array([10.,0.]) if name.startswith('SDSS') else np.zeros(2),
               samples if name.startswith('SDSS') else np.zeros_like(samples),100) for name in data.exposure_id}
    before=evaluated_fit(np.arange(4),data,xy,time,sigma,data.exposure_id.to_numpy(),old,[0,0],2015.5)
    after=evaluated_fit(np.arange(4),data,xy,time,sigma,data.exposure_id.to_numpy(),new,[0,0],2015.5)
    np.testing.assert_allclose(after['pm_icrs_masyr']-before['pm_icrs_masyr'],[1,0],atol=1e-8)
    np.testing.assert_allclose(after['reference_tangent_covariance'][1,1],np.var(samples[:,0],ddof=1)/100)
