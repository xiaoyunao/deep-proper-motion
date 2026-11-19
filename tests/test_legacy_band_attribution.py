import numpy as np
from deep_pm.astrometry import fit_motion
from deep_pm.legacy_band_attribution import conditional_contributions
from deep_pm.standard_motion import icrs_motion


def test_robust_attribution_by_family_closes_with_raw_and_calibration_parts():
    t=np.array([2001.,2014.,2015.,2018.]);sigma=np.full((4,2),50.)
    raw=np.array([3000.,-2000.])+np.column_stack([2*(t-2015.5),-3*(t-2015.5)])
    raw[2]+=[80,-30]
    shifts=np.array([[10,-20],[0,0],[0,0],[0,0.]])
    model=fit_motion(raw-shifts,t,sigma,2015.5)
    offsets=[(delta,np.zeros((48,2)),100) for delta in shifts]
    values,closure=conditional_contributions(model,t,sigma,offsets,raw,np.array([3000,-2000]),[359.99,70.],np.array(['SDSS:r','NSC:g','NSC:r','NSC:z']))
    expected=icrs_motion(model.coeff[0],model.coeff[1],[359.99,70.])['pm_icrs_masyr']
    np.testing.assert_allclose(sum(row['total'] for row in values.values()),expected,atol=1e-7)
    assert np.linalg.norm(values['SDSS:r']['C0_zero_point'])>0
    assert closure<1e-7
