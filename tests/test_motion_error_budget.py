import numpy as np
from deep_pm.astrometry import fit_motion
from deep_pm.motion_error_budget import allocate
from deep_pm.precision_v03 import fit_with_calibration


def test_terms_close_to_frozen_robust_covariance_with_shared_reference():
    time=np.array([2000.,2010.,2013.,2015.,2018.])
    native=np.array([[100.,120.],[30.,25.],[30.,25.],[30.,25.],[30.,25.]])
    sigma=np.hypot(native,50.)
    position=(time-2015.5)[:,None]*np.array([10.,-15.])
    position[2]+=np.array([700.,-200.])
    rng=np.random.default_rng(42)
    drift=rng.normal(size=(48,2))
    offsets=[(np.zeros(2),drift*(t-2015.5),100) for t in time]
    model,_=fit_with_calibration(position,time,sigma,2015.5,offsets)
    assert model.weight.min()<.5
    result=allocate(model,time,sigma,native,np.array([False,True,True,True,True]),offsets,(350.,0.))
    np.testing.assert_allclose(np.array(result['variance_fraction']).sum(axis=0),1,atol=1e-10)
    assert np.all(np.array(result['variance_fraction'])>=-1e-10)
    assert np.all(np.array(result['variance_masyr2'])[3]>0)


def test_two_epoch_no_floor_budget_assigns_expected_published_error_share():
    time=np.array([2000.,2010.]);sigma=np.array([[100.,100.],[10.,10.]])
    position=np.zeros((2,2));model=fit_motion(position,time,sigma,2015.5)
    offsets=[(np.zeros(2),np.zeros((48,2)),100)]*2
    result=allocate(model,time,sigma,sigma,np.array([False,True]),offsets,(350.,0.))
    np.testing.assert_allclose(result['variance_fraction'][0],[10000/10100]*2,atol=1e-10)
    np.testing.assert_allclose(result['variance_fraction'][2:],0,atol=1e-12)
