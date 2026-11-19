import numpy as np
from deep_pm.correlated_motion import fit_correlated_motion
from deep_pm.night_error_year import prediction_variance
from deep_pm.astrometry import fit_motion
from deep_pm.condition_holdout import held_prediction_variance


def test_shared_night_cancels_in_prediction_and_separate_night_does_not():
    t=np.array([2015.,2015.1]);p=np.zeros((2,2));sigma=np.ones((2,2));extra=np.full((2,2),50.);labels=np.array(['night','night']);offsets=[(np.zeros(2),np.zeros((8,2)),20)]*2
    fit=fit_correlated_motion(p,t,sigma,labels,extra,2015.,offsets,robust=False)
    shared=prediction_variance(fit,t,offsets,offsets[0],2015.05,np.ones(2),labels,'night',extra,np.full(2,50.))
    separate=prediction_variance(fit,t,offsets,offsets[0],2015.05,np.ones(2),labels,'other',extra,np.full(2,50.))
    np.testing.assert_allclose(shared,[1.5,1.5],atol=1e-7)
    np.testing.assert_allclose(separate,[5001.5,5001.5],atol=1e-7)
    plain=fit_motion(p,t,sigma,2015.,robust=False)
    base=held_prediction_variance(plain,t,sigma,offsets,offsets[0],np.ones(2),2015.05)
    none=fit_correlated_motion(p,t,sigma,labels,np.zeros_like(extra),2015.,offsets,robust=False)
    np.testing.assert_allclose(prediction_variance(none,t,offsets,offsets[0],2015.05,np.ones(2),labels,'night',np.zeros_like(extra),np.zeros(2)),base,atol=1e-10)
