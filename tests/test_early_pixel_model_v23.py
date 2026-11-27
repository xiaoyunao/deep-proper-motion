import numpy as np
from deep_pm.early_pixel_model_v23 import flux_profile,JointPixels,Cut,sky_at

class Gaussian:
    def sample(self,x,y,gx,gy):return np.exp(-((gx-x)**2+(gy-y)**2)/(2*1.3**2))/(2*np.pi*1.3**2)


def test_flux_and_background_profile_is_invariant_to_sky_offset():
    y,x=np.indices((11,11));p=Gaussian().sample(5.2,4.7,x.ravel(),y.ravel());sigma=np.full(len(p),.1)
    r,f,b,s=flux_profile(80*p+3,sigma,p)
    np.testing.assert_allclose([f,b],[80,3],rtol=1e-12);np.testing.assert_allclose(r,0,atol=1e-11)
    assert flux_profile(-80*p+3,sigma,p)[1]==0


def test_joint_free_motion_does_not_duplicate_late_information_and_recovers_common_shift():
    rng=np.random.default_rng(231);y,x=np.indices((13,13));x=x.ravel();y=y.ravel();state=np.array([30.,-4.,-20.,2.]);cuts=[]
    for run,t in enumerate([2001.,2005.]):
        base=np.array([6.,6.]);plate=np.eye(2)/396;true=base+plate@sky_at(state,t,2015.5);pixels=200*Gaussian().sample(*true,x,y)+1
        cuts.append(Cut(pixels,np.full(len(x),.1),x,y,base,plate,np.zeros(2),t,Gaussian(),str(run),run))
    times=np.array([2013.,2017.,2019.]);xy=sky_at(state,times,2015.5);c=np.tile(np.eye(2)*400.,(3,1,1));model=JointPixels(cuts,xy,times,c);fit=model.fit([0,0,0,0],starts=False)
    np.testing.assert_allclose(fit['state'],state,atol=1e-3);assert fit['conditional_covariance'] is not None
    draws=rng.normal(0,1,(48,2));early=np.repeat(draws[:,None,:],2,axis=1);late=np.repeat(draws[:,None,:],3,axis=1)
    response=model.common_response(fit,early,late)
    # A shared coordinate translation moves the state position, not PM.
    np.testing.assert_allclose(response[:,[0,2]],-draws,atol=.03);np.testing.assert_allclose(response[:,[1,3]],0,atol=.005)
    profile=model.profiles(fit,axes=(1,),levels=(1.,));lo,hi=profile['1']['1.0'];assert lo<state[1]<hi
