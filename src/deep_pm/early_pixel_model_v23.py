"""Finite-frame variable-projection position/motion likelihood, not a pipeline.

Flux/background are profiled per frame. The astrometric state is free; one
50mas position nuisance per independent early run implements the frozen floor.
Late positions appear once. Common reference/registration perturbations are
propagated jointly, separately from independent measurement information.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import map_coordinates,spline_filter
from scipy.optimize import least_squares,brentq
from astropy.io import fits

class SpatialPSF:
    def __init__(self,file):
        with fits.open(file,memmap=False) as h:
            t=h[3].data
            self.c=np.array(t['c'],float);self.nrow=np.array(t['nrow_b'],int);self.ncol=np.array(t['ncol_b'],int)
            shape=(int(t['rnrow'][0]),int(t['rncol'][0]));self.images=np.array([np.asarray(r,float).reshape(shape) for r in t['rrows']])
        self.splines=np.array([spline_filter(r,order=3) for r in self.images]);self.sums=self.images.sum(axis=(1,2));self.center=(np.array(shape)-1)/2

    def coefficients(self,x,y):
        # FITS TDIM reverses IDL axes: numpy c[eigen,row_power,col_power].
        return np.array([sum(self.c[j,i,k]*(y*.001)**i*(x*.001)**k for i in range(self.nrow[j]) for k in range(self.ncol[j])) for j in range(len(self.c))])

    def image(self,x,y):
        c=self.coefficients(x,y);den=c@self.sums
        if not np.isfinite(den) or den<=0:raise ValueError('Invalid spatial PSF normalization')
        return np.einsum('j,jxy->xy',c,self.images)/den

    def sample(self,x,y,gx,gy):
        c=self.coefficients(x,y);den=c@self.sums
        if not np.isfinite(den) or den<=0:raise ValueError('Invalid spatial PSF normalization')
        coords=np.array([np.asarray(gy)-y+self.center[0],np.asarray(gx)-x+self.center[1]])
        return sum(a*map_coordinates(b,coords,order=3,mode='constant',cval=0,prefilter=False) for a,b in zip(c,self.splines))/den


def flux_profile(data,sigma,psf):
    """Positive point flux and an unconstrained local background; no flux prior."""
    w=1/np.asarray(sigma)**2;y=np.asarray(data);p=np.asarray(psf)
    normal=np.array([[np.sum(w*p*p),np.sum(w*p)],[np.sum(w*p),np.sum(w)]])
    rhs=np.array([np.sum(w*p*y),np.sum(w*y)]);answer=np.linalg.solve(normal,rhs)
    if answer[0]<0:answer=np.array([0.,rhs[1]/normal[1,1]])
    residual=(y-answer[0]*p-answer[1])/sigma
    return residual,float(answer[0]),float(answer[1]),float(np.sqrt(np.linalg.inv(normal)[0,0]))

@dataclass
class Cut:
    data:np.ndarray
    sigma:np.ndarray
    gx:np.ndarray
    gy:np.ndarray
    base_pixel:np.ndarray
    plate:np.ndarray
    offset_mas:np.ndarray
    time:float
    psf:object
    group:str
    run:int
    reference_draws:object=None
    registration_draws:object=None
    floor:float=50.

    def point(self,sky):return self.base_pixel+self.plate@(np.asarray(sky)+self.offset_mas)
    def profile(self,sky):
        x,y=self.point(sky);return flux_profile(self.data,self.sigma,self.psf.sample(x,y,self.gx,self.gy))


def sky_at(state,time,epoch):
    return np.asarray(state)[[0,2]]+(np.asarray(time)-epoch)[...,None]*np.asarray(state)[[1,3]]


def numerical_jacobian(fun,x,steps):
    x=np.asarray(x);base=fun(x);j=np.empty((len(base),len(x)))
    for k,step in enumerate(steps):
        a=x.copy();b=x.copy();a[k]+=step;b[k]-=step;j[:,k]=(fun(a)-fun(b))/(2*step)
    return j


class JointPixels:
    def __init__(self,cuts,late_xy,late_time,late_covariance,epoch=2015.5):
        self.cuts=cuts;self.xy=np.asarray(late_xy,float).reshape(-1,2);self.time=np.asarray(late_time,float);self.cov=np.asarray(late_covariance,float).reshape(-1,2,2);self.epoch=epoch
        self.whiten=np.linalg.inv(np.linalg.cholesky(self.cov)) if len(self.xy) else np.empty((0,2,2))
        self.steps=np.r_[np.array([.5,.02,.5,.02]),np.full(2*len(cuts),.5)]
        self.scale=np.r_[np.array([100.,10.,100.,10.]),np.full(2*len(cuts),50.)]

    def residual(self,params,early_bias=None,late_bias=None):
        out=[];state=params[:4]
        if len(self.xy):
            delta=sky_at(state,self.time,self.epoch)-self.xy
            if late_bias is not None:delta=delta+late_bias
            out.append(np.einsum('nij,nj->ni',self.whiten,delta).ravel())
        for i,cut in enumerate(self.cuts):
            jitter=params[4+2*i:6+2*i];position=sky_at(state,cut.time,self.epoch)+jitter
            if early_bias is not None:position=position+early_bias[i]
            out.append(cut.profile(position)[0]);out.append(jitter/cut.floor)
        return np.concatenate(out)

    def fit(self,initial,starts=True,max_nfev=160):
        initial=np.r_[np.asarray(initial,float),np.zeros(2*len(self.cuts))] if len(initial)==4 else np.asarray(initial,float)
        fun=self.residual;jac=lambda x:numerical_jacobian(fun,x,self.steps)
        initial_states=[initial]
        if starts:
            # All targets, injections and blanks use the same search freedom.
            for k,shift in [(0,150.),(0,-150.),(2,150.),(2,-150.),(1,10.),(3,-10.)]:
                value=initial.copy();value[k]+=shift;initial_states.append(value)
        fits=[least_squares(fun,v,jac=jac,x_scale=self.scale,ftol=1e-8,xtol=1e-8,gtol=1e-8,max_nfev=max_nfev) for v in initial_states]
        best=min(fits,key=lambda a:float(a.fun@a.fun));normal=best.jac.T@best.jac
        values,vectors=np.linalg.eigh(normal);positive=values>max(values.max()*1e-12,1e-20);identified=np.sum(vectors[:,~positive]**2,axis=1)<1e-6
        cov=(vectors[:,positive]/values[positive])@vectors[:,positive].T if positive.any() else None
        if positive.all():cov=np.linalg.inv(normal)
        competing=[float(a.fun@a.fun) for a in fits if np.linalg.norm(a.x[[0,2]]-best.x[[0,2]])>200 and float(a.fun@a.fun-best.fun@best.fun)<4]
        return dict(fully_identified=bool(positive.all()),identifiable_state_axes=identified[:4],state=best.x[:4],parameters=best.x,conditional_covariance=None if cov is None else cov[:4,:4],full_covariance=cov,jacobian=best.jac,residual=best.fun,chi2=float(best.fun@best.fun),success=bool(best.success),nfev=int(best.nfev),competing_minima=len(competing),start_chi2=[float(a.fun@a.fun) for a in fits])

    def profiles(self,fit,axes=(0,1,2,3),levels=(1.,3.841458820694124)):
        """One-dimensional likelihood profiles, all other variables reoptimized."""
        x=fit['parameters'];cov=fit['full_covariance'];result={}
        if cov is None:return result
        full=np.arange(len(x));steps=self.steps
        for axis in axes:
            if not fit['identifiable_state_axes'][axis]:continue
            free=full[full!=axis]
            def delta(value):
                def fun(nuisance):
                    p=x.copy();p[free]=nuisance;p[axis]=value;return self.residual(p)
                answer=least_squares(fun,x[free],jac=lambda z:numerical_jacobian(fun,z,steps[free]),x_scale=self.scale[free],max_nfev=100,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                return float(answer.fun@answer.fun-fit['chi2'])
            bound={}
            for threshold in levels:
                ends=[]
                for sign in [-1,1]:
                    width=max(np.sqrt(cov[axis,axis]),self.steps[axis]);edge=x[axis]+sign*width
                    for attempt in range(9):
                        if delta(edge)>=threshold:break
                        width*=2;edge=x[axis]+sign*width
                    else:ends.append(None);continue
                    lo,hi=sorted([float(x[axis]),float(edge)])
                    ends.append(float(brentq(lambda v:delta(v)-threshold,lo,hi,xtol=max(.01,width*.002),maxiter=30)))
                bound[str(threshold)]=ends
            result[str(axis)]=bound
        return result

    def common_response(self,fit,early_draws,late_draws):
        """Synchronous frame/physical-group shifts enter both likelihood halves."""
        if fit['full_covariance'] is None:return None
        h=fit['full_covariance']@fit['jacobian'].T;response=[];p=fit['parameters'];original=self.residual(p)
        for early,late in zip(early_draws,late_draws):response.append(-(h@(self.residual(p,early,late)-original))[:4])
        return np.asarray(response)
