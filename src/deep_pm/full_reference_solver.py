"""Float64, object-block Schur reference solver without a 2N sparse design.

Rows must be contiguous by reference, and the entire connected component is
solved. One group is temporarily pinned for inversion; published gauge is the
pre-SE NSC median. Bayesian draws freeze final IRLS, as in the archived C0.
"""
from pathlib import Path
import inspect
from time import monotonic

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f

from .zero_point import read_checkpoint, write_checkpoint


@njit(cache=True)
def inverse2(a):
    out=np.empty_like(a)
    for j in range(len(a)):
        det=a[j,0]*a[j,2]-a[j,1]*a[j,1]
        if det<=0:raise ValueError('Non-positive normal block')
        out[j,0]=a[j,2]/det;out[j,1]=-a[j,1]/det;out[j,2]=a[j,0]/det
    return out


@njit(cache=True)
def prepare(ptr, groups, precision, values, robust, ng):
    """Per-source normal inverse, weighted datum and exact diagonal Schur blocks."""
    nr=len(ptr)-1
    weight=precision*robust.reshape(-1,1)
    a=np.zeros((nr,3));b=np.zeros((nr,2))
    for s in range(nr):
        for i in range(ptr[s],ptr[s+1]):
            u,v,w=weight[i];x,y=values[i]
            a[s,0]+=u;a[s,1]+=v;a[s,2]+=w
            b[s,0]+=u*x+v*y;b[s,1]+=v*x+w*y
    ai=inverse2(a)
    return weight,ai,b


@njit(cache=True)
def normal_system(ptr,groups,w,ai,b,y,boot,ng):
    diagonal=np.zeros((ng,3));rhs=np.zeros((ng,2))
    for s in range(len(ptr)-1):
        aa,ab,bb=ai[s];bx,by=b[s]
        px=aa*bx+ab*by;py=ab*bx+bb*by
        for i in range(ptr[s],ptr[s+1]):
            g=groups[i];u,v,t=w[i];z=boot[s]
            dx=u*aa+v*ab;dy=u*ab+v*bb
            ex=v*aa+t*ab;ey=v*ab+t*bb
            diagonal[g,0]+=z*(u-dx*u-dy*v)
            diagonal[g,1]+=z*(v-dx*v-dy*t)
            diagonal[g,2]+=z*(t-ex*v-ey*t)
            xx,yy=y[i,0]-px,y[i,1]-py
            rhs[g,0]+=z*(u*xx+v*yy);rhs[g,1]+=z*(v*xx+t*yy)
    return diagonal,rhs


@njit(cache=True)
def schur_product(ptr, groups, w, ai, boot, z, ng):
    result=np.zeros((ng,2))
    for s in range(len(ptr)-1):
        bx=0.;by=0.
        for i in range(ptr[s],ptr[s+1]):
            g=groups[i];u,v,t=w[i];x,y=z[g]
            bx+=u*x+v*y;by+=v*x+t*y
        aa,ab,bb=ai[s];px=aa*bx+ab*by;py=ab*bx+bb*by
        for i in range(ptr[s],ptr[s+1]):
            g=groups[i];u,v,t=w[i];x,y=z[g,0]-px,z[g,1]-py
            result[g,0]+=boot[s]*(u*x+v*y);result[g,1]+=boot[s]*(v*x+t*y)
    return result


@njit(cache=True)
def residual_distance(ptr,groups,w,ai,b,y,z,precision):
    result=np.empty(len(groups));source=np.empty((len(ptr)-1,2))
    for s in range(len(ptr)-1):
        bx,by=b[s]
        for i in range(ptr[s],ptr[s+1]):
            g=groups[i];u,v,t=w[i];x,yy=z[g]
            bx-=u*x+v*yy;by-=v*x+t*yy
        aa,ab,bb=ai[s];px=aa*bx+ab*by;py=ab*bx+bb*by
        source[s,0]=px;source[s,1]=py
        for i in range(ptr[s],ptr[s+1]):
            g=groups[i];x,yy=y[i,0]-px-z[g,0],y[i,1]-py-z[g,1]
            u,v,t=precision[i]
            result[i]=u*x*x+2*v*x*yy+t*yy*yy
    return result,source


class SchurProblem:
    def __init__(self,ptr,groups,precision,values,robust,ng):
        self.ptr=np.asarray(ptr,dtype=np.int64);self.groups=np.asarray(groups,dtype=np.int32)
        self.precision=np.asarray(precision,dtype=np.float64);self.values=np.asarray(values,dtype=np.float64)
        self.robust=np.asarray(robust,dtype=np.float64);self.ng=ng
        if self.ptr[0]!=0 or self.ptr[-1]!=len(groups) or np.any(np.diff(self.ptr)<2):
            raise ValueError('Each active reference needs at least two rows')
        if not np.isfinite(self.values).all() or not np.isfinite(self.precision).all():raise ValueError('Non-finite normal input')
        self.w,self.ai,self.b=prepare(self.ptr,self.groups,self.precision,self.values,self.robust,ng)

    def solve(self,boot=None,x0=None,rtol=1e-10,maxiter=4000):
        if boot is None:boot=np.ones(len(self.ptr)-1)
        boot=np.asarray(boot,dtype=np.float64)
        if len(boot)!=len(self.ptr)-1 or not np.isfinite(boot).all() or np.any(boot<=0):raise ValueError('Positive source bootstrap weights required')
        diagonal,rhs=normal_system(self.ptr,self.groups,self.w,self.ai,self.b,self.values,boot,self.ng)
        pre=inverse2(diagonal[:-1]);n=2*(self.ng-1)
        def product(x):
            z=np.zeros((self.ng,2));z[:-1]=x.reshape(-1,2)
            return schur_product(self.ptr,self.groups,self.w,self.ai,boot,z,self.ng)[:-1].ravel()
        def precondition(x):
            x=x.reshape(-1,2)
            return np.column_stack((pre[:,0]*x[:,0]+pre[:,1]*x[:,1],pre[:,1]*x[:,0]+pre[:,2]*x[:,1])).ravel()
        count=[0];start=monotonic()
        def callback(x):
            count[0]+=1
            if count[0]%50==0:print('SCHUR_CG_PROGRESS',count[0],round(monotonic()-start,2),flush=True)
        tolerance={'rtol':rtol} if 'rtol' in inspect.signature(cg).parameters else {'tol':rtol}
        x,info=cg(LinearOperator((n,n),matvec=product,dtype=float),rhs[:-1].ravel(),
            M=LinearOperator((n,n),matvec=precondition,dtype=float),x0=None if x0 is None else np.asarray(x0)[:-1].ravel(),
            atol=1e-14,maxiter=maxiter,callback=callback,**tolerance)
        residual=product(x)-rhs[:-1].ravel()
        relative=float(np.linalg.norm(residual)/max(np.linalg.norm(rhs[:-1]),1e-30))
        if info!=0 or relative>max(20*rtol,1e-8):raise RuntimeError(f'Schur CG nonconvergence: info={info}, relative={relative}')
        z=np.zeros((self.ng,2));z[:-1]=x.reshape(-1,2)
        return z,dict(iterations=count[0],relative_normal_residual=relative,elapsed_seconds=monotonic()-start)

    def distances(self,z):
        return residual_distance(self.ptr,self.groups,self.w,self.ai,self.b,self.values,z,self.precision)


def fit_component(ptr,groups,precision,values,names,draws=48,seed=45116,checkpoint=None,iterations=6):
    """Six weight updates plus a final point solve; draws fix that final weight."""
    ng=len(names);nr=len(ptr)-1
    checkpoint=None if checkpoint is None else Path(checkpoint)
    if checkpoint is not None:checkpoint.mkdir(parents=True,exist_ok=True)
    saved=None if checkpoint is None else read_checkpoint(checkpoint/'robust.npz')
    trace=[]
    if saved is None:
        weight=np.ones(len(groups));z=None
        for iteration in range(iterations+1):
            problem=SchurProblem(ptr,groups,precision,values,weight,ng)
            step_path=None if checkpoint is None else checkpoint/f'irls-{iteration:03d}.npz'
            step=None if step_path is None else read_checkpoint(step_path)
            if step is None:
                z,meta=problem.solve(x0=z)
            else:
                np.testing.assert_array_equal(weight,step['weight'])
                z=step['point']
                meta=dict(iterations=int(step['iterations']),relative_normal_residual=float(step['relative_normal_residual']),
                    elapsed_seconds=float(step['elapsed_seconds']),reused=True)
            distance,_=problem.distances(z)
            trace.append(dict(iteration=iteration,robust_objective=float(np.sum(np.where(distance<=2,distance,2+6*np.log((4+distance)/6)))),
                mahalanobis_sum=float(distance.sum()),**meta))
            if step is None and step_path is not None:
                write_checkpoint(step_path,weight=weight,point=z,iterations=np.array(meta['iterations']),
                    relative_normal_residual=np.array(meta['relative_normal_residual']),elapsed_seconds=np.array(meta['elapsed_seconds']))
            print('SCHUR_IRLS',trace[-1],flush=True)
            if iteration<iterations:weight=np.minimum(1,6/(4+distance))
        if checkpoint is not None:write_checkpoint(checkpoint/'robust.npz',weight=weight,point=z)
    else:
        weight,z=saved['weight'],saved['point']
        if weight.shape!=(len(groups),) or z.shape!=(ng,2):raise ValueError('Incompatible robust checkpoint')
        problem=SchurProblem(ptr,groups,precision,values,weight,ng)
    random=np.random.default_rng(seed);samples=[];draw_trace=[]
    for draw in range(draws):
        boot=random.exponential(size=nr)
        path=None if checkpoint is None else checkpoint/f'draw-{draw:03d}.npz'
        old=None if path is None else read_checkpoint(path)
        if old is None:
            value,meta=problem.solve(boot,x0=z)
            if path is not None:write_checkpoint(path,point=value)
            print('SCHUR_DRAW',draw,meta,flush=True)
        else:
            value=old['point'];meta=dict(reused=True)
            if value.shape!=(ng,2):raise ValueError('Incompatible draw checkpoint')
        samples.append(value);draw_trace.append(meta)
    # Computational pin does not define the published physical gauge.
    nsc=np.asarray([str(n).startswith('NSC:') for n in names])
    if not nsc.any():raise ValueError('No NSC gauge groups in connected component')
    z=z-np.median(z[nsc],axis=0)
    samples=np.asarray(samples).transpose(1,0,2)
    samples-=samples.mean(axis=1,keepdims=True)
    samples-=np.median(samples[nsc],axis=0)[None]
    return z,samples,weight,dict(irls=trace,draws=draw_trace,bootstrap_IRLS='frozen final',gauge_nsc_group_count=int(nsc.sum()))
