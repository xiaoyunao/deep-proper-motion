"""Physical angular metric and full two-axis errors in a shared graph plane.

All covariance inputs here are in native east/north mas, including an angular
floor exactly once. Existing reference draws already use graph-plane mas.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .astrometry import MAS_PER_RADIAN,basis,unproject


def sky_basis(ra,dec):
    a,d=np.deg2rad(np.broadcast_arrays(ra,dec));ca,sa=np.cos(a),np.sin(a);cd,sd=np.cos(d),np.sin(d)
    direction=np.stack([cd*ca,cd*sa,sd],axis=-1)
    east=np.stack([-sa,ca,np.zeros_like(a)],axis=-1)
    north=np.stack([-sd*ca,-sd*sa,cd],axis=-1)
    return np.stack([direction,east,north],axis=-2)


def forward_jacobian(ra,dec,center):
    """d(graph x,y mas)/d(native east,north mas), analytic and vectorized."""
    local=sky_basis(ra,dec);frame=basis(*center);unit=local[...,0,:]
    dot=unit@frame[0]
    if np.any(dot<=0):raise ValueError('Gnomonic forward Jacobian needs one hemisphere')
    q=(unit@frame[1:].T)/dot[...,None]
    derivative=(frame[1:]-q[..., :,None]*frame[0])/dot[...,None,None]
    return np.einsum('...ij,...kj->...ik',derivative,local[...,1:,:])


def native_covariance(sigma,floor=0.,rho=0.):
    sigma=np.asarray(sigma,float)
    if sigma.shape[-1]!=2 or np.any(~np.isfinite(sigma)) or np.any(sigma<=0):raise ValueError('Positive finite native east/north errors required')
    rho=np.broadcast_to(rho,sigma.shape[:-1])
    if np.any(~np.isfinite(rho)) or np.any(abs(rho)>=1):raise ValueError('Native correlation must be finite and strictly between -1 and 1')
    out=np.zeros(sigma.shape[:-1]+(2,2));out[...,0,0]=sigma[...,0]**2+floor**2;out[...,1,1]=sigma[...,1]**2+floor**2
    out[...,0,1]=out[...,1,0]=rho*sigma[...,0]*sigma[...,1]
    return out


def projected_covariance(ra,dec,center,covariance):
    a=forward_jacobian(ra,dec,center)
    return a@covariance@np.swapaxes(a,-1,-2)


def angular_distance(x,y,center):
    a=unproject(np.atleast_2d(x),center);b=unproject(np.atleast_2d(y),center)
    u=sky_basis(*a.T)[...,0,:];v=sky_basis(*b.T)[...,0,:]
    return np.arctan2(np.linalg.norm(np.cross(u,v),axis=-1),np.sum(u*v,axis=-1))*MAS_PER_RADIAN


class AngularTree:
    """cKDTree interface whose reported distances and radii are true sky mas."""
    def __init__(self,xy,center):
        self.center=center;self.xy=np.asarray(xy);self.tree=cKDTree(self.units(self.xy))

    def units(self,xy):
        pos=unproject(np.asarray(xy).reshape(-1,2),self.center)
        return sky_basis(*pos.T)[:,0,:]

    def query(self,xy,k=1,**kwargs):
        query=np.asarray(xy);distance,index=self.tree.query(self.units(query),k=k,**kwargs)
        angle=np.where(np.isfinite(distance),2*np.arcsin(np.clip(distance/2,0,1))*MAS_PER_RADIAN,np.inf)
        return (angle[0],index[0]) if query.ndim==1 else (angle,index)

    def query_ball_point(self,xy,r,**kwargs):
        query=np.asarray(xy);radius=2*np.sin(np.asarray(r)/(2*MAS_PER_RADIAN))
        values=self.tree.query_ball_point(self.units(query),radius,**kwargs)
        return values[0] if query.ndim==1 else values


def design_matrix(time,epoch):
    dt=np.asarray(time)-epoch;out=np.zeros((len(dt),2,4))
    out[:,0,0]=1;out[:,0,1]=dt;out[:,1,2]=1;out[:,1,3]=dt
    return out


def covariance_of_estimator(influence,covariance,weight=None):
    """H C H^T; pass weight only for the explicit C/w noise-inflation model."""
    c=np.asarray(covariance,float)
    if weight is not None:c=c/np.asarray(weight)[:,None,None]
    return np.einsum('pni,nij,qnj->pq',influence,c,influence)


def old_influence(row,time,sigma):
    design=np.column_stack([np.ones(len(time)),time-row['reference_epoch']])
    h=np.zeros((4,len(time),2));cov=np.asarray(row['independent_tangent_covariance'])
    for axis in range(2):
        sl=slice(2*axis,2*axis+2)
        h[sl,:,axis]=cov[sl,sl]@(design.T*(np.asarray(row['robust_weight'])/sigma[:,axis]**2))
    return h


@dataclass
class VectorMotion:
    coeff:np.ndarray
    cov:np.ndarray
    epoch:float
    residual:np.ndarray
    weight:np.ndarray
    influence:np.ndarray
    chi2:float
    span:float

    def predict(self,times):
        return self.coeff[0]+(np.asarray(times)[...,None]-self.epoch)*self.coeff[1]


def fit_vector_motion(xy,time,covariance,epoch,robust=True,frozen_weight=None):
    """Full GLS, same min(1,6/(4+d²)) IRLS loss and iteration count as W0."""
    xy=np.asarray(xy,float);time=np.asarray(time,float);covariance=np.asarray(covariance,float)
    if len(time)<2 or np.ptp(time)<=0 or covariance.shape!=(len(time),2,2):raise ValueError('Need distinct times and full two-axis covariance')
    if not np.isfinite(covariance).all() or np.any(np.linalg.eigvalsh(covariance)<=0):raise ValueError('Measurement covariance must be finite positive definite')
    origin=np.median(xy,axis=0);middle=time.mean();design=design_matrix(time,middle);inverse=np.linalg.inv(covariance)
    weight=np.ones(len(time)) if frozen_weight is None else np.asarray(frozen_weight,float).copy()
    if np.any(weight<=0):raise ValueError('Positive estimator weights required')
    iterations=8 if robust and len(time)>=3 and frozen_weight is None else 1
    for iteration in range(iterations):
        w=inverse*weight[:,None,None]
        normal=np.einsum('nip,nij,njq->pq',design,w,design)
        rhs=np.einsum('nip,nij,nj->p',design,w,xy-origin)
        state=np.linalg.solve(normal,rhs)
        residual=xy-origin-np.einsum('nip,p->ni',design,state)
        distance=np.einsum('ni,nij,nj->n',residual,inverse,residual)
        if iteration<iterations-1:weight=np.minimum(1,6/(4+distance))
    transform=np.eye(4);transform[0,1]=transform[2,3]=epoch-middle
    coefficient=transform@state;coefficient[[0,2]]+=origin
    independent=transform@np.linalg.inv(normal)@transform.T
    h=transform@np.linalg.inv(normal)@np.einsum('nip,nij->pnj',design,w).reshape(4,-1)
    h=h.reshape(4,len(time),2)
    return VectorMotion(coefficient.reshape(2,2).T,independent,epoch,residual,weight,h,float(distance.sum()),float(np.ptp(time)))


def reference_covariance(h,draws):
    """Draw input (draw,measurement,xy) is already in graph-plane units."""
    response=np.einsum('pni,dni->dp',h,draws)
    return np.cov(response,rowvar=False),-response
