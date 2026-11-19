"""One scalar geometric DCR coefficient jointly with local physical group zeros.

An independent local C0 control uses exactly the same centered sparse solve.
Point color proxies are conditional; synchronous reference/bootstrap color
perturbations propagate measured color uncertainty, not unknown SED systematics.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr


def fit(refs,xy,sigma,group,geometry,color,color_error,draws,seed,workers=1,with_dcr=True):
    r,names=pd.factorize(refs);g,groups=pd.factorize(group);nr,ng=len(names),len(groups);n=len(r)
    if n<3 or ng<2 or np.asarray(geometry).shape!=(n,2):raise ValueError('Insufficient connected reference geometry')
    design=coo_matrix((np.ones(n*2),(np.r_[np.arange(n),np.arange(n)],np.r_[r,nr+g])),shape=(n,nr+ng)).tocsr()[:,:-1]
    # Subtract a fixed per-reference origin: same mathematical nuisance model,
    # smaller RHS dynamic range; used identically in the no-DCR control.
    origin=np.column_stack([pd.Series(xy[:,a]).groupby(r).transform('median') for a in range(2)])
    y=np.asarray(xy)-origin;sigma=np.asarray(sigma);geometry=np.asarray(geometry)
    proxies=pd.DataFrame(dict(ref=r,color=color,error=color_error)).groupby('ref',sort=True).first()
    colors=proxies.color.to_numpy();errors=proxies.error.to_numpy()
    known=np.isfinite(colors+errors)&(errors>0);colors=np.where(known,colors,1.2);errors=np.where(known,errors,0.)
    def solve(weight,source_color):
        f=geometry*(source_color[r]-1.2)[:,None];parts=[];denom=0.;numer=0.
        for axis in range(2):
            root=np.sqrt(weight)/sigma[:,axis];matrix=design.multiply(root[:,None]).tocsr()
            norm=np.sqrt(np.asarray(matrix.power(2).sum(axis=0)).ravel());scale=np.divide(1,norm,out=np.ones_like(norm),where=norm>0);matrix.data*=scale[matrix.indices]
            def regression(target):
                answer=lsmr(matrix,target,atol=1e-10,btol=1e-10,maxiter=4000)
                if answer[1] not in (0,1,2,4,5):raise RuntimeError('DCR reference projection did not converge')
                return scale*answer[0],target-matrix@answer[0]
            ay,ry=regression(y[:,axis]*root)
            if with_dcr:
                af,rf=regression(f[:,axis]*root);numer+=rf@ry;denom+=rf@rf
            else:af=np.zeros_like(ay)
            parts.append((ay,af))
        if with_dcr and denom<=1e-10:raise ValueError('DCR term degenerate with source positions or group zeros')
        beta=numer/denom if with_dcr else 0.
        coef=np.column_stack([ay-beta*af for ay,af in parts]);residual=y-design@coef-beta*f
        return coef,beta,residual
    weight=np.ones(n)
    for _ in range(6):
        point,beta,residual=solve(weight,colors);weight=np.minimum(1,6/(4+np.sum((residual/sigma)**2,axis=1)))
    random=np.random.default_rng(seed);refweights=random.exponential(size=(draws,nr));color_noise=random.normal(size=(draws,nr))*errors
    def bootstrap(j):
        coeff,value,_=solve(weight*refweights[j][r],colors+color_noise[j]);return coeff[nr:],value
    with ThreadPoolExecutor(max_workers=workers) as pool:samples=list(pool.map(bootstrap,range(draws)))
    gamma=np.r_[point[nr:],np.zeros((1,2))];gamma_draw=np.concatenate([np.stack([v[0] for v in samples]),np.zeros((draws,1,2))],axis=1)
    counts=pd.Series(refs).groupby(np.asarray(group)).nunique()
    offsets={name:(gamma[j],gamma_draw[:,j],int(counts[name])) for j,name in enumerate(groups)}
    return offsets,float(beta),np.array([v[1] for v in samples]),dict(known_color_reference_count=int(known.sum()),reference_count=nr,
        median_color_error=float(np.median(errors[known])) if known.any() else None,color_spread=float(np.std(colors[known])) if known.any() else None)
