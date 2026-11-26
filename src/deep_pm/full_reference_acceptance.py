"""Independent explicit-whitened 2D check of the matrix-free Schur solver.

Only engineering-sized components: this intentionally constructs the design,
which is forbidden for the production hundred-million-row reference graph.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr

from .blind_catalogue_quality import read
from .full_reference_solver import SchurProblem
from .io import digest,save_json
from .public_summary import check_aggregate
from .zero_point import read_checkpoint


def run(root,output):
    reports=[]
    for dest in sorted((root/'reference').glob('component-*')):
        ptr,g,p,y,names=[np.load(dest/(k+'.npy')) for k in ('ptr','groups','precision','values','names')]
        n=len(g);nr=len(ptr)-1;ng=len(names)
        if n>200000:raise ValueError('Explicit check is limited to engineering graphs')
        ref=np.repeat(np.arange(nr),np.diff(ptr));i=np.arange(n)
        inverse=np.empty((n,2,2));inverse[:,0,0]=p[:,0];inverse[:,0,1]=inverse[:,1,0]=p[:,1];inverse[:,1,1]=p[:,2]
        white=np.linalg.cholesky(inverse).transpose(0,2,1)
        rr=[];cc=[];vv=[]
        for a in range(2):
            for b in range(2):
                rr.extend(2*i+a);cc.extend(2*ref+b);vv.extend(white[:,a,b])
                ok=g<ng-1;rr.extend(2*i[ok]+a);cc.extend(2*(nr+g[ok])+b);vv.extend(white[ok,a,b])
        design=coo_matrix((vv,(rr,cc)),shape=(2*n,2*(nr+ng-1))).tocsr();design.eliminate_zeros()
        rhs=np.einsum('nij,nj->ni',white,y).ravel()
        saved=read_checkpoint(dest/'checkpoints/robust.npz');weight=saved['weight'];problem=SchurProblem(ptr,g,p,y,weight,ng)
        random=np.random.default_rng(45116+int(dest.name.split('-')[-1]));maximum=0.;maximum_objective=0.;tests=0
        for draw in range(-1,48):
            boot=np.ones(nr) if draw<0 else random.exponential(size=nr)
            root_weight=np.repeat(np.sqrt(weight*boot[ref]),2)
            m=design.multiply(root_weight[:,None]).tocsr()
            scale=1/np.sqrt(np.asarray(m.power(2).sum(axis=0)).ravel())
            answer=lsmr(m.multiply(scale),rhs*root_weight,atol=1e-12,btol=1e-12,maxiter=6000)
            if answer[1] not in (0,1,2,4,5):raise RuntimeError('Explicit whitening check did not converge')
            explicit=np.r_[((answer[0]*scale)[2*nr:]).reshape(-1,2),np.zeros((1,2))]
            schur,_=problem.solve(boot)
            maximum=max(maximum,float(np.max(abs(explicit-schur))))
            normal_delta=np.linalg.norm(m@(answer[0]*scale)-rhs*root_weight)
            maximum_objective=max(maximum_objective,float(normal_delta));tests+=1
        if maximum>1e-3:raise ValueError(f'Explicit vs Schur disagreement {maximum}mas')
        reports.append(dict(component=int(dest.name.split('-')[-1]),measurements=n,reference_sources=nr,groups=ng,
            comparisons=tests,maximum_offset_difference_mas=maximum,maximum_whitened_residual_norm=maximum_objective,
            input_binding_sha256=digest(dest/'private_numeric_binding.json')))
    report=dict(components=reports,draws_per_component=48,IRLS='same final frozen weight',tolerance_mas=1e-3,
        scope='Independent explicit full2D sparse whitened solve vs Schur on same complete engineering-sized selected-input graph. These are not full production reference domains.',
        slurm_job=os.environ['SLURM_JOB_ID'])
    check_aggregate(report);save_json(output,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):parser.error('Use Slurm')
    run(**vars(args))
