import numpy as np
from scipy.linalg import block_diag
from deep_pm.full_reference_solver import SchurProblem,fit_component


def example():
    rng=np.random.default_rng(22);ng=7;nr=80
    groups=np.concatenate([np.sort(rng.choice(ng,size=4,replace=False)) for _ in range(nr)]).astype('i4')
    ptr=np.arange(0,len(groups)+1,4,dtype='i8');r=np.repeat(np.arange(nr),4)
    a=rng.normal(size=(len(groups),2,2));cov=a@a.transpose(0,2,1)+np.eye(2)*2
    inv=np.linalg.inv(cov);precision=inv[:,[0,0,1],[0,1,1]]
    y=rng.normal(size=(nr,2))[r]*20+rng.normal(size=(ng,2))[groups]*6+rng.normal(size=(len(groups),2))
    weight=rng.uniform(.3,1,size=len(groups))
    return ptr,groups,precision,y,weight,inv,r,ng,nr


def test_schur_equals_full_whitened_normal_and_synchronous_bootstrap():
    ptr,g,p,y,w,inv,r,ng,nr=example()
    problem=SchurProblem(ptr,g,p,y,w,ng)
    for boot in (np.ones(nr),np.random.default_rng(4).exponential(size=nr)):
        z,meta=problem.solve(boot)
        design=np.zeros((2*len(g),2*(nr+ng-1)))
        for i in range(len(g)):
            design[2*i:2*i+2,2*r[i]:2*r[i]+2]=np.eye(2)
            if g[i]<ng-1:design[2*i:2*i+2,2*(nr+g[i]):2*(nr+g[i])+2]=np.eye(2)
        precision=block_diag(*list(inv*(w*boot[r])[:,None,None]))
        full=np.linalg.solve(design.T@precision@design,design.T@precision@y.ravel())
        np.testing.assert_allclose(z[:-1],full[2*nr:].reshape(-1,2),atol=2e-8)
        assert meta['relative_normal_residual']<1e-8


def test_final_weight_point_gauge_and_checkpoint_resume(tmp_path):
    ptr,g,p,y,w,inv,r,ng,nr=example();names=['NSC:'+str(i) for i in range(ng-1)]+['SDSS:1']
    point,draw,weight,meta=fit_component(ptr,g,p,y,names,draws=4,checkpoint=tmp_path)
    problem=SchurProblem(ptr,g,p,y,weight,ng);fit,_=problem.solve()
    fit-=np.median(fit[:-1],axis=0)
    np.testing.assert_allclose(point,fit,atol=2e-8)
    np.testing.assert_allclose(np.median(point[:-1],axis=0),0,atol=1e-12)
    np.testing.assert_allclose(np.median(draw[:-1],axis=0),0,atol=1e-12)
    again=fit_component(ptr,g,p,y,names,draws=4,checkpoint=tmp_path)
    np.testing.assert_array_equal(point,again[0]);np.testing.assert_array_equal(draw,again[1])


def test_native_application_support_changes_along_one_scan():
    from deep_pm.full_reference_run import local_support
    rng=np.random.default_rng(4)
    draws=rng.normal(size=(1,48,2))*18
    keep,se=local_support([0.,60.],[0.,0.],(0.,0.),['SDSS:1','SDSS:1'],['SDSS:1'],draws)
    assert not keep[0] and keep[1]
    np.testing.assert_allclose(se[1]/se[0],[.25,.5],rtol=1e-12)


def test_partial_irls_steps_restore_the_same_final_point(tmp_path):
    ptr,g,p,y,w,inv,r,ng,nr=example();names=['NSC:'+str(i) for i in range(ng-1)]+['SDSS:1']
    before=fit_component(ptr,g,p,y,names,draws=2,checkpoint=tmp_path)
    # Simulate loss of the final envelope and all draws, retaining signed IRLS
    # steps. Resume must use those same weights, not mix a later point/weight.
    for path in tmp_path.glob('robust.*'):path.unlink()
    for path in tmp_path.glob('draw-*'):path.unlink()
    after=fit_component(ptr,g,p,y,names,draws=2,checkpoint=tmp_path)
    np.testing.assert_array_equal(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1])
    assert all(step.get('reused') for step in after[3]['irls'])


def test_forked_draws_preserve_exact_serial_sample_order_and_resume(tmp_path):
    ptr,g,p,y,w,inv,r,ng,nr=example()
    names=['NSC:'+str(i) for i in range(ng-1)]+['SDSS:1']
    serial=fit_component(ptr,g,p,y,names,draws=6)
    parallel=fit_component(ptr,g,p,y,names,draws=6,checkpoint=tmp_path,draw_workers=2)
    for left,right in zip(serial[:3],parallel[:3]):
        np.testing.assert_array_equal(left,right)
    (tmp_path/'draw-002.npz').unlink()
    recovered=fit_component(ptr,g,p,y,names,draws=6,checkpoint=tmp_path,draw_workers=3)
    np.testing.assert_array_equal(serial[1],recovered[1])
    assert sum(bool(t.get('reused')) for t in recovered[3]['draws'])==5
