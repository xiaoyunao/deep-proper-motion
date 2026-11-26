import numpy as np
import pytest

from deep_pm.astrometry import project,unproject,fit_motion,MAS_PER_RADIAN
from deep_pm.standard_motion import icrs_motion
from deep_pm.coordinate_metric import (sky_basis,forward_jacobian,native_covariance,
    projected_covariance,AngularTree,angular_distance,fit_vector_motion,covariance_of_estimator)


def shifted(ra,dec,east,north):
    local=sky_basis(ra,dec);p=local[0]+(east*local[1]+north*local[2])/MAS_PER_RADIAN
    p=p/np.linalg.norm(p)
    return np.rad2deg(np.arctan2(p[1],p[0]))%360,np.rad2deg(np.arcsin(p[2]))


@pytest.mark.parametrize('point,center', [((175,5),(170,3)),((235,5),(180,25)),((.00001,65),(355,30)),((123,87),(118,40))])
def test_forward_jacobian_and_inverse_at_near_far_wrap_high_declination(point,center):
    a=forward_jacobian(*point,center);h=10.
    numerical=np.column_stack([(project(*shifted(*point,*(h*np.eye(2)[j])),center)-project(*shifted(*point,*(-h*np.eye(2)[j])),center))[0]/(2*h) for j in range(2)])
    np.testing.assert_allclose(a,numerical,rtol=1.e-5,atol=1.e-7)
    xy=project(*point,center)[0];j=icrs_motion(xy,[30,-12],center)['state_jacobian'][np.ix_([1,3],[1,3])]
    np.testing.assert_allclose(j@a,np.eye(2),rtol=1.e-10,atol=1.e-10)


def test_full_gls_retains_original_loss_and_declared_inflated_noise_covariance():
    rng=np.random.default_rng(2401);time=np.array([2001,2006,2015,2015.01,2016,2018.])
    sigma=rng.uniform(40,70,(len(time),2));xy=np.array([15,-3])*(time-2015.5)[:,None]+rng.normal(size=(len(time),2))*sigma
    xy[2]+=[300,-400]
    old=fit_motion(xy,time,sigma,2015.5);cov=native_covariance(sigma)
    new=fit_vector_motion(xy,time,cov,2015.5)
    np.testing.assert_allclose(new.coeff,old.coeff,rtol=1.e-10,atol=1.e-10)
    np.testing.assert_allclose(new.weight,old.weight,rtol=1.e-10,atol=1.e-10)
    np.testing.assert_allclose(new.cov,covariance_of_estimator(new.influence,cov,new.weight),rtol=1.e-12,atol=1.e-12)
    sandwich=covariance_of_estimator(new.influence,cov)
    assert np.trace(sandwich)<np.trace(new.cov)
    noise=rng.multivariate_normal([0,0],np.eye(2),size=(20000,len(time)))*sigma
    propagated=np.einsum('pni,dni->dp',new.influence,noise)
    np.testing.assert_allclose(np.diag(np.cov(propagated,rowvar=False)),np.diag(sandwich),rtol=.035)


def test_one_physical_state_and_correlated_covariance_survive_graph_reexpression():
    point=(235.,5.);first=(180.,25.);second=(230.,0.)
    x1=project(*point,first)[0];vsky=np.array([70.,-30.]);v1=forward_jacobian(*point,first)@vsky
    cov=np.array([[900,2,150,1],[2,9,1,2],[150,1,1600,3],[1,2,3,16.]])
    a=icrs_motion(x1,v1,first,cov)
    x2=project(*point,second)[0];v2=forward_jacobian(*point,second)@a['pm_icrs_masyr']
    b=icrs_motion(x2,v2,second)
    transform=np.linalg.solve(b['state_jacobian'],a['state_jacobian'])
    b=icrs_motion(x2,v2,second,transform@cov@transform.T)
    np.testing.assert_allclose(b['pm_icrs_masyr'],a['pm_icrs_masyr'],atol=1.e-9)
    np.testing.assert_allclose(b['covariance'],a['covariance'],rtol=1.e-10,atol=1.e-9)


@pytest.mark.parametrize('center',[(359.,30.),(340.,5.),(305.,10.)])
def test_spherical_radius_has_identical_meaning_for_all_graph_centers(center):
    point=(.00001,35.);p=np.array([shifted(*point,s,0) for s in (300,800,1200)])
    xy=project(*p.T,center);goal=project(*point,center)[0]
    tree=AngularTree(xy,center)
    distance,index=tree.query(goal,k=3)
    np.testing.assert_allclose(distance,[300,800,1200],atol=.0001)
    assert set(tree.query_ball_point(goal,1000))=={0,1}
    np.testing.assert_allclose(angular_distance(xy,goal,center),[300,800,1200],atol=.0001)


def test_projection_keeps_full_covariance_and_invariant_mahalanobis_distance():
    covariance=native_covariance(np.array([[30.,80.],[70.,40.]]),floor=50,rho=[.4,-.2])
    ra,dec=np.array([181.,237.]),np.array([24.,5.]);a=forward_jacobian(ra,dec,[180.,25.])
    cp=projected_covariance(ra,dec,[180.,25.],covariance)
    residual=np.array([[70.,-20.],[60.,110.]])
    r=np.einsum('nij,nj->ni',a,residual)
    np.testing.assert_allclose(np.einsum('ni,nij,nj->n',r,np.linalg.inv(cp),r),np.einsum('ni,nij,nj->n',residual,np.linalg.inv(covariance),residual),rtol=1.e-12)
    assert np.any(abs(cp[:,0,1])>1)


def test_bounded_reference_has_explicit_fixed_boundary_and_paired_draws():
    import pandas as pd
    from deep_pm.coordinate_reference import bounded_fit
    rng=np.random.default_rng(21621);n=50
    reference=np.repeat(np.arange(n),3);group=np.tile(['fixed','freeA','freeB'],n)
    shift=np.tile(np.array([[0,0],[15,-21],[-11,7]]),(n,1))
    position=np.repeat(rng.normal(0,2000,(n,2)),3,axis=0)
    xy=position+shift+rng.normal(0,.01,(3*n,2))
    frame=pd.DataFrame(dict(reference=reference,group=group,corrected_e=xy[:,0],corrected_n=xy[:,1]))
    cov=np.repeat(np.array([[[4.,1.],[1.,9.]]]),3*n,axis=0)
    point,draws,meta=bounded_fit(frame,cov,['freeA','freeB'],draws=6)
    np.testing.assert_allclose(point,[[15,-21],[-11,7]],atol=.01)
    point2,draws2,_=bounded_fit(frame,cov,['freeA','freeB'],draws=6)
    np.testing.assert_array_equal(draws,draws2)
    assert meta['fixed_boundary_groups']==1
    with pytest.raises(ValueError,match='fixed boundary'):
        bounded_fit(frame,cov,['fixed','freeA','freeB'],draws=1)


def test_actual_chip_opportunity_keeps_band_and_depth_unknown_separate():
    import pandas as pd
    from types import SimpleNamespace
    from astropy.time import Time
    from deep_pm.coordinate_candidate import opportunities
    center=(210.,20.)
    data=pd.DataFrame(dict(survey=['NSC','NSC','NSC'],band=['r','r','g'],magnitude=[20.,22.,20.]))
    models=[SimpleNamespace(coeff=np.array([[0.,0.],[0.,0.]]),epoch=2015.5) for _ in range(3)]
    chip=dict(exposure='test',mjd=float(Time('2020-04-01',scale='utc').mjd),filter='r',depth10sig=21.)
    for i,(da,dd) in enumerate([(-.01,-.01),(.01,-.01),(.01,.01),(-.01,.01)],1):
        chip[f'vertex_ra{i}']=center[0]+da;chip[f'vertex_dec{i}']=center[1]+dd
    result,_=opportunities(data,models,[np.array([i]) for i in range(3)],pd.DataFrame([chip]),center,{'NSC:test':(np.zeros(2),None,None)},2020)
    np.testing.assert_array_equal(result['geometric'],[True,True,True])
    np.testing.assert_array_equal(result['catalogue_10sigma'],[True,False,False])
    np.testing.assert_array_equal(result['depth_or_band_unknown'],[False,False,True])


def test_legacy_sweep_filename_uses_integer_declination_degrees(tmp_path):
    from astropy.io import fits
    from deep_pm.coordinate_candidate import independent_morphology
    folder=tmp_path/'south/sweep/9.0';folder.mkdir(parents=True)
    table=fits.BinTableHDU.from_columns([
        fits.Column(name='RA',format='D',array=[211.]),fits.Column(name='DEC',format='D',array=[28.5]),
        fits.Column(name='TYPE',format='3A',array=['PSF'])])
    table.writeto(folder/'sweep-210p025-220p030.fits')
    values,meta=independent_morphology(np.zeros((1,2)),tmp_path,(211.,28.5))
    assert meta['sweep_files']==1 and values[0]['shape']=='PSF'
