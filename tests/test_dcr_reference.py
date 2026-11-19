import numpy as np
import pytest
from deep_pm.dcr_reference_fit import fit
from deep_pm.dcr_geometry import tangent_geometry


def test_one_physical_dcr_coefficient_and_group_covariance_recover_signal():
    random=np.random.default_rng(12);nr=40;ng=4
    refs=np.repeat(np.arange(nr),ng);groups=np.tile(['SDSS:r','NSC:g1','NSC:g2','NSC:z'],nr)
    c=np.repeat(np.linspace(.1,2.4,nr),ng);ce=np.full(len(c),.03)
    geometry=np.tile([[0,0],[.6,1.4],[-.4,1.8],[0,0]],(nr,1));gamma=np.tile([[15,-18],[4,6],[-3,4],[0,0]],(nr,1))
    origin=np.repeat(random.normal(size=(nr,2))*1e6,ng,axis=0);xy=origin+gamma+31.5*(c-1.2)[:,None]*geometry
    offsets,beta,draws,info=fit(refs,xy,np.full_like(xy,40),groups,geometry,c,ce,8,13,workers=2)
    assert beta==pytest.approx(31.5,abs=1e-4)
    for g,expected in zip(groups[:4],gamma[:4]):np.testing.assert_allclose(offsets[g][0],expected,atol=1e-4)
    assert np.std(draws)>0 and info['known_color_reference_count']==nr
    with pytest.raises(ValueError,match='degenerate'):fit(refs,xy,np.full_like(xy,40),groups,np.zeros_like(geometry),c,ce,2,13)


def test_tangent_dcr_geometry_is_zero_at_zenith_and_towards_north():
    z=np.array([[1.,0,0],[np.sqrt(.5),0,np.sqrt(.5)]])
    h,secz=tangent_geometry([0,0],[0,0],z,(0,0))
    np.testing.assert_allclose(h,[[0,0],[0,1]],atol=1e-12)
    np.testing.assert_allclose(secz,[1,np.sqrt(2)])


def test_photometric_error_read_uses_exact_physical_sdss_identity(tmp_path):
    from astropy.io import fits
    from deep_pm.dcr_geometry import sdss_colors,combine_color
    import pandas as pd
    root=tmp_path/'photoObj/301/123/2';root.mkdir(parents=True)
    columns=[fits.Column(name='ID',format='J',array=[7,9]),fits.Column(name='PSFMAG',format='5E',array=[[20,21,20.4,20.1,20],[20,22,21.4,21.2,21]]),fits.Column(name='PSFMAGERR',format='5E',array=[[.1,.03,.04,.04,.1],[.2,.06,.05,.08,.2]])]
    fits.BinTableHDU.from_columns(columns).writeto(root/'photoObj-000123-2-0010.fits')
    phot,receipts=sdss_colors(['SDSS:123:2:10:9','SDSS:123:2:10:7'],tmp_path)
    assert len(receipts)==1
    np.testing.assert_allclose(phot.color_error,[.1,.05],rtol=1e-6)
    assert phot.detection_id.tolist()==['SDSS:123:2:10:9','SDSS:123:2:10:7']
    mean,error=combine_color(phot)
    assert error==pytest.approx(1/np.sqrt(500),rel=1e-6)


def test_source_color_error_is_shared_and_joint_beta_group_draws_cancel():
    from deep_pm.dcr_validation import offsets
    draw=np.arange(8.)-3.5
    samples=np.column_stack([np.zeros(8),-draw])
    z=np.array([np.sqrt(.5),0,np.sqrt(.5)])
    model=dict(base={k:(np.zeros(2),samples.copy(),30) for k in ('g1','g2')},geometry={k:(True,z) for k in ('g1','g2')},beta=10.,beta_draw=draw,center=(0.,0.))
    exact,known=offsets(model,['g1','g2'],[0,0],2.2,1e-12,'same source')
    assert known
    np.testing.assert_allclose(exact['g1'][0],[0,10],atol=1e-10)
    np.testing.assert_allclose(exact['g1'][1],0,atol=1e-8)
    noisy,_=offsets(model,['g1','g2'],[0,0],2.2,.2,'same source')
    np.testing.assert_array_equal(noisy['g1'][1],noisy['g2'][1])
    assert np.std(noisy['g1'][1][:,1])>1


def test_no_dcr_local_solver_matches_original_c0_on_same_large_coordinate_graph():
    from deep_pm.dcr_reference_fit import fit
    from deep_pm.zero_point import fit_offsets
    rng=np.random.default_rng(42);nr=30;groups=np.tile(np.arange(3),nr);refs=np.repeat(np.arange(nr),3)
    xy=np.repeat(rng.normal(1e8,1e6,(nr,2)),3,axis=0)+np.array([[12.,-5.],[-8.,15.],[0.,0.]])[groups]+rng.normal(0,2,(len(groups),2));sigma=np.full_like(xy,50.)
    old=fit_offsets(refs,xy,sigma,groups,5,8,42)
    local,_,_,_=fit(refs,xy,sigma,groups,np.zeros_like(xy),np.repeat(1.2,len(xy)),np.repeat(.1,len(xy)),8,42,with_dcr=False)
    for group in old:np.testing.assert_allclose(old[group][0]-old[2][0],local[group][0]-local[2][0],atol=1e-3)
