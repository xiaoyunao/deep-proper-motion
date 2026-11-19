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
