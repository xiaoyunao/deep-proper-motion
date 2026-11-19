import numpy as np
import pytest
from deep_pm.zero_point import fit_offsets


def example():
    refs=np.repeat(np.arange(80).astype(str),3);groups=np.tile(['SDSS:1','NSC:r','NSC:g'],80)
    color=np.repeat(np.linspace(-.5,2.5,80),3)
    feature=((groups=='NSC:g')*color)[:,None]
    zero={'SDSS:1':np.array([10,-15.]),'NSC:r':np.array([-20,30.]),'NSC:g':np.array([40,10.])}
    position=np.repeat(np.random.default_rng(3).normal(size=(80,2))*1000,3,axis=0)
    xy=position+np.array([zero[g] for g in groups])+feature*np.array([[35.,-24.]])
    return refs,groups,color,feature,zero,xy


def test_joint_color_and_group_zeros_recover_identifiable_signal():
    refs,groups,color,feature,zero,xy=example()
    result,point,draws=fit_offsets(refs,xy,np.ones_like(xy)*20,groups,30,8,5,features=feature)
    np.testing.assert_allclose(point,[[35,-24]],atol=1e-5)
    np.testing.assert_allclose(result['SDSS:1'][0]-result['NSC:r'][0],zero['SDSS:1']-zero['NSC:r'],atol=1e-5)
    assert draws.shape==(8,1,2)


def test_band_constant_and_source_constant_are_rejected_as_degenerate():
    refs,groups,color,feature,zero,xy=example()
    for value in ((groups=='NSC:g').astype(float),color):
        with pytest.raises(ValueError,match='degenerate'):
            fit_offsets(refs,xy,np.ones_like(xy)*20,groups,30,8,5,features=value[:,None])
