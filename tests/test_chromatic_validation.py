import numpy as np
import pytest
from deep_pm.chromatic_validation import read_model,source_offsets
from deep_pm.continuous_reference import read_offsets


def test_reference_color_and_group_covariance_remain_synchronous(tmp_path):
    names=np.array(['NSC:g','SDSS:r']);draw=np.arange(8.)-3.5
    samples=np.zeros((2,8,2));samples[0,:,0]=draw
    feature=np.zeros((1,8,2));feature[0,:,0]=-draw
    path=tmp_path/'private_chromatic_reference.npz'
    np.savez_compressed(path,names=names,position=np.array([[5.,0],[0,0]]),draws=samples,counts=[100,100],gauge=['0','0'],
        projection_center_deg=[188.,33.],feature_component_labels=['0'],feature_coefficients=[[20.,-10.]],feature_draws=feature,feature_groups=[True,False],reference_color_g_i=1.2)
    with pytest.raises(ValueError,match='requires per-source color'):read_offsets(path)
    model=read_model(path);offset=source_offsets(model,names,2.2)
    np.testing.assert_allclose(offset['NSC:g'][0],[25.,-10.])
    np.testing.assert_allclose(offset['NSC:g'][1],0,atol=1e-14)
    np.testing.assert_allclose(offset['SDSS:r'][0],0)
