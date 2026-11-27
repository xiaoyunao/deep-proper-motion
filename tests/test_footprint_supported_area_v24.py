import numpy as np
from numba import types
from numba.typed import Dict

from deep_pm.astrometry import basis
from deep_pm.coordinate_metric import forward_jacobian
from deep_pm.footprint_supported_area_v24 import accepted_direction,mark
from deep_pm.footprint_raster_v24 import chip_geometry
import pandas as pd


def test_native_direction_threshold_matches_full_inverse_jacobian():
    center=(355.,31.)
    plane=np.array([[525.,110.],[110.,138.]])
    for ra,dec in [(354.,30.),(301.,7.),(46.,65.)]:
        inv=np.linalg.inv(forward_jacobian(np.array([ra]),np.array([dec]),center)[0])
        se2=np.diag(inv@plane@inv.T).max()
        for factor in (.999,1.001):
            covariance=plane*(144/se2)*factor
            accepted=accepted_direction(np.deg2rad(ra),np.deg2rad(dec),basis(*center),covariance)
            assert accepted==(factor<1.)


def test_sparse_tiles_keep_components_separate_across_meridian():
    frame=pd.DataFrame(dict(mjd=[56000.,58000.]))
    for i,(ra,dec) in enumerate([(-.61,-.61),(.61,-.61),(.61,.61),(-.61,.61)],1):
        frame[f"vertex_ra{i}"]=ra
        frame[f"vertex_dec{i}"]=dec
    tiles=Dict.empty(key_type=types.int64,value_type=types.float64[:,:,::1])
    geom=chip_geometry(frame)
    models=np.stack([basis(0.,0.)]*2)
    mark(tiles,1,np.array([0,1]),models,np.stack([np.eye(2),np.eye(2)]),np.arange(2),*geom)
    assert len(tiles)==8
    for key in tiles:
        val=tiles[key]
        assert val[0,0,0]==val[1,0,0]
        assert abs(val[0,0,0]-[56000.,58000.][key//64800])<.01
