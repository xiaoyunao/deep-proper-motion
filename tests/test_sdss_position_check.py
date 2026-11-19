import numpy as np
from deep_pm.footprints import sdss_great_circle
from deep_pm.sdss_position_check import great_circle_icrs


def test_scan_coordinate_rotation_round_trip_including_ra_zero():
    ra=np.array([359.99,.01,183.,188.]);dec=np.array([0.,-1.,33.,70.])
    mu,nu=sdss_great_circle(ra,dec,95.,32.5)
    ar,dr=great_circle_icrs(mu,nu,95.,32.5)
    np.testing.assert_allclose((ar-ra+180)%360-180,0,atol=1e-12)
    np.testing.assert_allclose(dr,dec,atol=1e-12)
