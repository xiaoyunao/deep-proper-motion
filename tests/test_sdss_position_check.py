import numpy as np
from deep_pm.footprints import sdss_great_circle
from deep_pm.sdss_position_check import great_circle_icrs


def test_scan_coordinate_rotation_round_trip_including_ra_zero():
    ra=np.array([359.99,.01,183.,188.]);dec=np.array([0.,-1.,33.,70.])
    mu,nu=sdss_great_circle(ra,dec,95.,32.5)
    ar,dr=great_circle_icrs(mu,nu,95.,32.5)
    np.testing.assert_allclose((ar-ra+180)%360-180,0,atol=1e-12)
    np.testing.assert_allclose(dr,dec,atol=1e-12)


def test_native_position_covariance_does_not_double_multiply_cos_declination():
    from deep_pm.sdss_position_check import position_covariance
    field={key:np.zeros(5) for key in ['B','C','E','FF','DROW1','DROW2','DROW3','DCOL1','DCOL2','DCOL3','MUERR','NUERR']}
    field['B'][:]=.396/3600;field['FF'][:]=.396/3600;field['MUERR'][:]=.03;field['NUERR'][:]=.03
    sky=great_circle_icrs(np.array([90.]),np.array([0.]),0.,70.)
    centroid,calibration=position_covariance(np.ones(1),np.ones(1),np.ones(1)*100,field,2,sky,0.,70.)
    np.testing.assert_allclose(np.diagonal(centroid+calibration,axis1=1,axis2=2),np.full((1,2),396**2+30**2),rtol=1e-6)
