import unittest
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from deep_pm.astrometry import fit_motion, project
from deep_pm.precision_v03 import fit_with_calibration
from deep_pm.standard_motion import calibrated_covariance, icrs_motion


class StandardMotionTest(unittest.TestCase):
    def test_inverse_velocity_matches_independent_spherical_propagation(self):
        # Include RA wrap and high declination, away from the singular pole.
        for center, target in [((350.,0.),(355.,5.)), ((0.,65.),(359.5,70.)), ((132.,33.),(135.,36.))]:
            sky = SkyCoord(*target*u.deg, pm_ra_cosdec=1234*u.mas/u.yr, pm_dec=-678*u.mas/u.yr)
            later = sky.directional_offset_by(np.arctan2(1234,-678)*u.rad, np.hypot(1234,678)*u.mas)
            a = project([sky.ra.deg], [sky.dec.deg], center)[0]
            b = project([later.ra.deg], [later.dec.deg], center)[0]
            # Central difference removes the O(dt) coordinate curvature.
            earlier = sky.directional_offset_by(np.arctan2(-1234,678)*u.rad, np.hypot(1234,678)*u.mas)
            c = project([earlier.ra.deg], [earlier.dec.deg], center)[0]
            out = icrs_motion(a,(b-c)/2,center)
            np.testing.assert_allclose(out['pm_icrs_masyr'], [1234,-678], atol=2e-6)

    def test_full_jacobian_and_positive_covariance(self):
        center = (350.,0.)
        position = project([355.], [5.], center)[0]
        motion = np.array([10000.,-7000.])
        out = icrs_motion(position,motion,center,np.eye(4))
        for i in range(2):
            step = np.zeros(2); step[i] = 100.
            diff = (icrs_motion(position+step,motion,center)['pm_icrs_masyr']-
                    icrs_motion(position-step,motion,center)['pm_icrs_masyr'])/200
            np.testing.assert_allclose(diff,out['state_jacobian'][[1,3],2*i],rtol=2e-5,atol=1e-13)
        self.assertGreater(np.linalg.eigvalsh(out['covariance']).min(),0)

    def test_reference_draw_covariance_reproduces_frozen_errors_and_cross_axis(self):
        time = np.array([2001.,2003.,2015.,2017.,2019.])
        sigma = np.full((len(time),2),60.)
        xy = (time-2015.5)[:,None]*np.array([4.,-2.])
        random = np.random.default_rng(72)
        g = random.normal(size=(48,2))
        draws = [g*np.array([1.,2.])*(i+1) for i in range(len(time))]
        offsets = [(np.zeros(2),d,40) for d in draws]
        model, errors = fit_with_calibration(xy,time,sigma,2015.5,offsets)
        full, independent, common, influence = calibrated_covariance(model,time,sigma,offsets)
        np.testing.assert_allclose(np.sqrt(np.diag(full)[[1,3]]),errors)
        np.testing.assert_allclose(independent+common,full)
        self.assertNotEqual(full[1,3],0)
        self.assertGreaterEqual(np.linalg.eigvalsh(full).min(),0)
        # A common constant positional shift has zero PM contribution.
        same = [(np.zeros(2),g,40)]*len(time)
        _,_,constant,_ = calibrated_covariance(model,time,sigma,same)
        np.testing.assert_allclose(constant[[1,3]][:,[1,3]],0,atol=1e-20)
