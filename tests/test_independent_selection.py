import unittest
from deep_pm.independent_selection import intersects_previous

class IndependentSelectionTest(unittest.TestCase):
    def test_excludes_previous_circle_and_rectangular_processing_halo(self):
        box=dict(ra=[132.,134.],dec=[-1.,1.])
        self.assertTrue(intersects_previous(box,[dict(ra=131.8,dec=.48,radius=.6)]))
        self.assertTrue(intersects_previous(box,[dict(ra=[134.,136.],dec=[-1.,1.])]))
        self.assertFalse(intersects_previous(box,[dict(ra=140.1,dec=.45,radius=.67)]))

    def test_conservative_spherical_bounds_handle_ra_wrap_and_high_dec(self):
        box=dict(ra=[0.,2.],dec=[60.,62.])
        self.assertTrue(intersects_previous(box,[dict(ra=359.8,dec=61.,radius=.5)]))
        self.assertFalse(intersects_previous(box,[dict(ra=355.,dec=61.,radius=.5)]))
