import unittest

import numpy as np

from deep_pm.chromatic import chromatic_fit, zenith_vector


class ChromaticTest(unittest.TestCase):
    def test_zenith_and_direction(self):
        vector, cos_z = zenith_vector(np.deg2rad(32), np.array([0., .25]), np.deg2rad([32, 32]))
        np.testing.assert_allclose(vector[0], [0., 0.], atol=1e-12)
        self.assertGreater(vector[1, 0], 0)
        np.testing.assert_allclose(cos_z[0], 1.)

    def test_joint_directional_slope_survives_source_and_group_offsets(self):
        n, g = 10, 6
        rng = np.random.default_rng(8)
        colors = np.repeat(np.arange(n)-4, g)
        angle = np.tile(np.linspace(-.7,.8,g), n)
        feature = np.column_stack([colors*np.sin(angle), colors*np.cos(angle)])[:, :, None]
        xy = np.repeat(rng.normal(size=(n,2))*300, g, axis=0)
        xy += np.tile(rng.normal(size=(g,2))*30, (n,1))
        xy += 25*feature[:, :, 0]
        beta, _, condition = chromatic_fit(np.repeat(np.arange(n),g), np.tile(np.arange(g),n),
                                           xy, np.full_like(xy,30.), feature, 10, 4)
        np.testing.assert_allclose(beta, [25.], atol=1e-7)
        self.assertAlmostEqual(condition, 1.)

    def test_constant_color_term_is_absorbed_by_source_positions(self):
        refs = np.repeat(np.arange(6), 4)
        groups = np.tile(np.arange(4), 6)
        feature = np.repeat(refs[:, None, None], 2, axis=1)
        with self.assertRaisesRegex(ValueError, "confounded"):
            chromatic_fit(refs, groups, np.zeros((24,2)), np.ones((24,2)), feature, 5, 1)
