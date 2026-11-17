import unittest

import numpy as np

from deep_pm.overlap_blocks import MODELS, blocked


class OverlapBlockTest(unittest.TestCase):
    def test_shared_observation_night_prevents_independent_source_intervals(self):
        records = [
            dict(exposures=[f"SDSS:{i}", f"visit_{i}"], nsc_epoch_days=[1],
                 motion=dict(C0=[3., 0.], C1=[2., 1.], C1b=[2., 0.]))
            for i in range(3)
        ]
        opts = dict(resample_draws=40, min_independent_blocks=3)
        strict = blocked(records, MODELS, True, opts, 10)
        exposure = blocked(records, MODELS, False, opts, 10)
        assert strict["independent_blocks"] == 1
        assert "gain95_vs" not in strict
        assert exposure["independent_blocks"] == 3
        np.testing.assert_allclose(exposure["axes_zero95"][2], [[2, 2], [0, 0]])
        np.testing.assert_allclose(exposure["gain95_vs"]["C0"], [1/3, 1/3])
