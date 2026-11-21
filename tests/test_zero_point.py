import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from deep_pm.precision_v03 import astrometric_metrics, fit_with_calibration
from deep_pm.zero_point import fit_offsets, lsmr


class ZeroPointTest(unittest.TestCase):
    def test_offsets_recover_relative_zero_point_with_noncontiguous_input_index(self):
        refs = pd.Series(np.repeat(np.arange(12), 3), index=np.arange(30, 66))
        groups = np.tile(["SDSS:run", "NSC:one", "NSC:two"], 12)
        imposed = np.array([[75., -80.], [0., 3.], [0., 0.]])
        data = np.repeat(np.arange(12)[:, None] * np.array([[1000., -750.]]), 3, axis=0)
        data += np.tile(imposed, (12, 1))
        offsets = fit_offsets(refs, data, np.full(data.shape, 40.), groups, 10, 8, 1)
        parallel = fit_offsets(refs, data, np.full(data.shape, 40.), groups, 10, 8, 1, workers=3)
        for group, expected in zip(groups[:3], imposed):
            np.testing.assert_allclose(offsets[group][0], expected, atol=1e-5)
            np.testing.assert_array_equal(offsets[group][1], parallel[group][1])
        t = np.array([2000., 2015., 2017.])
        pos = imposed + (t - 2000)[:, None] * np.array([[2., -6.]])
        model, _ = fit_with_calibration(pos, t, np.full(pos.shape, 50.), 2015,
                                        [offsets[g] for g in groups[:3]])
        np.testing.assert_allclose(model.coeff[1], [2., -6.], atol=1e-6)

    def test_disconnected_surveys_have_no_common_gauge(self):
        refs = np.repeat(np.arange(10), 2)
        groups = ["SDSS:one", "SDSS:two"] * 5 + ["NSC:one", "NSC:two"] * 5
        self.assertEqual(fit_offsets(refs, np.zeros((20, 2)), np.ones((20, 2)), groups, 3, 8, 1), {})

    def test_reference_resume_reuses_robust_fit_and_completed_draws(self):
        rng = np.random.default_rng(5)
        refs = np.repeat(np.arange(24), 3)
        groups = np.tile(["SDSS:run", "NSC:first", "NSC:last"], 24)
        xy = rng.normal(0, 70, (72, 2))
        args = (refs, xy, np.full_like(xy, 50.), groups, 10, 8, 9137)
        expected = fit_offsets(*args)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            fit_offsets(*args, checkpoint=checkpoint, workers=2)
            for number in (3, 7):
                (checkpoint/f"draw-{number:03d}.npz").unlink()
            with patch("deep_pm.zero_point.lsmr", wraps=lsmr) as solver:
                resumed = fit_offsets(*args, checkpoint=checkpoint)
            self.assertEqual(solver.call_count, 4)
            for group in expected:
                for original, recovered in zip(expected[group], resumed[group]):
                    np.testing.assert_array_equal(original, recovered)
            (checkpoint/"draw-004.npz").write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "Changed reference checkpoint"):
                fit_offsets(*args, checkpoint=checkpoint)

    def test_absolute_error_exposes_bias_when_scatter_is_zero(self):
        result = astrometric_metrics([[0., -4.4]] * 5, [[1., 1.]] * 5, 5)
        np.testing.assert_equal(result["nmad"], [0., 0.])
        np.testing.assert_equal(result["abs68"], [0., 4.4])
        np.testing.assert_equal(result["fraction_within_2sigma"], [1., 0.])

    def test_overlap_constraints_pull_relative_field_difference_in_correct_direction(self):
        groups = ["SDSS:run|1", "NSC:visit"] * 8 + ["SDSS:run|2", "NSC:visit"] * 8
        refs = np.repeat(np.arange(16), 2)
        positions = np.repeat(np.arange(16)[:, None] * np.array([[800., -300.]]), 2, axis=0)
        sigma = np.full_like(positions, 40.)
        first, second = groups[0], groups[16]
        edges = pd.DataFrame(dict(ref=np.arange(8), first=first, second=second,
                                  east=25., north=-15., sigma_e=8., sigma_n=8., weight=1.))
        parent = {first: "SDSS:run", second: "SDSS:run"}
        base = fit_offsets(refs, positions, sigma, groups, 5, 8, 1, parents=parent)
        fit = fit_offsets(refs, positions, sigma, groups, 5, 8, 1, parents=parent, overlaps=edges)
        np.testing.assert_allclose(base[first][0] - base[second][0], [0, 0])
        np.testing.assert_allclose(fit[first][0] - fit[second][0], [25, -15], atol=1)
        self.assertGreater(np.std(fit[first][1] - fit[second][1]), 0)
