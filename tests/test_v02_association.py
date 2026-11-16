import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from deep_pm.associate import Track, choose_tracks, motion_tracks, static_tracks


class ConflictRegressionTest(unittest.TestCase):
    def test_size_budget_does_not_skip_component(self):
        tracks = [Track((0, 1), 7, "secure", 3), Track((0, 2), 2, "secure", 3)]
        chosen, failed, used = choose_tracks(tracks, 1)
        self.assertEqual(failed, 0)
        self.assertEqual(used, {0, 1})

    @patch("deep_pm.associate.milp")
    def test_timeout_uses_only_a_feasible_tentative_incumbent(self, milp):
        tracks = [Track((0, 1), 7, "secure", 3), Track((0, 2), 2, "secure", 3)]
        milp.return_value = SimpleNamespace(status=1, success=False, x=np.array([1., 0.]),
                                            mip_gap=.1)
        chosen, _, _ = choose_tracks(tracks, 100)
        self.assertEqual([x.status for x in chosen], ["tentative_solver"])
        milp.return_value.x = np.array([1., 1.])
        chosen, failed, _ = choose_tracks(tracks, 100)
        self.assertEqual(chosen, [])
        self.assertEqual(failed, 1)

    def test_background_dates_cannot_erase_track_time_support(self):
        time = 2000 + np.arange(9) / 4
        positions = np.column_stack([time * 0, np.full(len(time), 100000.)])
        truth = [0, 4, 8]
        positions[truth] = 0
        exp = np.arange(len(time))
        sigma = np.full(positions.shape, 20.)
        config = dict(min_season_days=100, reference_epoch=2001, mu_max=2000, match_sigma=4,
                      model_margin_mas=100, min_seed_years=.6, max_seed_pairs=500, max_component=100)
        static = static_tracks(positions, exp, 1000)
        chosen, _, _, _ = motion_tracks(positions, time, sigma, exp, static, config)
        self.assertEqual(next(x.groups for x in chosen if x.indices == tuple(truth)), 3)

    def test_interleaved_fragments_have_lower_total_score(self):
        # Both three-detection subtracks cover three independent time groups.
        time = np.array([2000., 2000.01, 2002., 2002.01, 2004., 2004.01])
        positions = np.zeros((6, 2))
        exp = np.arange(6)
        config = dict(min_season_days=100, reference_epoch=2002, mu_max=2000, match_sigma=4,
                      model_margin_mas=100, min_seed_years=.6, max_seed_pairs=500, max_component=100)
        groups = [(0, 2, 4), (1, 3, 5), tuple(range(6))]
        chosen, _, _, _ = motion_tracks(positions, time, np.full(positions.shape, 20.),
                                         exp, groups, config)
        self.assertEqual([x.indices for x in chosen], [tuple(range(6))])

    def test_more_visits_do_not_inflate_local_background(self):
        from deep_pm.associate import exposure_density

        xy = np.array([[0, 0], [3000, 0]])
        sample = exposure_density(xy, [0, 0])
        repeated = exposure_density(np.tile(xy, (20, 1)), np.repeat(np.arange(20), 2))
        np.testing.assert_allclose(repeated, np.tile(sample, 20))

    def test_near_duplicates_merge_but_same_visit_alternatives_survive(self):
        from deep_pm.associate import merge_candidates

        model = SimpleNamespace(coeff=np.zeros((2, 2)))
        first = Track(tuple(range(9)), 10, "secure", 4)
        second = Track(tuple(range(1, 10)), 9, "secure", 4)
        third = Track(tuple(range(8)) + (10,), 8, "secure", 4)

        def build(i):
            return Track(i, len(i) * 1.1, "secure", 4), model

        kept, mapping = merge_candidates([first, second, third], [model] * 3, build,
                                          np.r_[np.arange(10), 8], {"merge_overlap": .75})
        self.assertEqual(kept[0].indices, tuple(range(10)))
        self.assertEqual(kept[1], third)
        self.assertEqual(mapping, [0, 0, 1])
