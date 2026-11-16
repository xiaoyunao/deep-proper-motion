import unittest

import numpy as np

from deep_pm.associate import Track, choose_tracks, motion_tracks, static_tracks


class AssociationTest(unittest.TestCase):
    def setUp(self):
        self.config = dict(min_season_days=100, reference_epoch=2015, mu_max=2000,
                           match_sigma=4, model_margin_mas=100, min_seed_years=.6,
                           max_seed_pairs=500, max_component=100)

    def test_fast_source_can_reuse_static_members(self):
        t = np.array([2005., 2010., 2015., 2020.])
        xy = (t - 2015)[:, None] * np.array([[1600, 1200]])
        exposure = np.arange(4)
        static = static_tracks(xy, exposure, 1000)
        self.assertEqual(len(static), 4)
        selected, _, unmatched, _ = motion_tracks(xy, t, np.full(xy.shape, 20), exposure,
                                                   static, self.config)
        self.assertEqual([x.indices for x in selected], [(0, 1, 2, 3)])
        self.assertEqual(unmatched, [])

    def test_two_groups_never_secure(self):
        t = np.array([2005., 2018.])
        xy = np.array([[0, 0], [0, 12000]])
        selected, _, _, _ = motion_tracks(xy, t, np.full(xy.shape, 50), np.arange(2),
                                           [(0,), (1,)], self.config)
        self.assertEqual(selected[0].status, "insufficient_time_information")

    def test_shared_detection(self):
        tracks = [Track((0, 1, 2), 7, "secure", 3),
                  Track((0, 3, 4), 6.9, "secure", 3),
                  Track((5, 6), 1, "insufficient_time_information", 2)]
        selected, failures, used = choose_tracks(tracks, 100)
        self.assertEqual(failures, 0)
        self.assertEqual(used, {0, 1, 2, 5, 6})
        self.assertEqual(selected[0].status, "ambiguous")

    def test_crossing_tracks_do_not_share_one_detection(self):
        t = np.array([2000., 2000., 2005., 2010., 2010., 2015., 2015.])
        xy = np.column_stack([[-5000., 5000., 0., 5000., -5000., 10000., -10000.], np.zeros(7)])
        exposure = np.array([0, 0, 1, 2, 2, 3, 3])
        static = static_tracks(xy, exposure, 1000)
        selected, _, _, _ = motion_tracks(xy, t, np.full(xy.shape, 15), exposure,
                                           static, self.config)
        members = [i for track in selected for i in track.indices]
        self.assertEqual(len(members), len(set(members)))
