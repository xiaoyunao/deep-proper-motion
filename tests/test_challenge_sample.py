import unittest

import pandas as pd

from deep_pm.challenge_sample import choose


class ChallengeSeparationTest(unittest.TestCase):
    def test_seen_fields_and_cross_split_neighbours_are_excluded(self):
        config = dict(seed=22, access=dict(halo_deg=.02),
                      regions=dict(legacy=dict(ra=359.98, dec=0., radius=.02,
                                               role="historical_challenge")),
                      challenge=dict(pm_edges=[100, 200], radius=.02,
                                     minimum_separation_deg=.08, sources_per_bin_per_split=5))
        pool = pd.DataFrame(dict(source_id=[101, 102, 103, 104], pm=[150.] * 4,
                                 ra=[.01, 2., 2.01, 4.], dec=[0.] * 4,
                                 split=["dev", "dev", "test", "test"]))
        sample = choose(pool, config)
        self.assertEqual(list(sample.source_id), ["102", "104"])
        self.assertEqual(list(sample.split), ["dev", "test"])
