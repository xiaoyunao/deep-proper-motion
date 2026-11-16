import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from deep_pm.associate import Track
from deep_pm.counterfactual import held_year


class HeldYearIndexTest(unittest.TestCase):
    @patch("deep_pm.counterfactual.associations")
    def test_held_visit_neighbours_map_to_actual_detection_rows(self, associate):
        associate.return_value = {"motion": [Track((0, 1, 2), 1., "secure", 3)]}
        time = np.array([2000., 2015., 2017., 2018.])
        xy = np.zeros((4, 2))
        data = pd.DataFrame(dict(survey=["SDSS", "NSC", "NSC", "NSC"],
                                 exposure_id=["first", "second", "held", "fourth"],
                                 band=["r"] * 4, magnitude=[19.] * 4))
        config = dict(reference_epoch=2015.5, min_seed_years=.6)
        result = held_year(data, xy, time, np.full((4, 2), 55.), np.zeros(4, dtype=int),
                           .02, config, ("motion",))
        selected = result["methods"]["motion"]["secure"]["cross_survey/with_gaia/r_lt_20p5"]
        self.assertEqual(selected["attempted"], 1)
        self.assertEqual(selected["labelled_correct"], 1)
