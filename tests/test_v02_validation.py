import unittest
import numpy as np

from deep_pm.validation import matched_object, truth_requirements


class RecoveryDefinitionTest(unittest.TestCase):
    def test_many_nsc_visits_cannot_hide_a_wrong_early_sdss_connection(self):
        survey = np.array(["SDSS"] * 2 + ["NSC"] * 30)
        time = np.r_[2000., 2002., 2010 + np.arange(30) / 365.25]
        labels = np.zeros(len(time), dtype=int)
        req = truth_requirements({0: np.arange(len(time))}, survey, time, 100)
        self.assertIsNone(matched_object(np.arange(1, len(time)), labels, req, .8))
        self.assertEqual(matched_object(np.arange(len(time)), labels, req, .8), 0)
        labels[-1] = 991
        self.assertIsNone(matched_object(np.arange(len(time)), labels, req, .8))

    def test_apparent_position_has_signed_parallax_and_proper_motion(self):
        from deep_pm.astrometry import apparent_positions, epoch_halo_deg

        xy = apparent_positions([0], [0], [200], [20], [100], [0], [5],
                                np.array([0, 1, 0]), [0, 0])
        np.testing.assert_allclose(xy, [[900, 100]], atol=1e-7)
        config = dict(mu_max=2000, reference_epoch=2015.5,
                      access=dict(epoch_min=1998, epoch_max=2020, edge_pad_mas=1000))
        self.assertAlmostEqual(epoch_halo_deg(config, 10000) * 3600000, 176000)

    def test_blind_association_outputs_feed_gaia_recovery_trace(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import pandas as pd

        from deep_pm.__main__ import inventory, run_association, run_fits, tomllib
        from deep_pm.validation import validate

        with open("configs/experiment-a-v02.toml", "rb") as config_file:
            cfg = tomllib.load(config_file)
        cfg["regions"]["smoke"].update(ra=30., dec=0.)
        center = (30., 0.)
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            visits = [2001., 2002., 2013., 2015., 2018.]
            pd.DataFrame(dict(survey=["SDSS"] * 2 + ["NSC"] * 3, ra_deg=np.full(5, 30.),
                              dec_deg=np.zeros(5), observation_time=51544.5 + (np.array(visits) - 2000) * 365.25,
                              sigma_east_mas=20., sigma_north_mas=20., in_core=True, band="r", magnitude=18.,
                              exposure_id=[f"visit{i}" for i in range(5)])).to_csv(work / "detections.csv", index=False)
            inventory(work, cfg, center, .02)
            run_association(work, center, cfg)
            run_fits(work, center, .02, cfg)
            (work / "validation").mkdir()
            (work / "run_manifest.json").write_text('{"region":"smoke"}')
            pd.DataFrame([dict(source_id=1, ra=30., dec=0., pmra=0., pmdec=0., parallax=1.,
                               ref_epoch=2016., ruwe=1., pmra_error=.1, pmdec_error=.1)]).to_csv(
                                   work / "validation" / "gaia.csv", index=False)
            validate(work, center, cfg)
            out = json.loads((work / "validation.json").read_text())
            self.assertEqual(out["methods"]["motion"]["secure_recall"]["n"], 1)
            self.assertEqual(out["recovery_stages"]["pm_compatible_secure"]["total"]["n"], 1)
