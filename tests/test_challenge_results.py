from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from deep_pm.challenge_results import METHODS, STAGES, empty_survey_returns, parallax_class, summarize_bin
from deep_pm.io import digest


class ChallengeResultsTest(unittest.TestCase):
    def test_zero_row_fetch_must_be_verified(self):
        with TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "sdss.csv").write_text("objid,ra\n")
            (base / "nsc_object.csv").write_text("measid,ra\na,20.0\n")
            stages = {"fetch-sdss": [dict(state="completed", outputs={"sdss.csv": digest(base / "sdss.csv")})],
                      "fetch-object": [dict(state="completed")]}
            self.assertEqual(empty_survey_returns(base, stages), ["SDSS"])
            (base / "sdss.csv").write_text("objid,dec\n")
            with self.assertRaisesRegex(ValueError, "changed after verified retrieval"):
                empty_survey_returns(base, stages)

    def test_target_parallax_uses_an_exact_catalogue_identifier(self):
        gaia = b"source_id,parallax\n6000000000000001,1\n6000000000000002,-12\n6000000000000003,\n"
        self.assertEqual(parallax_class(gaia, "6000000000000001", 10), "small_abs_parallax")
        self.assertEqual(parallax_class(gaia, "6000000000000002", 10), "large_abs_parallax")
        self.assertEqual(parallax_class(gaia, "6000000000000003", 10), "no_measured_parallax")

    def test_retains_frozen_and_eligible_denominators(self):
        def validation(eligible):
            methods = {method: dict(pm_bins=[dict(low=200, high=500, denominator=eligible)],
                                    selected_recall={"n": eligible if method == "motion" else 0})
                       for method in METHODS}
            stages = {step: dict(total={"n": eligible}) for step in STAGES}
            return dict(truth_cross_survey=eligible, methods=methods, recovery_stages=stages)

        summary = summarize_bin(4, [validation(1), validation(0)], 200, 500)
        self.assertEqual(summary["measured"], 2)
        self.assertEqual(summary["unknown_truth_status"], 2)
        self.assertEqual(summary["eligible"]["n"], 1)
        self.assertEqual(summary["selected"]["motion"]["among_eligible"]["value"], 1)
        self.assertEqual(summary["selected"]["motion"]["among_frozen"]["value"], .25)
        self.assertEqual(summary["selected"]["fixed"]["among_eligible"]["value"], 0)
        with self.assertRaisesRegex(ValueError, "outside its frozen speed bin"):
            summarize_bin(4, [validation(1)], 500, 1000)
