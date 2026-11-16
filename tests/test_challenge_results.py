import unittest

from deep_pm.challenge_results import METHODS, STAGES, parallax_class, summarize_bin


class ChallengeResultsTest(unittest.TestCase):
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
