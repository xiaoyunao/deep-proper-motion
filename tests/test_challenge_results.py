import unittest

from deep_pm.challenge_results import METHODS, STAGES, summarize_bin


class ChallengeResultsTest(unittest.TestCase):
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
