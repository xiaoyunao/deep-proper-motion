import unittest

from deep_pm.field_statistics import block_fraction, optical_summary


class FieldStatisticsTest(unittest.TestCase):
    def test_empty_regions_do_not_create_independent_evidence(self):
        res = block_fraction([(1, 2), (0, 0), (0, 0)])
        self.assertEqual(res["field_blocks"], 1)
        self.assertIsNone(res["field_bootstrap95"])

    def test_bins_and_cohorts_stay_separate(self):
        def probe(name, cohort, commit):
            positive = dict(n=1, denominator=2)
            empty = dict(n=0, denominator=0)
            method = {key: positive for key in ("candidate_recall", "selected_recall",
                      "secure_recall", "ambiguous_recall", "unresolved_fraction")}
            method["pm_bins"] = [dict(low=100, high=200, candidate=positive, secure=positive,
                                      **positive),
                                 dict(low=200, high=500, candidate=empty, secure=empty, **empty)]
            method["object_confirmation"] = dict(discordant=1, confirmed=1, unknown=1)
            return dict(region=name, role=cohort, receipt=dict(commit=commit, config_sha256="x"),
                        validation={"methods": dict.fromkeys(("fixed", "nsc_like", "motion"), method)})

        a = probe("holdout_a", "holdout", "first")
        b = probe("holdout_b", "holdout", "first")
        control = probe("development", "development", "second")
        result = optical_summary([a, b, control])
        first = result["holdout"]["motion"]["proper_motion_bins"][0]
        self.assertEqual(first["selected"]["denominator"], 4)
        self.assertEqual(first["secure"]["field_blocks"], 2)
        self.assertIsNone(result["holdout"]["motion"]["proper_motion_bins"][1]["candidate"]["value"])
        self.assertEqual(result["development"]["motion"]["selected_recall"]["denominator"], 2)
        with self.assertRaisesRegex(ValueError, "Mixed code or configuration"):
            optical_summary([a, probe("holdout_c", "holdout", "second")])
