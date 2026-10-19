from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from transcription_analysis.mixed_effects import (
    _fit_outcome,
    _fit_outcome_with_pair,
    _prepare_data,
    build_parser,
)


try:
    import numpy as np
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    import statsmodels  # noqa: F401

    HAS_MIXED_EFFECTS = True
except ImportError:
    HAS_MIXED_EFFECTS = False


class MixedEffectsTests(unittest.TestCase):
    @unittest.skipUnless(HAS_MIXED_EFFECTS, "mixed-effects optional dependencies not installed")
    def test_joint_condition_test_uses_crossed_random_intercepts(self):
        generator = np.random.default_rng(21)
        conditions = ["a", "b", "c"]
        rows = []
        listener_effects = generator.normal(0, 0.03, 10)
        utterance_effects = generator.normal(0, 0.03, 12)
        for listener in range(10):
            for utterance in range(12):
                condition = conditions[(listener + utterance) % len(conditions)]
                condition_effect = {"a": 0.00, "b": 0.18, "c": 0.36}[condition]
                outcome = (
                    0.08
                    + condition_effect
                    + listener_effects[listener]
                    + utterance_effects[utterance]
                    + generator.normal(0, 0.025)
                )
                rows.append(
                    {
                        "participant_id": f"p{listener}",
                        "session_id": f"s{listener}",
                        "utterance_id": f"u{utterance}",
                        "condition": condition,
                        "speaker_id": f"speaker_{utterance % 2}",
                        "wer": max(0.0, outcome),
                        "cer": max(0.0, outcome * 0.8),
                    }
                )

        with tempfile.TemporaryDirectory() as temp:
            scores = Path(temp) / "scores.csv"
            with scores.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            data = _prepare_data(scores, conditions)
            test_row, audit = _fit_outcome(data, "wer")

        self.assertEqual(test_row["degrees_of_freedom"], 2)
        self.assertTrue(test_row["model_converged"])
        self.assertLess(test_row["p_value"], 0.05)
        self.assertTrue(audit["speaker_fixed_effect"])
        self.assertTrue(audit["listener_random_intercept"])
        self.assertTrue(audit["utterance_random_intercept"])
        self.assertIn("residual_skewness", audit)
        self.assertEqual(set(audit["residual_sd_by_condition"]), set(conditions))

    @unittest.skipUnless(HAS_MIXED_EFFECTS, "mixed-effects optional dependencies not installed")
    def test_selected_condition_comparison_reports_first_minus_second(self):
        generator = np.random.default_rng(32)
        conditions = ["a", "b", "c"]
        rows = []
        for listener in range(8):
            for utterance in range(12):
                condition = conditions[(listener + utterance) % len(conditions)]
                condition_effect = {"a": 0.00, "b": 0.15, "c": 0.35}[condition]
                outcome = 0.10 + condition_effect + generator.normal(0, 0.015)
                rows.append(
                    {
                        "participant_id": f"p{listener}",
                        "session_id": f"s{listener}",
                        "utterance_id": f"u{utterance}",
                        "condition": condition,
                        "speaker_id": f"speaker_{utterance % 2}",
                        "wer": max(0.0, outcome),
                        "cer": max(0.0, outcome * 0.8),
                    }
                )

        with tempfile.TemporaryDirectory() as temp:
            scores = Path(temp) / "scores.csv"
            with scores.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            data = _prepare_data(scores, conditions)
            _, _, comparison = _fit_outcome_with_pair(data, "wer", ("b", "c"))

        self.assertIsNotNone(comparison)
        assert comparison is not None
        self.assertEqual(comparison["first_condition"], "b")
        self.assertEqual(comparison["second_condition"], "c")
        self.assertLess(comparison["estimated_difference"], 0)
        self.assertAlmostEqual(comparison["estimated_difference"], -0.20, delta=0.03)
        self.assertLess(comparison["p_value"], 0.05)
        self.assertLess(comparison["ci_95_upper"], 0)

    def test_pairwise_cli_option_accepts_exactly_two_conditions(self):
        args = build_parser().parse_args(
            ["--compare-conditions", "mantis_3.2", "whisper_audio"]
        )
        self.assertEqual(
            args.compare_conditions, ["mantis_3.2", "whisper_audio"]
        )


if __name__ == "__main__":
    unittest.main()
