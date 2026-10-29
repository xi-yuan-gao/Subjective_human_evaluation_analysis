from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from mos_analysis.mixed_effects import _fit_model, _prepare_data, build_parser


DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ("numpy", "pandas", "scipy", "statsmodels")
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_rows() -> list[dict[str, object]]:
    """Create balanced ordinal ratings with a clear condition effect."""
    conditions = ["model_a", "model_b", "model_c", "model_d"]
    condition_effect = {"model_a": 0, "model_b": 1, "model_c": 2, "model_d": 3}
    rows: list[dict[str, object]] = []
    for listener_number in range(12):
        for item_number in range(16):
            condition = conditions[(listener_number + item_number) % 4]
            latent = (
                1.2
                + condition_effect[condition]
                + 0.35 * (listener_number % 3)
                + 0.25 * (item_number % 4)
            )
            rating = min(5, max(1, round(latent)))
            rows.append(
                {
                    "participant_id": f"p{listener_number}",
                    "session_id": f"s{listener_number}",
                    "item_id": f"item_{item_number}",
                    "condition": condition,
                    "speaker_id": f"speaker_{item_number % 2}",
                    "mos_rating": rating,
                }
            )
    return rows


class MOSMixedEffectsTests(unittest.TestCase):
    def test_parser_requires_input_and_output(self) -> None:
        args = build_parser().parse_args(
            ["--input", "ratings.csv", "--output-root", "model_output"]
        )
        self.assertEqual(args.input, Path("ratings.csv"))
        self.assertEqual(args.output_root, Path("model_output"))
        self.assertIsNone(args.compare_conditions)

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "mixed-effects extras not installed")
    def test_invalid_rating_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ratings.csv"
            rows = synthetic_rows()
            rows[0]["mos_rating"] = 6
            write_csv(path, rows)
            with self.assertRaisesRegex(ValueError, "whole-number values"):
                _prepare_data(path, ["model_a", "model_b", "model_c", "model_d"])

    @unittest.skipUnless(DEPENDENCIES_AVAILABLE, "mixed-effects extras not installed")
    def test_omnibus_and_selected_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ratings.csv"
            write_csv(path, synthetic_rows())
            data = _prepare_data(
                path, ["model_a", "model_b", "model_c", "model_d"]
            )
            condition_test, _, adjusted, audit, comparison = _fit_model(
                data, ("model_d", "model_a")
            )

        self.assertTrue(condition_test["model_converged"])
        self.assertEqual(condition_test["degrees_of_freedom"], 3)
        self.assertLess(condition_test["p_value"], 0.05)
        self.assertTrue(audit["listener_random_intercept"])
        self.assertTrue(audit["item_random_intercept"])
        self.assertEqual(
            [row["condition"] for row in adjusted],
            ["model_a", "model_b", "model_c", "model_d"],
        )
        self.assertGreater(
            adjusted[-1]["adjusted_mean_mos"], adjusted[0]["adjusted_mean_mos"]
        )
        self.assertIsNotNone(comparison)
        assert comparison is not None
        self.assertGreater(comparison["estimated_difference_mos_points"], 0)
        self.assertLess(comparison["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
