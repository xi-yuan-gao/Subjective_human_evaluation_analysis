from __future__ import annotations

import csv
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from mos_analysis.analysis import MOSAnalysisError, run_mos_analysis
from mos_analysis.summarize import _student_t_quantile
from mos_analysis.visualize import _draw_rating_distribution


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def settings() -> dict:
    return {
        "list_ids": [1],
        "expected_participants_per_list": 2,
        "expected_conditions": ["model_a", "model_b"],
        "rating_scale": [1, 2, 3, 4, 5],
        "mos_analysis": {
            "confidence_level": 0.95,
        },
    }


def valid_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ratings = {
        "p1": {"model_a": [4, 5], "model_b": [2, 3]},
        "p2": {"model_a": [3, 4], "model_b": [1, 2]},
    }
    for participant, by_condition in ratings.items():
        for condition, values in by_condition.items():
            for item_number, rating in enumerate(values, start=1):
                rows.append(
                    {
                        "participant_id": participant,
                        "session_id": f"s_{participant}",
                        "list_id": "1",
                        "item_id": f"item_{item_number}",
                        "condition": condition,
                        "speaker_id": "speaker_1",
                        "gender": "F",
                        "mos_rating": rating,
                        "source_file": f"{participant}.csv",
                        "source_row": item_number + 1,
                    }
                )
    return rows


class MOSAnalysisTests(unittest.TestCase):
    def test_analysis_writes_condition_and_supporting_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.csv"
            output = root / "output"
            write_csv(input_path, valid_rows())

            manifest = run_mos_analysis(input_path, output, settings())

            self.assertEqual(manifest["status"], "PASSED")
            self.assertEqual(manifest["valid_score_rows"], 8)
            with (output / "02_summaries" / "mos_by_condition.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["condition"] for row in rows], ["model_a", "model_b"]
            )
            self.assertAlmostEqual(float(rows[0]["mean_mos"]), 4.0)
            self.assertAlmostEqual(float(rows[1]["mean_mos"]), 2.0)
            self.assertEqual(
                rows[0]["ci_method"],
                "listener_mean_student_t",
            )
            self.assertEqual(rows[0]["n_listener_means"], "2")
            self.assertEqual(rows[0]["degrees_of_freedom"], "1")
            self.assertFalse((output / "01_scores").exists())
            self.assertTrue(
                (output / "02_summaries" / "mos_by_listener.csv").exists()
            )
            figure_root = output / "03_figures"
            expected_figures = {
                "mos_by_condition.svg",
                "mos_rating_distribution.svg",
                "mos_by_listener_condition.svg",
                "mos_by_speaker_condition.svg",
            }
            self.assertTrue(
                expected_figures.issubset(
                    {path.name for path in figure_root.glob("*.svg")}
                )
            )
            for name in expected_figures:
                ET.parse(figure_root / name)
            with (figure_root / "mos_listener_condition_plot_data.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                listener_rows = list(csv.DictReader(handle))
            self.assertEqual(len(listener_rows), 4)
            self.assertEqual(
                {row["condition"] for row in listener_rows},
                {"model_a", "model_b"},
            )

    def test_invalid_rating_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = valid_rows()
            rows[0]["mos_rating"] = 6
            input_path = root / "input.csv"
            write_csv(input_path, rows)
            with self.assertRaises(MOSAnalysisError):
                run_mos_analysis(input_path, root / "output", settings())

    def test_duplicate_response_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = valid_rows()
            rows.append(dict(rows[0]))
            input_path = root / "input.csv"
            write_csv(input_path, rows)
            with self.assertRaises(MOSAnalysisError):
                run_mos_analysis(input_path, root / "output", settings())

    def test_nonempty_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.csv"
            output = root / "output"
            write_csv(input_path, valid_rows())
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(MOSAnalysisError):
                run_mos_analysis(input_path, output, settings())

    def test_student_t_quantile_matches_known_critical_value(self) -> None:
        known_values = {
            1: 12.70620474,
            2: 4.30265273,
            10: 2.22813885,
            39: 2.02269092,
        }
        for degrees_of_freedom, expected in known_values.items():
            with self.subTest(degrees_of_freedom=degrees_of_freedom):
                self.assertAlmostEqual(
                    _student_t_quantile(0.975, degrees_of_freedom),
                    expected,
                    places=7,
                )

    def test_rating_distribution_shows_readable_narrow_segment_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "distribution.svg"
            data = [
                {
                    "condition": "model_a",
                    "condition_display_name": "Model A",
                    "mos_rating": 4,
                    "rating_proportion": 0.93,
                },
                {
                    "condition": "model_a",
                    "condition_display_name": "Model A",
                    "mos_rating": 5,
                    "rating_proportion": 0.07,
                },
            ]

            _draw_rating_distribution(data, output, ["model_a"])

            svg = output.read_text(encoding="utf-8")
            self.assertIn(">7%</text>", svg)


if __name__ == "__main__":
    unittest.main()
