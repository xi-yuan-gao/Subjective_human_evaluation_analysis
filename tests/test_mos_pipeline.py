from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from mos_analysis.pipeline import MOSQCError, run_mos_pipeline
from mos_analysis.workflow import run_full_mos_analysis


CONDITIONS = ["a", "b"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def config() -> dict:
    return {
        "list_directory_pattern": "pilot_mos_list{list_id}",
        "list_ids": [1],
        "expected_participants_per_list": 2,
        "expected_ratings_per_participant": 2,
        "expected_attention_checks_per_participant": 1,
        "expected_ratings_per_stimulus_condition": 2,
        "expected_conditions": CONDITIONS,
        "rating_scale": [1, 2, 3, 4, 5],
        "column_aliases": {
            "participant_id": ["participant_id"],
            "session_id": ["session_id"],
            "native_english": ["native_english"],
            "headphones": ["headphones"],
            "heard_clearly": ["heard_clearly"],
            "item_id": ["item_id"],
            "condition": ["condition"],
            "speaker_id": ["speaker_id"],
            "mos_rating": ["mos_rating"],
            "is_attention_check": ["is_attention_check"],
            "expected_response": ["expected_response"],
            "item_order": ["item_order"],
            "response_time_sec": ["response_time_sec"],
            "study_elapsed_sec": ["study_elapsed_sec"],
            "question": ["question"],
        },
        "attention_trial_values": ["attention_check"],
        "attention_check_include_overrides": [],
        "incomplete_session_include_overrides": [],
        "eligibility_field_overrides": [],
        "technical_exclusion_overrides": [],
        "timing_qc": {
            "minimum_session_duration_sec": None,
            "maximum_session_duration_sec": None,
            "maximum_trial_response_time_sec": None,
        },
    }


def create_results(
    root: Path,
    *,
    invalid_rating: bool = False,
    failed_attention: bool = False,
    headphones: bool = True,
    native_english: bool = True,
) -> None:
    folder = root / "pilot_mos_list1"
    for participant_number in (1, 2):
        participant = f"p{participant_number}"
        session = f"s{participant_number}"
        rows: list[dict[str, object]] = []
        for index, condition in enumerate(CONDITIONS):
            rows.append(
                {
                    "participant_id": participant,
                    "session_id": session,
                    "list_id": "list1",
                    "item_order": index,
                    "item_id": f"item_{condition}",
                    "condition": condition,
                    "speaker_id": "speaker_1",
                    "question": "Rate naturalness",
                    "mos_rating": 6 if invalid_rating and participant_number == 1 and index == 0 else index + 3,
                    "is_attention_check": "FALSE",
                    "expected_response": "",
                    "response_time_sec": 4,
                    "study_elapsed_sec": (index + 1) * 5,
                }
            )
        rows.append(
            {
                "participant_id": participant,
                "session_id": session,
                "list_id": "list1",
                "item_order": 2,
                "item_id": "attention",
                "condition": "",
                "speaker_id": "",
                "question": "Select 2",
                "mos_rating": 1 if failed_attention and participant_number == 1 else 2,
                "is_attention_check": "TRUE",
                "expected_response": 2,
                "response_time_sec": 2,
                "study_elapsed_sec": 15,
            }
        )
        write_csv(folder / "main" / f"{participant}.csv", rows)
        write_csv(
            folder / "prescreen" / f"{participant}.csv",
            [
                {
                    "participant_id": participant,
                    "session_id": session,
                    "native_english": (
                        "TRUE" if native_english or participant_number == 2 else "FALSE"
                    ),
                    "headphones": "TRUE" if headphones or participant_number == 2 else "FALSE",
                }
            ],
        )
        write_csv(
            folder / "sound_check" / f"{participant}.csv",
            [
                {
                    "participant_id": participant,
                    "session_id": session,
                    "heard_clearly": "TRUE",
                }
            ],
        )
        write_csv(
            folder / "postquestionnaire" / f"{participant}.csv",
            [{"participant_id": participant, "session_id": session, "comments": ""}],
        )


class MOSPipelineTests(unittest.TestCase):
    def test_complete_run_writes_strict_r_ready_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            output = root / "output"
            create_results(results)
            manifest = run_mos_pipeline(results, output, config())
            self.assertEqual(manifest["status"], "PASSED")
            ready_path = output / "02_preprocessed" / "mos_analysis_ready.csv"
            with ready_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["mos_rating"] for row in rows}, {"3", "4"})
            self.assertIn("source_file", rows[0])
            self.assertIn("source_row", rows[0])
            self.assertNotIn("mos_rating_raw", rows[0])
            self.assertTrue((output / "01_qc" / "rating_value_audit.csv").exists())
            timing_qc = (output / "01_qc" / "session_timing_qc.csv").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("nonmonotonic_study_elapsed_sec", timing_qc)

    def test_legacy_item_id_harmonization_is_explicit_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            output = root / "output"
            create_results(results)
            settings = config()
            settings["legacy_item_id_harmonization"] = {
                "enabled": True,
                "strategy": "append_speaker_id_if_missing",
                "list_ids": ["1"],
            }

            manifest = run_mos_pipeline(results, output, settings)

            with (
                output / "02_preprocessed" / "mos_analysis_ready.csv"
            ).open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(manifest["legacy_item_id_harmonization"]["changed_rows"], 4)
            self.assertEqual(rows[0]["source_item_id"], "item_a")
            self.assertEqual(rows[0]["item_id"], "item_a_speaker_1")

    def test_full_workflow_writes_qc_summaries_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            output = root / "output"
            create_results(results)

            manifest = run_full_mos_analysis(results, output, config())

            self.assertEqual(manifest["status"], "PASSED")
            self.assertEqual(manifest["analysis_mode"], "full_pipeline")
            self.assertEqual(manifest["valid_score_rows"], 4)
            self.assertTrue((output / "01_qc" / "qc_summary.txt").exists())
            self.assertTrue(
                (output / "02_preprocessed" / "mos_analysis_ready.csv").exists()
            )
            self.assertTrue(
                (output / "03_summaries" / "mos_by_condition.csv").exists()
            )
            self.assertTrue(
                (output / "04_figures" / "mos_rating_distribution.svg").exists()
            )

    def test_invalid_rating_blocks_analysis_ready_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            output = root / "output"
            create_results(results, invalid_rating=True)
            with self.assertRaises(MOSQCError):
                run_mos_pipeline(results, output, config())
            self.assertFalse((output / "02_preprocessed" / "mos_analysis_ready.csv").exists())
            issues = (output / "01_qc" / "qc_issues.csv").read_text(encoding="utf-8")
            self.assertIn("INVALID_MOS_RATING", issues)

    def test_decimal_rating_is_not_silently_coerced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            output = root / "output"
            create_results(results)
            path = results / "pilot_mos_list1" / "main" / "p1.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["mos_rating"] = "5.0"
            write_csv(path, rows)
            with self.assertRaises(MOSQCError):
                run_mos_pipeline(results, output, config())
            issues = (output / "01_qc" / "qc_issues.csv").read_text(encoding="utf-8")
            self.assertIn("INVALID_MOS_RATING", issues)

    def test_failed_attention_excludes_participant_and_blocks_required_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            output = root / "output"
            create_results(results, failed_attention=True)
            with self.assertRaises(MOSQCError):
                run_mos_pipeline(results, output, config())
            participant_qc = (output / "01_qc" / "participant_qc.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("failed at least one attention check", participant_qc)
            self.assertIn("ELIGIBLE_PARTICIPANT_COUNT", (output / "01_qc" / "qc_issues.csv").read_text(encoding="utf-8"))

    def test_headphone_failure_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            output = root / "output"
            create_results(results, headphones=False)
            with self.assertRaises(MOSQCError):
                run_mos_pipeline(results, output, config())
            participant_qc = (output / "01_qc" / "participant_qc.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("headphones is not TRUE", participant_qc)

    def test_incomplete_session_override_retains_observed_ratings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            output = root / "output"
            create_results(results)
            path = results / "pilot_mos_list1" / "main" / "p1.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            write_csv(path, rows[1:])
            settings = config()
            settings["incomplete_session_include_overrides"] = [
                {
                    "list_id": "1",
                    "participant_id": "p1",
                    "session_id": "s1",
                    "expected_ratings": 2,
                    "observed_ratings": 1,
                    "reason": "Confirmed test interruption.",
                }
            ]

            manifest = run_mos_pipeline(results, output, settings)

            self.assertEqual(manifest["status"], "PASSED")
            self.assertEqual(manifest["analysis_ready_rows"], 3)
            self.assertEqual(manifest["incomplete_session_overrides_applied"], 1)
            issues = (output / "01_qc" / "qc_issues.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("INCOMPLETE_SESSION_OVERRIDDEN", issues)
            self.assertIn("STIMULUS_CONDITION_SHORTFALL_OVERRIDDEN", issues)
            participant_qc = (output / "01_qc" / "participant_qc.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("incomplete_session_override_applied", participant_qc)

    def test_eligibility_field_override_preserves_raw_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            output = root / "output"
            create_results(results, native_english=False)
            settings = config()
            settings["eligibility_field_overrides"] = [
                {
                    "list_id": "1",
                    "participant_id": "p1",
                    "session_id": "s1",
                    "field": "native_english",
                    "corrected_value": True,
                    "reason": "Participant confirmed a selection typo.",
                }
            ]

            manifest = run_mos_pipeline(results, output, settings)

            self.assertEqual(manifest["status"], "PASSED")
            self.assertEqual(manifest["analysis_ready_rows"], 4)
            self.assertEqual(manifest["eligibility_field_overrides_applied"], 1)
            with (
                output / "01_qc" / "eligibility_field_override_audit.csv"
            ).open(encoding="utf-8", newline="") as handle:
                audit = list(csv.DictReader(handle))
            self.assertEqual(len(audit), 1)
            self.assertEqual(audit[0]["raw_value"], "False")
            self.assertEqual(audit[0]["effective_value"], "True")
            self.assertEqual(audit[0]["override_applied"], "True")

    def test_incomplete_override_with_wrong_count_does_not_apply(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            output = root / "output"
            create_results(results)
            path = results / "pilot_mos_list1" / "main" / "p1.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            write_csv(path, rows[1:])
            settings = config()
            settings["incomplete_session_include_overrides"] = [
                {
                    "list_id": "1",
                    "participant_id": "p1",
                    "session_id": "s1",
                    "expected_ratings": 2,
                    "observed_ratings": 0,
                    "reason": "Incorrect declared count.",
                }
            ]

            with self.assertRaises(MOSQCError):
                run_mos_pipeline(results, output, settings)

            issues = (output / "01_qc" / "qc_issues.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("INCOMPLETE_OVERRIDE_COUNT_MISMATCH", issues)


if __name__ == "__main__":
    unittest.main()
