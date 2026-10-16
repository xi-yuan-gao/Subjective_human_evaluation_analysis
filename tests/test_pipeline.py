from __future__ import annotations

import csv
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from transcription_analysis.pipeline import QCError, run_pipeline


CONDITIONS = ["a", "b", "c", "d", "e"]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def config() -> dict:
    return {
        "list_directory_pattern": "pilot_transcription_experiment_list{list_id}",
        "list_ids": [1, 2, 3, 4, 5],
        "expected_participants_per_list": 3,
        "expected_ratings_per_utterance_condition": 3,
        "expected_main_trials_per_participant": 5,
        "expected_attention_checks_per_participant": 1,
        "expected_conditions": CONDITIONS,
        "column_aliases": {
            "participant_id": ["participant_id"],
            "session_id": ["session_id"],
            "utterance_id": ["utterance_id", "item_id"],
            "condition": ["condition"],
            "speaker_id": ["speaker_id", "speaker"],
            "reference_text": ["reference_text"],
            "transcription": ["transcription"],
            "is_attention_check": ["is_attention_check"],
            "attention_check_passed": ["attention_check_passed"],
            "expected_response": ["expected_response"],
            "item_order": ["item_order"],
            "plays_used": ["plays_used"],
            "response_time_sec": ["response_time_sec"],
            "study_elapsed_sec": ["study_elapsed_sec"],
            "submitted_at": ["submitted_at"],
            "gender": ["gender"],
            "native_english": ["native_english"],
            "headphones": ["headphones"],
            "heard_clearly": ["heard_clearly"],
            "difficulty": ["difficulty"],
            "audio_quality": ["audio_quality"],
            "playback_difficulty": ["playback_difficulty"],
            "comments": ["comments"],
        },
        "attention_trial_values": ["attention_check"],
        "attention_check_include_overrides": [],
        "technical_exclusion_overrides": [],
        "timing_qc": {
            "minimum_session_duration_sec": 50,
            "maximum_session_duration_sec": 120,
            "maximum_trial_response_time_sec": 30,
            "two_play_threshold": 2,
        },
        "normalization": {
            "lowercase": True, "unicode_nfkc": True, "standardize_apostrophes": True,
            "remove_apostrophes": True, "remove_punctuation": True,
            "collapse_whitespace": True, "cer_remove_spaces": True,
            "word_map": {}, "excluded_name_phrases": [],
        },
        "questionnaire": {
            "categorical_columns": ["difficulty", "audio_quality", "playback_difficulty"],
            "free_text_columns": ["comments"],
        },
        "visualization": {
            "enabled": True,
            "bootstrap_samples": 100,
            "random_seed": 123,
            "condition_order": CONDITIONS,
            "condition_display_names": {condition: condition.upper() for condition in CONDITIONS},
            "gender_order": ["F", "M"],
            "gender_display_names": {"F": "Female speakers", "M": "Male speakers"},
            "speaker_order": ["speaker_1"],
            "speaker_display_names": {"speaker_1": "Speaker 1"},
        },
        "fail_on_structural_qc": True,
    }


def create_complete_results(
    root: Path,
    invalid_participant: tuple[int, int] | None = None,
    failed_attention: tuple[int, int] | None = None,
) -> None:
    for list_id in range(1, 6):
        folder = root / f"pilot_transcription_experiment_list{list_id}"
        for participant_number in range(1, 4):
            participant = f"p{list_id}_{participant_number}"
            session = f"s{list_id}_{participant_number}"
            main_rows = []
            for utterance_number in range(5):
                condition = CONDITIONS[(utterance_number + list_id - 1) % 5]
                main_rows.append(
                    {
                        "participant_id": participant,
                        "session_id": session,
                        "list_id": f"list{list_id}",
                        "item_order": utterance_number,
                        "item_id": f"u{utterance_number + 1}",
                        "condition": condition,
                        "speaker_id": f"speaker{utterance_number % 2 + 1}",
                        "gender": "F" if utterance_number % 2 else "M",
                        "reference_text": "The quick fox",
                        "transcription": "the quick fox" if participant_number < 3 else "the xxx fox",
                        "is_attention_check": "FALSE",
                        "attention_check_passed": "",
                        "expected_response": "",
                        "response_time_sec": 5 + utterance_number,
                        "study_elapsed_sec": 10 * (utterance_number + 1),
                        "submitted_at": f"2026-10-12T12:00:{10 * (utterance_number + 1):02d}+00:00",
                        "plays_used": 1 if utterance_number % 2 == 0 else 2,
                    }
                )
            main_rows.append(
                {
                    "participant_id": participant,
                    "session_id": session,
                    "list_id": f"list{list_id}",
                    "item_order": 5,
                    "item_id": "attention",
                    "condition": "attention",
                    "speaker_id": "",
                    "reference_text": "attention",
                    "transcription": "wrong" if failed_attention == (list_id, participant_number) else "type blue.",
                    "is_attention_check": "TRUE",
                    "attention_check_passed": "",
                    "expected_response": "Type BLUE",
                    "response_time_sec": 4,
                    "study_elapsed_sec": 59,
                    "submitted_at": "2026-10-12T12:00:59+00:00",
                    "plays_used": 1,
                }
            )
            write_csv(folder / "main" / f"{participant}.csv", main_rows)
            eligible = invalid_participant != (list_id, participant_number)
            write_csv(
                folder / "prescreen" / f"{participant}.csv",
                [{"participant_id": participant, "session_id": session, "native_english": "TRUE" if eligible else "FALSE", "headphones": "TRUE"}],
            )
            write_csv(
                folder / "sound_check" / f"{participant}.csv",
                [{"participant_id": participant, "session_id": session, "heard_clearly": "TRUE"}],
            )
            write_csv(
                folder / "postquestionnaire" / f"{participant}.csv",
                [{"participant_id": participant, "session_id": session, "difficulty": "Easy", "audio_quality": "Good", "playback_difficulty": "No", "comments": ""}],
            )


class PipelineTests(unittest.TestCase):
    def test_complete_counterbalanced_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            output = root / "analysis_output"
            create_complete_results(results)
            manifest = run_pipeline(results, output, config())
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["eligible_participants"], 15)
            self.assertEqual(manifest["score_rows"], 75)
            self.assertTrue(manifest["qc_summary"].startswith("QC PASSED:"))
            summary = (output / "01_qc" / "qc_summary.txt").read_text(encoding="utf-8")
            self.assertIn("15/15 required participant slots are filled", summary)
            self.assertIn("25/25 utterance-condition pairs are complete", summary)
            self.assertTrue((output / "OUTPUT_GUIDE.md").exists())
            self.assertTrue((output / "01_qc" / "duplicate_transcriptions.csv").exists())
            self.assertTrue((output / "02_preprocessed" / "eligible_transcriptions.csv").exists())
            self.assertTrue((output / "02_preprocessed" / "normalized_transcriptions.csv").exists())
            self.assertTrue((output / "03_scores" / "transcription_scores.csv").exists())
            self.assertTrue((output / "03_scores" / "xxx_diagnostics.csv").exists())
            self.assertTrue((output / "04_summaries" / "by_condition.csv").exists())
            self.assertTrue((output / "04_summaries" / "plays_wer_summary.csv").exists())
            self.assertTrue((output / "04_summaries" / "by_gender_condition.csv").exists())
            self.assertTrue((output / "05_questionnaire" / "eligible_responses.csv").exists())
            self.assertTrue((output / "06_figures" / "condition_plot_data.csv").exists())
            self.assertTrue((output / "06_figures" / "gender_condition_plot_data.csv").exists())
            self.assertTrue((output / "06_figures" / "speaker_condition_plot_data.csv").exists())
            self.assertTrue((output / "06_figures" / "condition_error_rates.svg").exists())
            self.assertTrue((output / "06_figures" / "gender_error_rates.svg").exists())
            self.assertTrue((output / "06_figures" / "speaker_error_rates.svg").exists())
            self.assertTrue((output / "06_figures" / "error_composition.svg").exists())
            self.assertTrue((output / "06_figures" / "xxx_rates.svg").exists())
            self.assertEqual(len(manifest["figures"]), 5)
            for figure in manifest["figures"]:
                path = output / "06_figures" / figure
                ET.parse(path)
                self.assertNotIn("nan", path.read_text(encoding="utf-8").lower())
            with (output / "06_figures" / "condition_plot_data.csv").open(encoding="utf-8") as handle:
                plot_rows = list(csv.DictReader(handle))
            self.assertIn("character_substitution_rate", plot_rows[0])
            self.assertIn("character_deletion_rate", plot_rows[0])
            self.assertIn("character_insertion_rate", plot_rows[0])
            with (output / "06_figures" / "gender_condition_plot_data.csv").open(encoding="utf-8") as handle:
                gender_plot_rows = list(csv.DictReader(handle))
            self.assertIn("condition", gender_plot_rows[0])
            self.assertIn("gender", gender_plot_rows[0])
            with (output / "06_figures" / "speaker_condition_plot_data.csv").open(encoding="utf-8") as handle:
                speaker_plot_rows = list(csv.DictReader(handle))
            self.assertIn("condition", speaker_plot_rows[0])
            self.assertIn("speaker_id", speaker_plot_rows[0])
            self.assertTrue((output / "01_qc" / "session_timing_qc.csv").exists())
            with (output / "02_preprocessed" / "normalized_transcriptions.csv").open(encoding="utf-8") as handle:
                normalized = list(csv.DictReader(handle))
            self.assertIn("transcription_normalized", normalized[0])
            self.assertNotIn("wer", normalized[0])
            with (output / "03_scores" / "transcription_scores.csv").open(encoding="utf-8") as handle:
                scores = list(csv.DictReader(handle))
            self.assertIn("transcription_normalized", scores[0])
            self.assertIn("wer", scores[0])
            with (output / "01_qc" / "utterance_condition_coverage.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 25)
            self.assertTrue(all(row["observed_transcriptions"] == "3" for row in rows))
            with (output / "01_qc" / "attention_check_audit.csv").open(encoding="utf-8") as handle:
                attention = list(csv.DictReader(handle))
            self.assertEqual(len(attention), 15)
            self.assertTrue(all(row["computed_pass"] == "True" for row in attention))
            with (output / "01_qc" / "session_timing_qc.csv").open(encoding="utf-8") as handle:
                timing = list(csv.DictReader(handle))
            self.assertTrue(all(row["timing_qc_status"] == "OK" for row in timing))

    def test_ineligible_participant_blocks_incomplete_list(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            output = root / "analysis_output"
            create_complete_results(results, invalid_participant=(1, 1))
            with self.assertRaises(QCError):
                run_pipeline(results, output, config())
            self.assertTrue((output / "01_qc" / "participant_qc.csv").exists())
            with (output / "run_manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["status"], "BLOCKED_STRUCTURE")
            summary = (output / "01_qc" / "qc_summary.txt").read_text(encoding="utf-8")
            self.assertIn("QC BLOCKED:", summary)
            self.assertIn("list 1 needs 1 eligible participant", summary)
            self.assertIn("analysis stopped before scoring", summary)

    def test_expected_response_can_fail_attention_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            output = root / "analysis_output"
            create_complete_results(results, failed_attention=(1, 1))
            with self.assertRaises(QCError):
                run_pipeline(results, output, config())
            with (output / "01_qc" / "attention_check_audit.csv").open(encoding="utf-8") as handle:
                attention = list(csv.DictReader(handle))
            failed = [row for row in attention if row["participant_id"] == "p1_1"]
            self.assertEqual(failed[0]["computed_pass"], "False")
            summary = (output / "01_qc" / "qc_summary.txt").read_text(encoding="utf-8")
            self.assertIn("1 participant was excluded (1 for failed at least one attention check)", summary)

    def test_attention_failure_can_be_explicitly_overridden(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            output = root / "analysis_output"
            create_complete_results(results, failed_attention=(1, 1))
            settings = config()
            settings["attention_check_include_overrides"] = [
                {
                    "list_id": "1",
                    "participant_id": "p1_1",
                    "session_id": "s1_1",
                    "reviewed_at": "2026-10-16",
                    "reason": "Manual review approved inclusion for testing.",
                }
            ]

            manifest = run_pipeline(results, output, settings)

            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["eligible_participants"], 15)
            self.assertEqual(manifest["attention_overrides_applied"], 1)
            self.assertIn("1 manual attention-check override was applied", manifest["qc_summary"])

            with (output / "01_qc" / "participant_qc.csv").open(encoding="utf-8") as handle:
                participants = list(csv.DictReader(handle))
            participant = next(row for row in participants if row["participant_id"] == "p1_1")
            self.assertEqual(participant["include"], "True")
            self.assertEqual(participant["attention_checks_passed"], "False")
            self.assertEqual(participant["attention_override_applied"], "True")
            self.assertEqual(participant["attention_override_reason"], "Manual review approved inclusion for testing.")

            with (output / "01_qc" / "attention_check_audit.csv").open(encoding="utf-8") as handle:
                attention = list(csv.DictReader(handle))
            failed = next(row for row in attention if row["participant_id"] == "p1_1")
            self.assertEqual(failed["final_pass"], "False")
            self.assertEqual(failed["attention_override_applied"], "True")

            with (output / "01_qc" / "qc_issues.csv").open(encoding="utf-8") as handle:
                issues = list(csv.DictReader(handle))
            override_warnings = [
                row for row in issues
                if row["code"] == "ATTENTION_FAILURE_OVERRIDDEN" and row["participant_id"] == "p1_1"
            ]
            self.assertEqual(len(override_warnings), 1)
            self.assertFalse(
                any(row["code"] == "PARTICIPANT_EXCLUDED" and row["participant_id"] == "p1_1" for row in issues)
            )

    def test_technical_failure_excludes_only_matching_transcription(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            output = root / "analysis_output"
            create_complete_results(results)
            settings = config()
            settings["technical_exclusion_overrides"] = [
                {
                    "list_id": "1",
                    "participant_id": "p1_1",
                    "session_id": "s1_1",
                    "utterance_id": "u1",
                    "condition": "a",
                    "reviewed_at": "2026-10-16",
                    "reason": "Confirmed playback failure in participant feedback.",
                }
            ]

            manifest = run_pipeline(results, output, settings)

            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["eligible_participants"], 15)
            self.assertEqual(manifest["eligible_participant_transcriptions"], 75)
            self.assertEqual(manifest["scored_transcriptions"], 74)
            self.assertEqual(manifest["score_rows"], 74)
            self.assertEqual(manifest["technical_exclusions_applied"], 1)
            self.assertIn(
                "1 approved technical transcription exclusion was applied",
                manifest["qc_summary"],
            )

            with (output / "01_qc" / "technical_exclusion_audit.csv").open(
                encoding="utf-8"
            ) as handle:
                audit = list(csv.DictReader(handle))
            self.assertEqual(len(audit), 1)
            self.assertEqual(audit[0]["exclusion_applied"], "True")
            self.assertEqual(audit[0]["utterance_id"], "u1")

            with (output / "01_qc" / "utterance_condition_coverage.csv").open(
                encoding="utf-8"
            ) as handle:
                coverage = list(csv.DictReader(handle))
            affected = next(
                row
                for row in coverage
                if row["utterance_id"] == "u1" and row["condition"] == "a"
            )
            self.assertEqual(affected["observed_transcriptions"], "2")
            self.assertEqual(affected["approved_technical_exclusions"], "1")
            self.assertEqual(
                affected["coverage_status"], "APPROVED_TECHNICAL_EXCLUSION"
            )

            with (output / "03_scores" / "transcription_scores.csv").open(
                encoding="utf-8"
            ) as handle:
                scores = list(csv.DictReader(handle))
            self.assertEqual(len(scores), 74)
            self.assertFalse(
                any(
                    row["participant_id"] == "p1_1"
                    and row["utterance_id"] == "u1"
                    for row in scores
                )
            )

            with (
                output / "04_summaries" / "technical_exclusion_sensitivity.csv"
            ).open(encoding="utf-8") as handle:
                sensitivity = list(csv.DictReader(handle))
            self.assertEqual(len(sensitivity), 1)
            self.assertEqual(sensitivity[0]["condition"], "a")
            self.assertEqual(
                sensitivity[0]["primary_excluding_technical_n_trials"], "14"
            )
            self.assertEqual(
                sensitivity[0]["sensitivity_including_as_submitted_n_trials"],
                "15",
            )


if __name__ == "__main__":
    unittest.main()
