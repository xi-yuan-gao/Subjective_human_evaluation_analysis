from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import parse_bool, participant_session_key, read_folder, write_csv, write_json
from .metrics import edit_counts, edit_counts_with_wildcards
from .normalize import (
    EXCLUDED_NAME_TOKEN,
    character_sequence,
    collapse_excluded_name_phrases,
    normalize_text,
)
from .questionnaire import summarize_questionnaire
from .schema import REQUIRED_MAIN_FIELDS, canonicalize, duplicate_groups, resolve_columns
from .summarize import (
    summarize_inter_listener_consistency,
    summarize_plays_wer,
    summarize_scores,
    summarize_xxx,
)
from .timing import session_timing_qc
from .visualize import generate_figures


SOURCE_KINDS = ("main", "prescreen", "sound_check", "postquestionnaire")

SCORE_FIELDS = {
    "raw_wer",
    "raw_cer",
    "wer_with_names",
    "cer_with_names",
    "word_substitutions",
    "word_deletions",
    "word_insertions",
    "word_errors",
    "reference_words",
    "wer",
    "character_substitutions",
    "character_deletions",
    "character_insertions",
    "character_errors",
    "reference_characters",
    "cer",
    "contains_xxx",
    "whole_utterance_xxx",
    "xxx_token_count",
    "xxx_token_rate",
    "perfect_word_transcription",
    "perfect_character_transcription",
}

OUTPUT_GUIDE = """# Analysis output guide

This directory contains one complete analysis run.

1. Start with `01_qc/qc_summary.txt`. If QC is blocked, inspect `01_qc/qc_issues.csv` and the relevant detailed audit. Approved playback failures are recorded in `01_qc/technical_exclusion_audit.csv`.
2. Review `02_preprocessed/normalized_transcriptions.csv` to verify text normalization and personal-name exclusions.
3. Use `03_scores/transcription_scores.csv` for response-level WER, CER, substitution, deletion, insertion, and `xxx` results.
4. Use `04_summaries/by_condition.csv` for the primary system comparison. When technical exclusions were applied, compare it with `04_summaries/technical_exclusion_sensitivity.csv`. The two `inter_listener_consistency` tables provide a supplementary descriptive check of variation across listeners.
5. Review participant feedback separately under `05_questionnaire/`.
6. Use the vector figures in `06_figures/` for reports and presentations. The plot-data CSVs record the exact condition values and the gender-by-condition and speaker-by-condition values with confidence intervals. Gender and speaker comparisons are descriptive for this experiment's four speakers.

`run_manifest.json` records the run status, input location, resolved columns, and row counts.
"""


class QCError(RuntimeError):
    """Raised after QC reports are written and a blocking problem is present."""


def _join_phrases(parts: list[str]) -> str:
    """Join short phrases into readable English for the QC summary."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _build_qc_summary(
    status: str,
    participant_qc: list[dict[str, Any]],
    utterance_condition_coverage: list[dict[str, Any]],
    timing_audit: list[dict[str, Any]],
    technical_exclusion_audit: list[dict[str, Any]],
    structural_errors: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    """Return a one-sentence QC verdict with the most useful next action."""
    expected_per_list = int(config["expected_participants_per_list"])
    list_ids = [str(list_id) for list_id in config["list_ids"]]
    expected_total = expected_per_list * len(list_ids)
    included_total = sum(row["include"] for row in participant_qc)

    included_by_list = Counter(
        str(row["list_id"]) for row in participant_qc if row["include"]
    )
    list_adjustments: list[str] = []
    for list_id in list_ids:
        difference = expected_per_list - included_by_list[list_id]
        if difference > 0:
            noun = "participant" if difference == 1 else "participants"
            list_adjustments.append(f"list {list_id} needs {difference} eligible {noun}")
        elif difference < 0:
            extra = abs(difference)
            noun = "participant" if extra == 1 else "participants"
            list_adjustments.append(f"list {list_id} has {extra} extra eligible {noun}")

    excluded = [row for row in participant_qc if not row["include"]]
    attention_overrides = sum(
        bool(row.get("attention_override_applied")) for row in participant_qc
    )
    exclusion_reasons = Counter(
        str(row.get("exclusion_reasons", "")).strip() or "unspecified reason"
        for row in excluded
    )
    exclusion_phrases = [
        f"{count} for {reason}" for reason, count in sorted(exclusion_reasons.items())
    ]

    incomplete_pairs = sum(
        row.get("coverage_status") == "INCOMPLETE"
        for row in utterance_condition_coverage
    )
    approved_shortfall_pairs = sum(
        row.get("coverage_status") == "APPROVED_TECHNICAL_EXCLUSION"
        for row in utterance_condition_coverage
    )
    technical_exclusions = sum(
        bool(row.get("exclusion_applied")) for row in technical_exclusion_audit
    )
    total_pairs = len(utterance_condition_coverage)
    timing_reviews = sum(
        row["included_in_analysis"] and row["timing_qc_status"] == "REVIEW"
        for row in timing_audit
    )
    summarized_codes = {"ELIGIBLE_PARTICIPANT_COUNT", "UTTERANCE_CONDITION_COUNT"}
    other_blockers = Counter(
        row["code"] for row in structural_errors if row["code"] not in summarized_codes
    )

    if status == "BLOCKED_STRUCTURE":
        parts = [f"{included_total}/{expected_total} required participant slots are filled"]
        if list_adjustments:
            parts.append(_join_phrases(list_adjustments))
        if excluded:
            details = _join_phrases(exclusion_phrases)
            noun = "participant was" if len(excluded) == 1 else "participants were"
            parts.append(f"{len(excluded)} {noun} excluded ({details})")
        if attention_overrides:
            noun = "override was" if attention_overrides == 1 else "overrides were"
            parts.append(f"{attention_overrides} manual attention-check {noun} applied")
        if incomplete_pairs:
            parts.append(
                f"{incomplete_pairs}/{total_pairs} utterance-condition pairs are incomplete"
            )
        if technical_exclusions:
            noun = "exclusion was" if technical_exclusions == 1 else "exclusions were"
            parts.append(
                f"{technical_exclusions} approved technical transcription {noun} applied to "
                f"{approved_shortfall_pairs} utterance-condition pairs"
            )
        if other_blockers:
            labels = [f"{count} {code}" for code, count in sorted(other_blockers.items())]
            parts.append(f"other blockers: {_join_phrases(labels)}")
        if timing_reviews:
            noun = "session" if timing_reviews == 1 else "sessions"
            parts.append(f"{timing_reviews} included {noun} also need timing review")
        return f"QC BLOCKED: {'; '.join(parts)}; analysis stopped before scoring."

    parts = [f"{included_total}/{expected_total} required participant slots are filled"]
    if total_pairs:
        full_coverage_pairs = total_pairs - incomplete_pairs - approved_shortfall_pairs
        coverage_label = "have full scorable coverage" if approved_shortfall_pairs else "are complete"
        parts.append(
            f"{full_coverage_pairs}/{total_pairs} utterance-condition pairs {coverage_label}"
        )
    if excluded:
        details = _join_phrases(exclusion_phrases)
        noun = "participant was" if len(excluded) == 1 else "participants were"
        parts.append(f"{len(excluded)} {noun} excluded ({details})")
    if attention_overrides:
        noun = "override was" if attention_overrides == 1 else "overrides were"
        parts.append(f"{attention_overrides} manual attention-check {noun} applied")
    if technical_exclusions:
        noun = "exclusion was" if technical_exclusions == 1 else "exclusions were"
        parts.append(
            f"{technical_exclusions} approved technical transcription {noun} applied to "
            f"{approved_shortfall_pairs} utterance-condition pairs"
        )
    if timing_reviews:
        noun = "session" if timing_reviews == 1 else "sessions"
        parts.append(f"{timing_reviews} included {noun} need timing review")
    prefix = "QC PASSED WITH EXCLUSIONS" if excluded else "QC PASSED"
    return f"{prefix}: {'; '.join(parts)}."


def issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    list_id: Any = "",
    participant_id: str = "",
    session_id: str = "",
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "list_id": list_id,
            "participant_id": participant_id,
            "session_id": session_id,
            "message": message,
        }
    )


def _index_unique(
    rows: list[dict[str, Any]],
    label: str,
    issues: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[participant_session_key(row)].append(row)
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in grouped.items():
        participant_id, session_id = key
        if not participant_id or not session_id:
            issue(issues, "ERROR", f"{label.upper()}_MISSING_KEY", f"{label} row has a blank participant_id or session_id.", group[0].get("list_id", ""), participant_id, session_id)
        if len(group) > 1:
            issue(issues, "ERROR", f"{label.upper()}_DUPLICATE", f"Found {len(group)} {label} rows for the same participant/session.", group[0].get("list_id", ""), participant_id, session_id)
        output[key] = group[0]
    return output


def _canonicalize_source(
    rows: list[dict[str, Any]],
    aliases: dict[str, list[str]],
    required: list[str],
    source_kind: str,
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    mapping = resolve_columns(rows, aliases)
    missing = [field for field in required if field not in mapping]
    if rows and missing:
        issue(issues, "ERROR", f"{source_kind.upper()}_MISSING_COLUMNS", f"Missing required columns: {', '.join(missing)}")
    return canonicalize(rows, mapping), mapping


def _is_attention(row: dict[str, Any], config: dict[str, Any]) -> bool:
    explicit = parse_bool(row.get("is_attention_check"))
    if explicit is not None:
        return explicit
    trial_type = str(row.get("trial_type", "")).strip().lower()
    return trial_type in {str(value).strip().lower() for value in config.get("attention_trial_values", [])}


def _attention_evaluation(row: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    expected_raw = str(row.get("expected_response", ""))
    transcription_raw = str(row.get("transcription", ""))
    expected_normalized = normalize_text(expected_raw, settings) if expected_raw.strip() else ""
    transcription_normalized = normalize_text(transcription_raw, settings)
    computed_pass = (expected_normalized == transcription_normalized) if expected_normalized else None
    explicit_pass = parse_bool(row.get("attention_check_passed"))
    final_pass = computed_pass if computed_pass is not None else explicit_pass
    return {
        "expected_response_raw": expected_raw,
        "transcription_raw": transcription_raw,
        "expected_response_normalized": expected_normalized,
        "transcription_normalized": transcription_normalized,
        "computed_pass": computed_pass,
        "explicit_pass": explicit_pass,
        "final_pass": final_pass,
        "explicit_computed_disagreement": (
            explicit_pass is not None and computed_pass is not None and explicit_pass != computed_pass
        ),
    }


def _reported_list_matches(reported: Any, folder_list_id: Any) -> bool:
    text = str(reported or "").strip().lower()
    if not text:
        return True
    expected = str(folder_list_id).strip().lower()
    return text in {expected, f"list{expected}", f"list_{expected}"}


def _score_row(row: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    scored = dict(row)
    reference_raw = str(row.get("reference_text", ""))
    transcription_raw = str(row.get("transcription", ""))
    reference_normalized = normalize_text(reference_raw, settings)
    transcription_normalized = normalize_text(transcription_raw, settings)
    raw_words = edit_counts(reference_raw.split(), transcription_raw.split())
    raw_characters = edit_counts(
        character_sequence(reference_raw, settings.get("cer_remove_spaces", True)),
        character_sequence(transcription_raw, settings.get("cer_remove_spaces", True)),
    )
    words_with_names = edit_counts(reference_normalized.split(), transcription_normalized.split())
    characters_with_names = edit_counts(
        character_sequence(reference_normalized, settings.get("cer_remove_spaces", True)),
        character_sequence(transcription_normalized, settings.get("cer_remove_spaces", True)),
    )
    name_phrases = [
        normalize_text(str(phrase), settings)
        for phrase in settings.get("excluded_name_phrases", [])
        if str(phrase).strip()
    ]
    reference_pattern, excluded_names, excluded_name_tokens = collapse_excluded_name_phrases(
        reference_normalized,
        name_phrases,
    )
    if excluded_names:
        words = edit_counts_with_wildcards(
            reference_pattern,
            transcription_normalized.split(),
            EXCLUDED_NAME_TOKEN,
        )
        scored_reference = " ".join(words["evaluated_reference"])
        scored_transcription = " ".join(words["evaluated_hypothesis"])
        characters = edit_counts(
            character_sequence(scored_reference, settings.get("cer_remove_spaces", True)),
            character_sequence(scored_transcription, settings.get("cer_remove_spaces", True)),
        )
    else:
        words = words_with_names
        characters = characters_with_names
        scored_reference = reference_normalized
        scored_transcription = transcription_normalized
    hypothesis_tokens = transcription_normalized.split()
    xxx_count = sum(token == "xxx" for token in hypothesis_tokens)
    scored.update(
        {
            "reference_text_raw": reference_raw,
            "transcription_raw": transcription_raw,
            "reference_text_normalized": reference_normalized,
            "transcription_normalized": transcription_normalized,
            "reference_text_scored": scored_reference,
            "transcription_scored": scored_transcription,
            "excluded_name_phrases": " | ".join(excluded_names),
            "excluded_reference_name_tokens": excluded_name_tokens,
            "raw_wer": raw_words["error_rate"] if raw_words["error_rate"] is not None else "",
            "raw_cer": raw_characters["error_rate"] if raw_characters["error_rate"] is not None else "",
            "wer_with_names": words_with_names["error_rate"] if words_with_names["error_rate"] is not None else "",
            "cer_with_names": characters_with_names["error_rate"] if characters_with_names["error_rate"] is not None else "",
            "word_substitutions": words["substitutions"],
            "word_deletions": words["deletions"],
            "word_insertions": words["insertions"],
            "word_errors": words["errors"],
            "reference_words": words["reference_length"],
            "wer": words["error_rate"] if words["error_rate"] is not None else "",
            "character_substitutions": characters["substitutions"],
            "character_deletions": characters["deletions"],
            "character_insertions": characters["insertions"],
            "character_errors": characters["errors"],
            "reference_characters": characters["reference_length"],
            "cer": characters["error_rate"] if characters["error_rate"] is not None else "",
            "contains_xxx": int(xxx_count > 0),
            "whole_utterance_xxx": int(hypothesis_tokens == ["xxx"]),
            "xxx_token_count": xxx_count,
            "xxx_token_rate": xxx_count / len(hypothesis_tokens) if hypothesis_tokens else 0,
            "perfect_word_transcription": int(words["error_rate"] == 0),
            "perfect_character_transcription": int(characters["error_rate"] == 0),
        }
    )
    return scored


def _technical_exclusion_sensitivity(
    primary_scores: list[dict[str, Any]],
    scores_including_exclusions: list[dict[str, Any]],
    affected_conditions: set[str],
    two_play_threshold: float,
) -> list[dict[str, Any]]:
    """Compare primary scores with technical failures scored as submitted."""
    primary = {
        str(row["condition"]): row
        for row in summarize_scores(primary_scores, ["condition"], two_play_threshold)
    }
    sensitivity = {
        str(row["condition"]): row
        for row in summarize_scores(
            scores_including_exclusions,
            ["condition"],
            two_play_threshold,
        )
    }
    output: list[dict[str, Any]] = []
    for condition in sorted(affected_conditions):
        primary_row = primary[condition]
        sensitivity_row = sensitivity[condition]
        result: dict[str, Any] = {"condition": condition}
        for field in (
            "n_trials",
            "mean_wer",
            "median_wer",
            "micro_wer",
            "mean_cer",
            "median_cer",
        ):
            result[f"primary_excluding_technical_{field}"] = primary_row[field]
            result[f"sensitivity_including_as_submitted_{field}"] = sensitivity_row[field]
        for field in ("mean_wer", "micro_wer", "mean_cer"):
            result[f"including_minus_excluding_{field}"] = (
                float(sensitivity_row[field]) - float(primary_row[field])
            )
        output.append(result)
    return output


def run_pipeline(results_root: Path, output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_root}. Use a new directory for each run.")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "OUTPUT_GUIDE.md").write_text(OUTPUT_GUIDE, encoding="utf-8")
    issues: list[dict[str, Any]] = []
    attention_override_index: dict[tuple[str, str], dict[str, Any]] = {}
    for override in config.get("attention_check_include_overrides", []):
        participant_id = str(override.get("participant_id", "")).strip()
        session_id = str(override.get("session_id", "")).strip()
        reason = str(override.get("reason", "")).strip()
        if not participant_id or not session_id or not reason:
            raise ValueError(
                "Each attention_check_include_overrides entry requires participant_id, session_id, and reason."
            )
        key = (participant_id, session_id)
        if key in attention_override_index:
            raise ValueError(
                f"Duplicate attention-check override for participant/session {key}."
            )
        attention_override_index[key] = override
    technical_exclusion_overrides: list[dict[str, Any]] = []
    technical_exclusion_keys: set[tuple[str, str, str, str, str]] = set()
    required_technical_fields = (
        "list_id",
        "participant_id",
        "session_id",
        "utterance_id",
        "condition",
        "reason",
    )
    for override in config.get("technical_exclusion_overrides", []):
        missing = [
            field
            for field in required_technical_fields
            if not str(override.get(field, "")).strip()
        ]
        if missing:
            raise ValueError(
                "Each technical_exclusion_overrides entry requires "
                f"{', '.join(required_technical_fields)}; missing {', '.join(missing)}."
            )
        key = tuple(
            str(override[field]).strip()
            for field in required_technical_fields[:-1]
        )
        if key in technical_exclusion_keys:
            raise ValueError(f"Duplicate technical exclusion override: {key}.")
        technical_exclusion_keys.add(key)
        technical_exclusion_overrides.append(override)
    inventory: list[dict[str, Any]] = []
    collected: dict[str, list[dict[str, Any]]] = {kind: [] for kind in SOURCE_KINDS}

    for list_id in config["list_ids"]:
        list_directory = results_root / config["list_directory_pattern"].format(list_id=list_id)
        if not list_directory.exists():
            issue(issues, "ERROR", "MISSING_LIST_DIRECTORY", f"List directory does not exist: {list_directory}", list_id)
        for source_kind in SOURCE_KINDS:
            rows, file_inventory = read_folder(list_directory / source_kind, list_id, source_kind)
            collected[source_kind].extend(rows)
            inventory.extend(file_inventory)
            if not rows:
                issue(issues, "ERROR", "MISSING_SOURCE_ROWS", f"No CSV rows found in {source_kind}/.", list_id)

    aliases = config["column_aliases"]
    main, main_mapping = _canonicalize_source(collected["main"], aliases, REQUIRED_MAIN_FIELDS, "main", issues)
    prescreen, prescreen_mapping = _canonicalize_source(collected["prescreen"], aliases, ["participant_id", "session_id"], "prescreen", issues)
    sound, sound_mapping = _canonicalize_source(collected["sound_check"], aliases, ["participant_id", "session_id"], "sound_check", issues)
    questionnaire, questionnaire_mapping = _canonicalize_source(collected["postquestionnaire"], aliases, ["participant_id", "session_id"], "postquestionnaire", issues)

    for source_kind, source_rows in (("main", main), ("prescreen", prescreen), ("sound_check", sound), ("postquestionnaire", questionnaire)):
        for row in source_rows:
            if not _reported_list_matches(row.get("reported_list_id"), row.get("list_id")):
                issue(
                    issues,
                    "ERROR",
                    "REPORTED_LIST_ID_MISMATCH",
                    f"{source_kind} row reports list_id={row.get('reported_list_id', '')!r}, but the file is under list {row.get('list_id', '')}.",
                    row.get("list_id", ""),
                    str(row.get("participant_id", "")),
                    str(row.get("session_id", "")),
                )

    for row in main:
        if _is_attention(row, config):
            continue
        # A blank transcription is a valid all-deletion outcome. Other identifiers and the reference must be present.
        nonblank_fields = [field for field in REQUIRED_MAIN_FIELDS if field != "transcription"]
        blank_fields = [field for field in nonblank_fields if not str(row.get(field, "")).strip()]
        if blank_fields:
            issue(
                issues,
                "ERROR",
                "MAIN_BLANK_REQUIRED_VALUE",
                f"Blank required values: {', '.join(blank_fields)} (source row {row.get('source_row', '')}).",
                row.get("list_id", ""),
                str(row.get("participant_id", "")),
                str(row.get("session_id", "")),
            )

    prescreen_index = _index_unique(prescreen, "prescreen", issues)
    sound_index = _index_unique(sound, "sound_check", issues)
    questionnaire_index = _index_unique(questionnaire, "postquestionnaire", issues)

    main_sessions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in main:
        main_sessions[participant_session_key(row)].append(row)

    participants_seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for key, rows in main_sessions.items():
        participants_seen[key[0]].add((str(rows[0].get("list_id", "")), key[1]))
    for participant_id, appearances in participants_seen.items():
        if participant_id and len(appearances) > 1:
            issue(issues, "ERROR", "PARTICIPANT_ACROSS_LISTS_OR_SESSIONS", f"Participant appears in multiple list/session combinations: {sorted(appearances)}", participant_id=participant_id)

    participant_qc: list[dict[str, Any]] = []
    included_keys: set[tuple[str, str]] = set()
    expected_trials = config.get("expected_main_trials_per_participant")
    expected_attention = config.get("expected_attention_checks_per_participant")
    for key, rows in sorted(main_sessions.items()):
        participant_id, session_id = key
        list_id = rows[0].get("list_id", "")
        pre = prescreen_index.get(key)
        snd = sound_index.get(key)
        post = questionnaire_index.get(key)
        native = parse_bool(pre.get("native_english")) if pre else None
        headphones = parse_bool(pre.get("headphones")) if pre else None
        heard = parse_bool(snd.get("heard_clearly")) if snd else None
        attention_rows = [row for row in rows if _is_attention(row, config)]
        scored_rows = [row for row in rows if not _is_attention(row, config)]
        attention_evaluations = [_attention_evaluation(row, config["normalization"]) for row in attention_rows]
        attention_pass_values = [row["final_pass"] for row in attention_evaluations]
        attention_pass = all(value is True for value in attention_pass_values) if attention_rows and all(value is not None for value in attention_pass_values) else None
        attention_override = attention_override_index.get(key)
        override_list_id = str(attention_override.get("list_id", "")).strip() if attention_override else ""
        attention_override_valid = bool(
            attention_override
            and (not override_list_id or override_list_id == str(list_id))
        )
        attention_override_applied = bool(attention_pass is False and attention_override_valid)
        attention_override_reason = (
            str(attention_override.get("reason", "")).strip()
            if attention_override_applied and attention_override
            else ""
        )
        if attention_override and not attention_override_valid:
            issue(
                issues,
                "WARNING",
                "ATTENTION_OVERRIDE_LIST_MISMATCH",
                f"Configured override belongs to list {override_list_id}, but the participant/session is in list {list_id}; override was not applied.",
                list_id,
                participant_id,
                session_id,
            )
        elif attention_override_applied:
            issue(
                issues,
                "WARNING",
                "ATTENTION_FAILURE_OVERRIDDEN",
                attention_override_reason,
                list_id,
                participant_id,
                session_id,
            )
        elif attention_override_valid:
            issue(
                issues,
                "WARNING",
                "ATTENTION_OVERRIDE_NOT_NEEDED",
                "Configured override was not applied because the attention result was not a confirmed failure.",
                list_id,
                participant_id,
                session_id,
            )
        blockers: list[str] = []
        if pre is None:
            blockers.append("missing prescreen row")
        elif str(pre.get("list_id", "")) != str(list_id):
            blockers.append(f"prescreen row belongs to list {pre.get('list_id', '')}")
        if snd is None:
            blockers.append("missing sound-check row")
        elif str(snd.get("list_id", "")) != str(list_id):
            blockers.append(f"sound-check row belongs to list {snd.get('list_id', '')}")
        if post is not None and str(post.get("list_id", "")) != str(list_id):
            blockers.append(f"postquestionnaire row belongs to list {post.get('list_id', '')}")
        if native is not True:
            blockers.append("native_english is not TRUE")
        if headphones is not True:
            blockers.append("headphones is not TRUE")
        if heard is not True:
            blockers.append("heard_clearly is not TRUE")
        if expected_trials is not None and len(scored_rows) != int(expected_trials):
            blockers.append(f"expected {expected_trials} scored trials; found {len(scored_rows)}")
        if expected_attention is not None and len(attention_rows) != int(expected_attention):
            blockers.append(f"expected {expected_attention} attention checks; found {len(attention_rows)}")
        if attention_rows and attention_pass is False and not attention_override_applied:
            blockers.append("failed at least one attention check")
        if any(row["explicit_computed_disagreement"] for row in attention_evaluations):
            issue(issues, "WARNING", "ATTENTION_RESULT_DISAGREEMENT", "Stored attention-check pass/fail disagrees with normalized expected-response matching.", list_id, participant_id, session_id)
        if post is None:
            issue(issues, "WARNING", "MISSING_POSTQUESTIONNAIRE", "No postquestionnaire row matched this participant/session.", list_id, participant_id, session_id)
        if attention_rows and attention_pass is None:
            issue(issues, "WARNING", "ATTENTION_PASS_UNKNOWN", "Attention-check rows were found, but no unambiguous pass/fail value was available.", list_id, participant_id, session_id)
        include = not blockers
        if include:
            included_keys.add(key)
        else:
            issue(issues, "ERROR", "PARTICIPANT_EXCLUDED", "; ".join(blockers), list_id, participant_id, session_id)
        participant_qc.append(
            {
                "list_id": list_id,
                "participant_id": participant_id,
                "session_id": session_id,
                "native_english": native,
                "headphones": headphones,
                "heard_clearly": heard,
                "main_rows": len(rows),
                "scored_trial_rows": len(scored_rows),
                "attention_check_rows": len(attention_rows),
                "attention_checks_passed": attention_pass,
                "attention_override_applied": attention_override_applied,
                "attention_override_reason": attention_override_reason,
                "attention_override_reviewed_at": (
                    str(attention_override.get("reviewed_at", "")).strip()
                    if attention_override_applied and attention_override
                    else ""
                ),
                "postquestionnaire_present": post is not None,
                "include": include,
                "exclusion_reasons": "; ".join(blockers),
            }
        )

    for participant_id, session_id in sorted(set(attention_override_index) - set(main_sessions)):
        issue(
            issues,
            "WARNING",
            "ATTENTION_OVERRIDE_NOT_FOUND",
            "Configured attention-check override has no matching main participant/session.",
            participant_id=participant_id,
            session_id=session_id,
        )

    main_keys = set(main_sessions)
    for label, source_index in (("prescreen", prescreen_index), ("sound_check", sound_index), ("postquestionnaire", questionnaire_index)):
        for participant_id, session_id in sorted(set(source_index) - main_keys):
            source_row = source_index[(participant_id, session_id)]
            issue(
                issues,
                "WARNING",
                "ORPHAN_SOURCE_ROW",
                f"{label} row has no matching participant/session in main results.",
                source_row.get("list_id", ""),
                participant_id,
                session_id,
            )

    for list_id in config["list_ids"]:
        count = sum(str(row["list_id"]) == str(list_id) and row["include"] for row in participant_qc)
        expected = int(config["expected_participants_per_list"])
        if count != expected:
            issue(issues, "ERROR", "ELIGIBLE_PARTICIPANT_COUNT", f"Expected {expected} included participants; found {count}.", list_id)

    duplicates = duplicate_groups(
        [row for row in main if not _is_attention(row, config)],
        ["list_id", "participant_id", "session_id", "utterance_id", "condition"],
    )
    if duplicates:
        issue(issues, "ERROR", "DUPLICATE_MAIN_TRIALS", f"Found {len(duplicates)} rows belonging to duplicate trial keys.")

    eligible_main_before_technical_exclusions = [
        row
        for row in main
        if participant_session_key(row) in included_keys
        and not _is_attention(row, config)
    ]
    technical_exclusion_audit: list[dict[str, Any]] = []
    excluded_source_rows: set[tuple[str, str]] = set()
    for override in technical_exclusion_overrides:
        list_id = str(override["list_id"]).strip()
        participant_id = str(override["participant_id"]).strip()
        session_id = str(override["session_id"]).strip()
        utterance_id = str(override["utterance_id"]).strip()
        condition = str(override["condition"]).strip()
        matches = [
            row
            for row in main
            if not _is_attention(row, config)
            and str(row.get("list_id", "")) == list_id
            and str(row.get("participant_id", "")) == participant_id
            and str(row.get("session_id", "")) == session_id
            and str(row.get("utterance_id", "")) == utterance_id
            and str(row.get("condition", "")) == condition
        ]
        participant_included = (participant_id, session_id) in included_keys
        exclusion_applied = len(matches) == 1 and participant_included
        if not matches:
            issue(
                issues,
                "ERROR",
                "TECHNICAL_EXCLUSION_NOT_FOUND",
                "Configured technical exclusion did not match a non-attention transcription.",
                list_id,
                participant_id,
                session_id,
            )
        elif len(matches) > 1:
            issue(
                issues,
                "ERROR",
                "TECHNICAL_EXCLUSION_MULTIPLE_MATCHES",
                f"Configured technical exclusion matched {len(matches)} transcriptions.",
                list_id,
                participant_id,
                session_id,
            )
        elif not participant_included:
            issue(
                issues,
                "WARNING",
                "TECHNICAL_EXCLUSION_PARTICIPANT_NOT_INCLUDED",
                "The matching transcription belongs to a participant already excluded by participant-level QC.",
                list_id,
                participant_id,
                session_id,
            )
        else:
            source_key = (
                str(matches[0].get("source_file", "")),
                str(matches[0].get("source_row", "")),
            )
            excluded_source_rows.add(source_key)
            issue(
                issues,
                "WARNING",
                "TECHNICAL_TRANSCRIPTION_EXCLUDED",
                str(override["reason"]).strip(),
                list_id,
                participant_id,
                session_id,
            )
        matched = matches[0] if len(matches) == 1 else {}
        would_be_score = (
            _score_row(matched, config["normalization"]) if matched else {}
        )
        technical_exclusion_audit.append(
            {
                "list_id": list_id,
                "participant_id": participant_id,
                "session_id": session_id,
                "item_order": matched.get("item_order", ""),
                "utterance_id": utterance_id,
                "condition": condition,
                "reference_text": matched.get("reference_text", ""),
                "transcription": matched.get("transcription", ""),
                "plays_used": matched.get("plays_used", ""),
                "source_file": matched.get("source_file", ""),
                "source_row": matched.get("source_row", ""),
                "matched_rows": len(matches),
                "participant_included": participant_included,
                "exclusion_applied": exclusion_applied,
                "reviewed_at": str(override.get("reviewed_at", "")).strip(),
                "reason": str(override["reason"]).strip(),
                "would_be_wer": would_be_score.get("wer", ""),
                "would_be_cer": would_be_score.get("cer", ""),
            }
        )
    technical_exclusion_count_by_participant = Counter(
        (str(row["participant_id"]), str(row["session_id"]))
        for row in technical_exclusion_audit
        if row["exclusion_applied"]
    )
    for row in participant_qc:
        row["technical_exclusions_applied"] = (
            technical_exclusion_count_by_participant[
                (str(row["participant_id"]), str(row["session_id"]))
            ]
        )
    eligible_main = [
        row
        for row in eligible_main_before_technical_exclusions
        if (
            str(row.get("source_file", "")),
            str(row.get("source_row", "")),
        )
        not in excluded_source_rows
    ]
    participant_qc_by_key = {
        (str(row["participant_id"]), str(row["session_id"])): row
        for row in participant_qc
    }
    attention_audit: list[dict[str, Any]] = []
    for row in main:
        if not _is_attention(row, config):
            continue
        evaluation = _attention_evaluation(row, config["normalization"])
        participant_decision = participant_qc_by_key.get(
            (str(row.get("participant_id", "")), str(row.get("session_id", ""))),
            {},
        )
        attention_audit.append(
            {
            "list_id": row.get("list_id", ""),
            "participant_id": row.get("participant_id", ""),
            "session_id": row.get("session_id", ""),
            "source_file": row.get("source_file", ""),
            "source_row": row.get("source_row", ""),
            "trial_type": row.get("trial_type", ""),
            "item_order": row.get("item_order", ""),
            "expected_response_raw": evaluation["expected_response_raw"],
            "transcription_raw": evaluation["transcription_raw"],
            "expected_response_normalized": evaluation["expected_response_normalized"],
            "transcription_normalized": evaluation["transcription_normalized"],
            "computed_pass": evaluation["computed_pass"],
            "explicit_pass": evaluation["explicit_pass"],
            "final_pass": evaluation["final_pass"],
            "attention_override_applied": participant_decision.get("attention_override_applied", False),
            "attention_override_reason": participant_decision.get("attention_override_reason", ""),
            "explicit_computed_disagreement": evaluation["explicit_computed_disagreement"],
            "response_time_sec": row.get("response_time_sec", ""),
            "plays_used": row.get("plays_used", ""),
            }
        )
    if main and "is_attention_check" not in main_mapping and "trial_type" not in main_mapping:
        issue(issues, "WARNING", "ATTENTION_COLUMN_NOT_FOUND", "No attention-check marker column was found. Configure an alias if attention checks exist.")

    list_condition_counts: Counter[tuple[str, str]] = Counter(
        (str(row.get("list_id", "")), str(row.get("condition", ""))) for row in eligible_main
    )
    list_condition_coverage: list[dict[str, Any]] = []
    for list_id in config["list_ids"]:
        observed = {condition for current_list, condition in list_condition_counts if current_list == str(list_id)}
        for condition in config.get("expected_conditions", []):
            count = list_condition_counts[(str(list_id), condition)]
            list_condition_coverage.append({"list_id": list_id, "condition": condition, "trial_count": count, "present": count > 0})
            if count == 0:
                issue(issues, "ERROR", "MISSING_LIST_CONDITION", f"Expected condition is absent: {condition}", list_id)
        unexpected = observed - set(config.get("expected_conditions", []))
        for condition in sorted(unexpected):
            issue(issues, "ERROR", "UNEXPECTED_CONDITION", f"Unexpected condition found: {condition}", list_id)

    pair_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible_main:
        pair_rows[(str(row.get("utterance_id", "")), str(row.get("condition", "")))].append(row)
    pair_rows_before_exclusions: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in eligible_main_before_technical_exclusions:
        pair_rows_before_exclusions[
            (str(row.get("utterance_id", "")), str(row.get("condition", "")))
        ].append(row)
    excluded_pair_counts = Counter(
        (str(row["utterance_id"]), str(row["condition"]))
        for row in technical_exclusion_audit
        if row["exclusion_applied"]
    )
    utterance_ids = sorted(
        {
            str(row.get("utterance_id", ""))
            for row in eligible_main_before_technical_exclusions
        }
    )
    utterance_condition_coverage: list[dict[str, Any]] = []
    for utterance_id in utterance_ids:
        for condition in config.get("expected_conditions", []):
            group = pair_rows.get((utterance_id, condition), [])
            lists = {str(row.get("list_id", "")) for row in group}
            participants = {str(row.get("participant_id", "")) for row in group}
            sessions = {str(row.get("session_id", "")) for row in group}
            expected_count = int(config["expected_ratings_per_utterance_condition"])
            complete_for_scoring = (
                len(group) == expected_count
                and len(lists) == 1
                and len(participants) == expected_count
                and len(sessions) == expected_count
            )
            group_before_exclusions = pair_rows_before_exclusions.get(
                (utterance_id, condition), []
            )
            raw_lists = {
                str(row.get("list_id", "")) for row in group_before_exclusions
            }
            raw_participants = {
                str(row.get("participant_id", ""))
                for row in group_before_exclusions
            }
            raw_sessions = {
                str(row.get("session_id", "")) for row in group_before_exclusions
            }
            complete_before_exclusions = (
                len(group_before_exclusions) == expected_count
                and len(raw_lists) == 1
                and len(raw_participants) == expected_count
                and len(raw_sessions) == expected_count
            )
            approved_exclusions = excluded_pair_counts[(utterance_id, condition)]
            approved_shortfall = bool(
                not complete_for_scoring
                and approved_exclusions
                and complete_before_exclusions
                and len(group) + approved_exclusions == expected_count
            )
            coverage_status = (
                "COMPLETE"
                if complete_for_scoring
                else "APPROVED_TECHNICAL_EXCLUSION"
                if approved_shortfall
                else "INCOMPLETE"
            )
            utterance_condition_coverage.append(
                {
                    "utterance_id": utterance_id,
                    "condition": condition,
                    "observed_transcriptions": len(group),
                    "expected_transcriptions": expected_count,
                    "distinct_lists": len(lists),
                    "list_ids": " | ".join(sorted(lists)),
                    "distinct_participants": len(participants),
                    "distinct_sessions": len(sessions),
                    "approved_technical_exclusions": approved_exclusions,
                    "complete_before_technical_exclusions": complete_before_exclusions,
                    "complete_for_scoring": complete_for_scoring,
                    "coverage_status": coverage_status,
                    "complete": coverage_status != "INCOMPLETE",
                }
            )
    for row in utterance_condition_coverage:
        if row["coverage_status"] == "INCOMPLETE":
            issue(issues, "ERROR", "UTTERANCE_CONDITION_COUNT", f"{row['utterance_id']} / {row['condition']}: expected {row['expected_transcriptions']} ratings from one list and distinct participants/sessions; found {row['observed_transcriptions']} ratings across {row['distinct_lists']} lists.")

    utterance_metadata_audit: list[dict[str, Any]] = []
    for utterance_id in utterance_ids:
        rows = [
            row
            for row in eligible_main_before_technical_exclusions
            if str(row.get("utterance_id", "")) == utterance_id
        ]
        speakers = sorted({str(row.get("speaker_id", "")) for row in rows})
        references = sorted({str(row.get("reference_text", "")) for row in rows})
        consistent = len(speakers) == 1 and len(references) == 1
        utterance_metadata_audit.append(
            {
                "utterance_id": utterance_id,
                "distinct_speakers": len(speakers),
                "speakers": " | ".join(speakers),
                "distinct_reference_texts": len(references),
                "consistent": consistent,
            }
        )
        if len(speakers) != 1:
            issue(issues, "ERROR", "UTTERANCE_SPEAKER_CONFLICT", f"Utterance {utterance_id} has {len(speakers)} speaker IDs.")
        if len(references) != 1:
            issue(issues, "ERROR", "UTTERANCE_REFERENCE_CONFLICT", f"Utterance {utterance_id} has {len(references)} reference texts.")

    attention_by_session: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in attention_audit:
        attention_by_session[(str(row["participant_id"]), str(row["session_id"]))].append(row)
    attention_summary: list[dict[str, Any]] = []
    for participant in participant_qc:
        participant_id = str(participant["participant_id"])
        session_id = str(participant["session_id"])
        rows = attention_by_session.get((participant_id, session_id), [])
        attention_summary.append(
            {
                "list_id": participant["list_id"],
                "participant_id": participant_id,
                "session_id": session_id,
                "included_in_analysis": participant["include"],
                "attention_checks": len(rows),
                "passed": sum(row["final_pass"] is True for row in rows),
                "failed": sum(row["final_pass"] is False for row in rows),
                "unknown": sum(row["final_pass"] is None for row in rows),
                "all_passed": bool(rows) and all(row["final_pass"] is True for row in rows),
                "attention_override_applied": participant["attention_override_applied"],
                "attention_override_reason": participant["attention_override_reason"],
            }
        )

    timing_columns = {"response_time_sec", "study_elapsed_sec", "submitted_at"} & set(main_mapping)
    if timing_columns:
        timing_audit = session_timing_qc(main_sessions, included_keys, config.get("timing_qc", {}))
        for row in timing_audit:
            if row["included_in_analysis"] and row["timing_qc_status"] == "REVIEW":
                issue(
                    issues,
                    "WARNING",
                    "SESSION_TIMING_REVIEW",
                    str(row["timing_qc_reasons"]),
                    row["list_id"],
                    str(row["participant_id"]),
                    str(row["session_id"]),
                )
    else:
        timing_audit = []
        issue(issues, "WARNING", "TIMING_COLUMNS_NOT_FOUND", "No response-time, elapsed-time, or submitted-at column was found; session timing QC was skipped.")

    qc_root = output_root / "01_qc"
    write_csv(qc_root / "source_file_inventory.csv", inventory)
    write_csv(qc_root / "participant_qc.csv", participant_qc)
    write_csv(qc_root / "duplicate_transcriptions.csv", duplicates)
    write_csv(qc_root / "attention_check_audit.csv", attention_audit)
    write_csv(qc_root / "attention_check_summary.csv", attention_summary)
    write_csv(qc_root / "session_timing_qc.csv", timing_audit)
    write_csv(qc_root / "list_condition_coverage.csv", list_condition_coverage)
    write_csv(qc_root / "utterance_condition_coverage.csv", utterance_condition_coverage)
    write_csv(qc_root / "utterance_metadata_audit.csv", utterance_metadata_audit)
    write_csv(qc_root / "technical_exclusion_audit.csv", technical_exclusion_audit)

    structural_codes = {
        "MISSING_LIST_DIRECTORY", "MISSING_SOURCE_ROWS", "MAIN_MISSING_COLUMNS", "PRESCREEN_MISSING_COLUMNS",
        "SOUND_CHECK_MISSING_COLUMNS", "PARTICIPANT_ACROSS_LISTS_OR_SESSIONS", "DUPLICATE_MAIN_TRIALS",
        "MAIN_BLANK_REQUIRED_VALUE", "PRESCREEN_MISSING_KEY", "SOUND_CHECK_MISSING_KEY", "POSTQUESTIONNAIRE_MISSING_KEY",
        "PRESCREEN_DUPLICATE", "SOUND_CHECK_DUPLICATE", "POSTQUESTIONNAIRE_DUPLICATE",
        "ELIGIBLE_PARTICIPANT_COUNT", "MISSING_LIST_CONDITION", "UNEXPECTED_CONDITION", "UTTERANCE_CONDITION_COUNT",
        "UTTERANCE_SPEAKER_CONFLICT", "UTTERANCE_REFERENCE_CONFLICT",
        "REPORTED_LIST_ID_MISMATCH",
        "TECHNICAL_EXCLUSION_NOT_FOUND", "TECHNICAL_EXCLUSION_MULTIPLE_MATCHES",
    }
    structural_errors = [row for row in issues if row["severity"] == "ERROR" and row["code"] in structural_codes]
    eligibility_errors = [row for row in issues if row["severity"] == "ERROR" and row["code"] == "PARTICIPANT_EXCLUDED"]
    write_csv(qc_root / "qc_issues.csv", issues)

    if structural_errors and config.get("fail_on_structural_qc", True):
        status = "BLOCKED_STRUCTURE"
    else:
        status = "PASSED_WITH_EXCLUSIONS" if eligibility_errors else "PASSED"

    qc_summary = _build_qc_summary(
        status,
        participant_qc,
        utterance_condition_coverage,
        timing_audit,
        technical_exclusion_audit,
        structural_errors,
        config,
    )
    summary_path = qc_root / "qc_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(qc_summary + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root.resolve()),
        "output_root": str(output_root.resolve()),
        "status": status,
        "eligible_participants": len(included_keys),
        "eligible_participant_transcriptions": len(eligible_main_before_technical_exclusions),
        "scored_transcriptions": len(eligible_main),
        "error_count": sum(row["severity"] == "ERROR" for row in issues),
        "warning_count": sum(row["severity"] == "WARNING" for row in issues),
        "attention_overrides_applied": sum(
            bool(row["attention_override_applied"]) for row in participant_qc
        ),
        "technical_exclusions_applied": sum(
            bool(row["exclusion_applied"]) for row in technical_exclusion_audit
        ),
        "qc_summary": qc_summary,
        "resolved_columns": {
            "main": main_mapping,
            "prescreen": prescreen_mapping,
            "sound_check": sound_mapping,
            "postquestionnaire": questionnaire_mapping,
        },
    }
    write_json(output_root / "run_manifest.json", manifest)

    if status == "BLOCKED_STRUCTURE":
        raise QCError(f"{qc_summary} Details: {qc_root / 'qc_issues.csv'}")

    scored = [_score_row(row, config["normalization"]) for row in eligible_main]
    scores_including_technical_exclusions = [
        _score_row(row, config["normalization"])
        for row in eligible_main_before_technical_exclusions
    ]
    normalized = [
        {key: value for key, value in row.items() if key not in SCORE_FIELDS}
        for row in scored
    ]
    eligible_questionnaire = [row for row in questionnaire if participant_session_key(row) in included_keys]
    write_csv(output_root / "02_preprocessed" / "eligible_transcriptions.csv", eligible_main)
    write_csv(output_root / "02_preprocessed" / "normalized_transcriptions.csv", normalized)
    write_csv(output_root / "03_scores" / "transcription_scores.csv", scored)

    summaries = {
        "by_condition.csv": ["condition"],
        "by_speaker.csv": ["speaker_id"],
        "by_speaker_condition.csv": ["speaker_id", "condition"],
        "by_listener.csv": ["participant_id", "list_id"],
        "by_listener_condition.csv": ["participant_id", "list_id", "condition"],
        "by_utterance.csv": ["utterance_id"],
        "by_list.csv": ["list_id", "condition"],
        "by_utterance_condition.csv": ["utterance_id", "condition"],
    }
    if any(str(row.get("gender", "")).strip() for row in scored):
        summaries["by_gender_condition.csv"] = ["gender", "condition"]
    two_play_threshold = float(config.get("timing_qc", {}).get("two_play_threshold", 2))
    for filename, keys in summaries.items():
        write_csv(output_root / "04_summaries" / filename, summarize_scores(scored, keys, two_play_threshold))
    write_csv(output_root / "04_summaries" / "plays_wer_summary.csv", summarize_plays_wer(scored))
    consistency_by_item, consistency_by_condition = summarize_inter_listener_consistency(
        scored,
        int(config["expected_ratings_per_utterance_condition"]),
    )
    write_csv(
        output_root
        / "04_summaries"
        / "inter_listener_consistency_by_utterance_condition.csv",
        consistency_by_item,
    )
    write_csv(
        output_root
        / "04_summaries"
        / "inter_listener_consistency_by_condition.csv",
        consistency_by_condition,
    )
    affected_conditions = {
        str(row["condition"])
        for row in technical_exclusion_audit
        if row["exclusion_applied"]
    }
    if affected_conditions:
        write_csv(
            output_root / "04_summaries" / "technical_exclusion_sensitivity.csv",
            _technical_exclusion_sensitivity(
                scored,
                scores_including_technical_exclusions,
                affected_conditions,
                two_play_threshold,
            ),
        )
    write_csv(output_root / "03_scores" / "xxx_diagnostics.csv", summarize_xxx(scored))

    questionnaire_settings = config["questionnaire"]
    categorical = [column for column in questionnaire_settings.get("categorical_columns", []) if any(column in row for row in eligible_questionnaire)]
    free_text = [column for column in questionnaire_settings.get("free_text_columns", []) if any(column in row for row in eligible_questionnaire)]
    question_summary, comments = summarize_questionnaire(eligible_questionnaire, categorical, free_text)
    questionnaire_root = output_root / "05_questionnaire"
    write_csv(questionnaire_root / "eligible_responses.csv", eligible_questionnaire)
    write_csv(questionnaire_root / "categorical_summary.csv", question_summary)
    write_csv(questionnaire_root / "comments.csv", comments)

    figures = generate_figures(
        scored,
        output_root / "06_figures",
        [str(condition) for condition in config.get("expected_conditions", [])],
        config.get("visualization", {}),
    )

    manifest["status"] = "COMPLETE_WITH_EXCLUSIONS" if eligibility_errors else "COMPLETE"
    manifest["score_rows"] = len(scored)
    manifest["figures"] = figures
    write_json(output_root / "run_manifest.json", manifest)
    return manifest
