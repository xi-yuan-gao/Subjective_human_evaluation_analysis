from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SOURCE_KINDS = ("main", "prescreen", "sound_check", "postquestionnaire")
REQUIRED_MAIN_FIELDS = (
    "participant_id",
    "session_id",
    "item_id",
    "condition",
    "speaker_id",
    "mos_rating",
)
TRUE_VALUES = {"true", "1", "yes", "y", "t"}
FALSE_VALUES = {"false", "0", "no", "n", "f"}
ELIGIBILITY_FIELDS = ("native_english", "headphones", "heard_clearly")

OUTPUT_GUIDE = """# MOS analysis output guide

Start with `01_qc/qc_summary.txt`. If QC is blocked, use
`01_qc/qc_issues.csv` and the named audit files to resolve the problem.

`02_preprocessed/mos_analysis_ready.csv` is created only after structural QC
passes. It contains one eligible, non-attention MOS rating per row and is the
source for the descriptive summaries, figures, and separate ordinal
mixed-effects analysis.

Raw result CSVs are never edited. `run_manifest.json` records input and output
paths, resolved source columns, row counts, and the final QC status.
"""


class MOSQCError(RuntimeError):
    """Raised after MOS QC outputs are saved and a blocking error remains."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV while stripping headers and surrounding cell whitespace."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [
            {str(key).strip(): (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def _read_folder(
    folder: Path,
    list_id: Any,
    source_kind: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read every CSV in one source folder and attach row provenance."""
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    for path in files:
        file_rows = _read_csv(path)
        for row_number, row in enumerate(file_rows, start=2):
            enriched: dict[str, Any] = dict(row)
            if "list_id" in row:
                enriched["reported_list_id"] = row["list_id"]
            enriched["list_id"] = str(list_id)
            enriched["source_file"] = str(path)
            enriched["source_row"] = row_number
            rows.append(enriched)
        inventory.append(
            {
                "list_id": list_id,
                "source_kind": source_kind,
                "source_file": str(path),
                "row_count": len(file_rows),
                "columns": " | ".join(file_rows[0]) if file_rows else "",
            }
        )
    if not files:
        inventory.append(
            {
                "list_id": list_id,
                "source_kind": source_kind,
                "source_file": "",
                "row_count": 0,
                "columns": "",
            }
        )
    return rows, inventory


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    """Write dictionaries to a UTF-8 CSV with a stable union of columns."""
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(path: Path, value: Any) -> None:
    """Write indented UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _parse_bool(value: Any) -> bool | None:
    """Parse common Boolean spellings without guessing unknown values."""
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def _number(value: Any) -> float | None:
    """Return a finite numeric value or None."""
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _parse_rating(value: Any, allowed: set[int]) -> int | None:
    """Accept only whole-number MOS values that belong to the configured scale."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        rating = value
    elif isinstance(value, str) and value.strip().isdigit():
        rating = int(value.strip())
    else:
        return None
    return rating if rating in allowed else None


def _participant_key(row: dict[str, Any]) -> tuple[str, str]:
    """Return the participant/session join key."""
    return (
        str(row.get("participant_id", "")).strip(),
        str(row.get("session_id", "")).strip(),
    )


def _resolve_columns(
    rows: list[dict[str, Any]],
    aliases: dict[str, list[str]],
) -> dict[str, str]:
    """Resolve configured aliases to the actual case-preserving CSV headers."""
    present: set[str] = set()
    for row in rows:
        present.update(row)
    lowered = {name.lower(): name for name in present}
    resolved: dict[str, str] = {}
    for canonical, options in aliases.items():
        for option in options:
            if option.lower() in lowered:
                resolved[canonical] = lowered[option.lower()]
                break
    return resolved


def _canonicalize(
    rows: list[dict[str, Any]],
    aliases: dict[str, list[str]],
    required: Iterable[str],
    source_kind: str,
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Add canonical fields and report missing required source columns."""
    mapping = _resolve_columns(rows, aliases)
    missing = [field for field in required if field not in mapping]
    if missing:
        _issue(
            issues,
            "ERROR",
            f"{source_kind.upper()}_MISSING_COLUMNS",
            f"Missing required columns: {', '.join(missing)}.",
        )
    output: list[dict[str, Any]] = []
    for row in rows:
        canonical = dict(row)
        for target, source in mapping.items():
            canonical[target] = row.get(source, "")
        output.append(canonical)
    return output, mapping


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    list_id: Any = "",
    participant_id: str = "",
    session_id: str = "",
) -> None:
    """Append one consistently shaped QC issue."""
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
    source_kind: str,
    issues: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index one-row-per-participant sources and report missing or duplicate keys."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _participant_key(row)
        if not all(key):
            _issue(
                issues,
                "ERROR",
                f"{source_kind.upper()}_MISSING_KEY",
                "Participant or session ID is blank.",
                row.get("list_id", ""),
                key[0],
                key[1],
            )
            continue
        groups[key].append(row)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in groups.items():
        if len(group) != 1:
            _issue(
                issues,
                "ERROR",
                f"{source_kind.upper()}_DUPLICATE",
                f"Expected one {source_kind} row; found {len(group)}.",
                group[0].get("list_id", ""),
                key[0],
                key[1],
            )
        else:
            index[key] = group[0]
    return index


def _reported_list_matches(reported: Any, containing: Any) -> bool:
    """Compare values such as list1 and 1 without changing the raw field."""
    text = str(reported or "").strip().lower()
    if not text:
        return True
    return text.removeprefix("list") == str(containing).strip().lower().removeprefix("list")


def _is_attention(row: dict[str, Any], config: dict[str, Any]) -> bool:
    """Identify attention trials from a Boolean marker or configured trial type."""
    explicit = _parse_bool(row.get("is_attention_check"))
    if explicit is not None:
        return explicit
    trial_type = str(row.get("trial_type", "")).strip().lower()
    accepted = {str(value).strip().lower() for value in config.get("attention_trial_values", [])}
    return bool(trial_type and trial_type in accepted)


def _attention_result(
    row: dict[str, Any],
    allowed_ratings: set[int],
) -> dict[str, Any]:
    """Evaluate a MOS attention check using expected and submitted scale values."""
    expected = _parse_rating(row.get("expected_response"), allowed_ratings)
    observed = _parse_rating(row.get("mos_rating"), allowed_ratings)
    explicit = _parse_bool(row.get("attention_check_passed"))
    computed = expected == observed if expected is not None and observed is not None else None
    final = computed if computed is not None else explicit
    return {
        "expected_rating": expected if expected is not None else "",
        "observed_rating": observed if observed is not None else "",
        "computed_pass": computed,
        "explicit_pass": explicit,
        "final_pass": final,
        "explicit_computed_disagreement": (
            explicit is not None and computed is not None and explicit != computed
        ),
    }


def _duplicate_rows(
    rows: list[dict[str, Any]],
    keys: list[str],
) -> list[dict[str, Any]]:
    """Return source locations for every duplicated MOS response key."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for values, group in groups.items():
        if len(group) <= 1:
            continue
        shared = {key: value for key, value in zip(keys, values)}
        for row in group:
            output.append(
                {
                    **shared,
                    "duplicate_count": len(group),
                    "source_file": row.get("source_file", ""),
                    "source_row": row.get("source_row", ""),
                }
            )
    return output


def _timing_audit(
    sessions: dict[tuple[str, str], list[dict[str, Any]]],
    included_keys: set[tuple[str, str]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create non-blocking participant timing diagnostics."""
    minimum = _number(settings.get("minimum_session_duration_sec"))
    maximum = _number(settings.get("maximum_session_duration_sec"))
    maximum_trial = _number(settings.get("maximum_trial_response_time_sec"))
    output: list[dict[str, Any]] = []
    for key, session_rows in sorted(sessions.items()):
        def timing_order(row: dict[str, Any]) -> tuple[bool, float]:
            item_order = _number(row.get("item_order"))
            source_row = _number(row.get("source_row")) or 0
            return (
                item_order is None,
                item_order if item_order is not None else source_row,
            )

        rows = sorted(
            session_rows,
            key=timing_order,
        )
        response_times = [_number(row.get("response_time_sec")) for row in rows]
        elapsed = [_number(row.get("study_elapsed_sec")) for row in rows]
        valid_elapsed = [value for value in elapsed if value is not None]
        negative = sum(value is not None and value < 0 for value in response_times)
        excessive = sum(
            value is not None and maximum_trial is not None and value > maximum_trial
            for value in response_times
        )
        nonmonotonic = sum(
            previous is not None and current is not None and current < previous
            for previous, current in zip(elapsed, elapsed[1:])
        )
        duration = max(valid_elapsed) if valid_elapsed else None
        reasons: list[str] = []
        if any(value is None for value in response_times):
            reasons.append("missing_or_invalid_response_time")
        if negative:
            reasons.append("negative_response_time")
        if excessive:
            reasons.append("response_time_above_configured_maximum")
        if nonmonotonic:
            reasons.append("nonmonotonic_study_elapsed_sec")
        if duration is None:
            reasons.append("session_duration_unavailable")
        if duration is not None and minimum is not None and duration < minimum:
            reasons.append("session_below_configured_minimum")
        if duration is not None and maximum is not None and duration > maximum:
            reasons.append("session_above_configured_maximum")
        output.append(
            {
                "list_id": rows[0].get("list_id", "") if rows else "",
                "participant_id": key[0],
                "session_id": key[1],
                "included_in_analysis": key in included_keys,
                "trial_rows": len(rows),
                "missing_or_invalid_response_time_rows": sum(
                    value is None for value in response_times
                ),
                "negative_response_time_rows": negative,
                "response_time_above_configured_maximum_rows": excessive,
                "nonmonotonic_study_elapsed_steps": nonmonotonic,
                "session_duration_sec": duration if duration is not None else "",
                "timing_qc_status": "REVIEW" if reasons else "OK",
                "timing_qc_reasons": "; ".join(reasons),
            }
        )
    return output


def _qc_summary(
    status: str,
    participants: list[dict[str, Any]],
    eligible_ratings: int,
    attention_overrides: int,
    incomplete_session_overrides: int,
    eligibility_field_overrides: int,
    technical_exclusions: int,
    structural_errors: list[dict[str, Any]],
) -> str:
    """Return a short result and the next file to inspect when blocked."""
    included = sum(bool(row["include"]) for row in participants)
    total = len(participants)
    if status == "BLOCKED":
        codes = ", ".join(sorted({str(row["code"]) for row in structural_errors}))
        return (
            f"MOS QC BLOCKED: {included}/{total} participants are eligible and "
            f"{eligible_ratings} ratings remain; resolve {codes or 'the reported structural errors'} "
            "in 01_qc/qc_issues.csv before creating mos_analysis_ready.csv."
        )
    details: list[str] = []
    if attention_overrides:
        details.append(f"{attention_overrides} attention override(s)")
    if incomplete_session_overrides:
        details.append(
            f"{incomplete_session_overrides} incomplete-session override(s)"
        )
    if eligibility_field_overrides:
        details.append(f"{eligibility_field_overrides} eligibility-field override(s)")
    if technical_exclusions:
        details.append(f"{technical_exclusions} technical exclusion(s)")
    suffix = f" Documented exceptions: {', '.join(details)}." if details else ""
    label = "PASSED WITH EXCLUSIONS" if any(not row["include"] for row in participants) else "PASSED"
    return (
        f"MOS QC {label}: {included}/{total} participants are eligible and "
        f"{eligible_ratings} ratings were written to mos_analysis_ready.csv.{suffix}"
    )


def run_mos_pipeline(
    results_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate MOS results and write one long-format CSV for ordinal analysis."""
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {output_root}. Use a new directory for each run."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "OUTPUT_GUIDE.md").write_text(OUTPUT_GUIDE, encoding="utf-8")

    scale_values = config.get("rating_scale", [])
    allowed_ratings: set[int] = set()
    for value in scale_values:
        parsed = _parse_rating(value, set(range(-1000, 1001)))
        if parsed is None:
            raise ValueError("rating_scale must contain only whole numbers.")
        allowed_ratings.add(parsed)
    if allowed_ratings != {1, 2, 3, 4, 5}:
        raise ValueError("This MOS pipeline requires rating_scale to be exactly [1, 2, 3, 4, 5].")

    issues: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    collected: dict[str, list[dict[str, Any]]] = {kind: [] for kind in SOURCE_KINDS}
    for list_id in config["list_ids"]:
        list_directory = results_root / config["list_directory_pattern"].format(
            list_id=list_id
        )
        if not list_directory.exists():
            _issue(
                issues,
                "ERROR",
                "MISSING_LIST_DIRECTORY",
                f"List directory does not exist: {list_directory}",
                list_id,
            )
        for source_kind in SOURCE_KINDS:
            rows, source_inventory = _read_folder(
                list_directory / source_kind,
                list_id,
                source_kind,
            )
            collected[source_kind].extend(rows)
            inventory.extend(source_inventory)
            if source_kind != "postquestionnaire" and not rows:
                _issue(
                    issues,
                    "ERROR",
                    "MISSING_SOURCE_ROWS",
                    f"No CSV rows found in {source_kind}/.",
                    list_id,
                )

    aliases = config["column_aliases"]
    main, main_mapping = _canonicalize(
        collected["main"], aliases, REQUIRED_MAIN_FIELDS, "main", issues
    )
    prescreen, prescreen_mapping = _canonicalize(
        collected["prescreen"],
        aliases,
        ("participant_id", "session_id", "native_english", "headphones"),
        "prescreen",
        issues,
    )
    sound, sound_mapping = _canonicalize(
        collected["sound_check"],
        aliases,
        ("participant_id", "session_id", "heard_clearly"),
        "sound_check",
        issues,
    )
    if collected["postquestionnaire"]:
        questionnaire, questionnaire_mapping = _canonicalize(
            collected["postquestionnaire"],
            aliases,
            ("participant_id", "session_id"),
            "postquestionnaire",
            issues,
        )
    else:
        questionnaire, questionnaire_mapping = [], {}

    for source_kind, rows in (
        ("main", main),
        ("prescreen", prescreen),
        ("sound_check", sound),
        ("postquestionnaire", questionnaire),
    ):
        for row in rows:
            if not _reported_list_matches(row.get("reported_list_id"), row.get("list_id")):
                _issue(
                    issues,
                    "ERROR",
                    "REPORTED_LIST_ID_MISMATCH",
                    f"{source_kind} row reports list_id={row.get('reported_list_id', '')!r}, "
                    f"but its folder is list {row.get('list_id', '')}.",
                    row.get("list_id", ""),
                    str(row.get("participant_id", "")),
                    str(row.get("session_id", "")),
                )

    prescreen_index = _index_unique(prescreen, "prescreen", issues)
    sound_index = _index_unique(sound, "sound_check", issues)
    questionnaire_index = _index_unique(questionnaire, "postquestionnaire", issues)
    sessions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in main:
        sessions[_participant_key(row)].append(row)

    for row in main:
        if _is_attention(row, config):
            continue
        blank = [
            field
            for field in REQUIRED_MAIN_FIELDS
            if not str(row.get(field, "")).strip()
        ]
        if blank:
            _issue(
                issues,
                "ERROR",
                "MAIN_BLANK_REQUIRED_VALUE",
                f"Blank required values: {', '.join(blank)} (source row {row.get('source_row', '')}).",
                row.get("list_id", ""),
                str(row.get("participant_id", "")),
                str(row.get("session_id", "")),
            )

    participant_appearances: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for key, rows in sessions.items():
        participant_appearances[key[0]].add((str(rows[0].get("list_id", "")), key[1]))
    for participant_id, appearances in participant_appearances.items():
        if participant_id and len(appearances) > 1:
            _issue(
                issues,
                "ERROR",
                "PARTICIPANT_ACROSS_LISTS_OR_SESSIONS",
                f"Participant appears in multiple list/session combinations: {sorted(appearances)}",
                participant_id=participant_id,
            )

    attention_override_index: dict[tuple[str, str], dict[str, Any]] = {}
    for override in config.get("attention_check_include_overrides", []):
        key = (
            str(override.get("participant_id", "")).strip(),
            str(override.get("session_id", "")).strip(),
        )
        reason = str(override.get("reason", "")).strip()
        if not all(key) or not reason:
            raise ValueError(
                "Each attention override requires participant_id, session_id, and reason."
            )
        if key in attention_override_index:
            raise ValueError(f"Duplicate attention override for {key}.")
        attention_override_index[key] = override

    incomplete_override_index: dict[tuple[str, str], dict[str, Any]] = {}
    for override in config.get("incomplete_session_include_overrides", []):
        key = (
            str(override.get("participant_id", "")).strip(),
            str(override.get("session_id", "")).strip(),
        )
        reason = str(override.get("reason", "")).strip()
        declared_expected = override.get("expected_ratings")
        declared_observed = override.get("observed_ratings")
        valid_counts = all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (declared_expected, declared_observed)
        )
        if not all(key) or not reason or not valid_counts:
            raise ValueError(
                "Each incomplete-session override requires participant_id, session_id, "
                "non-negative integer expected_ratings and observed_ratings, and reason."
            )
        if declared_observed >= declared_expected:
            raise ValueError(
                f"Incomplete-session override for {key} must have observed_ratings "
                "below expected_ratings."
            )
        if key in incomplete_override_index:
            raise ValueError(f"Duplicate incomplete-session override for {key}.")
        incomplete_override_index[key] = override

    eligibility_override_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for override in config.get("eligibility_field_overrides", []):
        participant_id = str(override.get("participant_id", "")).strip()
        session_id = str(override.get("session_id", "")).strip()
        field = str(override.get("field", "")).strip()
        reason = str(override.get("reason", "")).strip()
        corrected_value = _parse_bool(override.get("corrected_value"))
        if (
            not participant_id
            or not session_id
            or field not in ELIGIBILITY_FIELDS
            or corrected_value is None
            or not reason
        ):
            raise ValueError(
                "Each eligibility-field override requires participant_id, session_id, "
                f"field in {ELIGIBILITY_FIELDS}, a Boolean corrected_value, and reason."
            )
        key = (participant_id, session_id, field)
        if key in eligibility_override_index:
            raise ValueError(f"Duplicate eligibility-field override for {key}.")
        eligibility_override_index[key] = override

    participant_qc: list[dict[str, Any]] = []
    included_keys: set[tuple[str, str]] = set()
    incomplete_override_applied_keys: set[tuple[str, str]] = set()
    attention_audit: list[dict[str, Any]] = []
    incomplete_override_audit: list[dict[str, Any]] = []
    eligibility_override_audit: list[dict[str, Any]] = []
    expected_ratings = config.get("expected_ratings_per_participant")
    expected_checks = config.get("expected_attention_checks_per_participant")
    for key, rows in sorted(sessions.items()):
        participant_id, session_id = key
        list_id = rows[0].get("list_id", "")
        pre = prescreen_index.get(key)
        snd = sound_index.get(key)
        post = questionnaire_index.get(key)
        attention_rows = [row for row in rows if _is_attention(row, config)]
        rating_rows = [row for row in rows if not _is_attention(row, config)]
        evaluations = [_attention_result(row, allowed_ratings) for row in attention_rows]
        final_values = [evaluation["final_pass"] for evaluation in evaluations]
        attention_pass = (
            all(value is True for value in final_values)
            if attention_rows and all(value is not None for value in final_values)
            else None
        )
        attention_override = attention_override_index.get(key)
        attention_override_list = (
            str(attention_override.get("list_id", "")).strip()
            if attention_override
            else ""
        )
        attention_override_valid = bool(
            attention_override
            and (
                not attention_override_list
                or attention_override_list == str(list_id)
            )
        )
        attention_override_applied = bool(
            attention_pass is False and attention_override_valid
        )
        attention_override_reason = (
            str(attention_override.get("reason", "")).strip()
            if attention_override_applied
            else ""
        )

        incomplete_override = incomplete_override_index.get(key)
        incomplete_override_list = (
            str(incomplete_override.get("list_id", "")).strip()
            if incomplete_override
            else ""
        )
        incomplete_override_valid = bool(
            incomplete_override
            and (
                not incomplete_override_list
                or incomplete_override_list == str(list_id)
            )
        )
        incomplete_override_counts_match = bool(
            incomplete_override_valid
            and expected_ratings is not None
            and int(incomplete_override["expected_ratings"]) == int(expected_ratings)
            and int(incomplete_override["observed_ratings"]) == len(rating_rows)
        )
        incomplete_override_applied = bool(
            expected_ratings is not None
            and len(rating_rows) != int(expected_ratings)
            and incomplete_override_counts_match
        )
        incomplete_override_reason = (
            str(incomplete_override.get("reason", "")).strip()
            if incomplete_override_applied
            else ""
        )

        blockers: list[str] = []
        raw_eligibility = {
            "native_english": _parse_bool(pre.get("native_english")) if pre else None,
            "headphones": _parse_bool(pre.get("headphones")) if pre else None,
            "heard_clearly": _parse_bool(snd.get("heard_clearly")) if snd else None,
        }
        effective_eligibility = dict(raw_eligibility)
        eligibility_fields_applied: list[str] = []
        eligibility_reasons: list[str] = []
        for field in ELIGIBILITY_FIELDS:
            eligibility_override = eligibility_override_index.get(
                (participant_id, session_id, field)
            )
            if not eligibility_override:
                continue
            eligibility_override_list = str(
                eligibility_override.get("list_id", "")
            ).strip()
            list_matches = (
                not eligibility_override_list
                or eligibility_override_list == str(list_id)
            )
            corrected_value = _parse_bool(
                eligibility_override.get("corrected_value")
            )
            applied = bool(list_matches and corrected_value is not None)
            if applied:
                effective_eligibility[field] = corrected_value
                eligibility_fields_applied.append(field)
                reason = str(eligibility_override.get("reason", "")).strip()
                eligibility_reasons.append(reason)
                _issue(
                    issues,
                    "WARNING",
                    "ELIGIBILITY_FIELD_OVERRIDDEN",
                    f"{field}: {reason}",
                    list_id,
                    participant_id,
                    session_id,
                )
            else:
                _issue(
                    issues,
                    "WARNING",
                    "ELIGIBILITY_OVERRIDE_LIST_MISMATCH",
                    f"Configured {field} override belongs to another list.",
                    list_id,
                    participant_id,
                    session_id,
                )
            eligibility_override_audit.append(
                {
                    "list_id": list_id,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "field": field,
                    "raw_value": raw_eligibility[field],
                    "corrected_value": corrected_value,
                    "effective_value": effective_eligibility[field],
                    "override_applied": applied,
                    "reviewed_at": eligibility_override.get("reviewed_at", ""),
                    "reason": eligibility_override.get("reason", ""),
                }
            )
        native = effective_eligibility["native_english"]
        headphones = effective_eligibility["headphones"]
        heard = effective_eligibility["heard_clearly"]
        if pre is None:
            blockers.append("missing prescreen row")
        elif str(pre.get("list_id", "")) != str(list_id):
            blockers.append("prescreen list does not match main data")
        if snd is None:
            blockers.append("missing sound-check row")
        elif str(snd.get("list_id", "")) != str(list_id):
            blockers.append("sound-check list does not match main data")
        if native is not True:
            blockers.append("native_english is not TRUE")
        if headphones is not True:
            blockers.append("headphones is not TRUE")
        if heard is not True:
            blockers.append("heard_clearly is not TRUE")
        if (
            expected_ratings is not None
            and len(rating_rows) != int(expected_ratings)
            and not incomplete_override_applied
        ):
            blockers.append(
                f"expected {expected_ratings} MOS ratings; found {len(rating_rows)}"
            )
        if expected_checks is not None and len(attention_rows) != int(expected_checks):
            blockers.append(
                f"expected {expected_checks} attention checks; found {len(attention_rows)}"
            )
        if (
            attention_rows
            and attention_pass is False
            and not attention_override_applied
        ):
            blockers.append("failed at least one attention check")
        if attention_rows and attention_pass is None:
            blockers.append("attention-check result is unknown")
        if any(evaluation["explicit_computed_disagreement"] for evaluation in evaluations):
            _issue(
                issues,
                "WARNING",
                "ATTENTION_RESULT_DISAGREEMENT",
                "Stored attention pass/fail disagrees with expected-rating matching.",
                list_id,
                participant_id,
                session_id,
            )
        if attention_override_applied:
            _issue(
                issues,
                "WARNING",
                "ATTENTION_FAILURE_OVERRIDDEN",
                attention_override_reason,
                list_id,
                participant_id,
                session_id,
            )
        elif attention_override and not attention_override_valid:
            _issue(
                issues,
                "WARNING",
                "ATTENTION_OVERRIDE_LIST_MISMATCH",
                "Configured attention override belongs to another list.",
                list_id,
                participant_id,
                session_id,
            )
        if incomplete_override_applied:
            incomplete_override_applied_keys.add(key)
            _issue(
                issues,
                "WARNING",
                "INCOMPLETE_SESSION_OVERRIDDEN",
                incomplete_override_reason,
                list_id,
                participant_id,
                session_id,
            )
        elif incomplete_override and not incomplete_override_valid:
            _issue(
                issues,
                "WARNING",
                "INCOMPLETE_OVERRIDE_LIST_MISMATCH",
                "Configured incomplete-session override belongs to another list.",
                list_id,
                participant_id,
                session_id,
            )
        elif incomplete_override and not incomplete_override_counts_match:
            _issue(
                issues,
                "ERROR",
                "INCOMPLETE_OVERRIDE_COUNT_MISMATCH",
                "Configured expected/observed rating counts do not match the session data.",
                list_id,
                participant_id,
                session_id,
            )
        if incomplete_override:
            incomplete_override_audit.append(
                {
                    "list_id": list_id,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "configured_expected_ratings": incomplete_override.get(
                        "expected_ratings", ""
                    ),
                    "configured_observed_ratings": incomplete_override.get(
                        "observed_ratings", ""
                    ),
                    "actual_observed_ratings": len(rating_rows),
                    "override_applied": incomplete_override_applied,
                    "reviewed_at": incomplete_override.get("reviewed_at", ""),
                    "reason": incomplete_override.get("reason", ""),
                }
            )
        if post is None:
            _issue(
                issues,
                "WARNING",
                "MISSING_POSTQUESTIONNAIRE",
                "No postquestionnaire row matched this participant/session.",
                list_id,
                participant_id,
                session_id,
            )
        include = not blockers
        if include:
            included_keys.add(key)
        else:
            _issue(
                issues,
                "ERROR",
                "PARTICIPANT_EXCLUDED",
                "; ".join(blockers),
                list_id,
                participant_id,
                session_id,
            )
        participant_qc.append(
            {
                "list_id": list_id,
                "participant_id": participant_id,
                "session_id": session_id,
                "native_english_raw": raw_eligibility["native_english"],
                "native_english": native,
                "headphones_raw": raw_eligibility["headphones"],
                "headphones": headphones,
                "heard_clearly_raw": raw_eligibility["heard_clearly"],
                "heard_clearly": heard,
                "eligibility_override_applied": bool(eligibility_fields_applied),
                "eligibility_override_fields": " | ".join(
                    eligibility_fields_applied
                ),
                "eligibility_override_reason": " | ".join(eligibility_reasons),
                "mos_rating_rows": len(rating_rows),
                "incomplete_session_override_applied": incomplete_override_applied,
                "incomplete_session_override_reason": incomplete_override_reason,
                "attention_check_rows": len(attention_rows),
                "attention_checks_passed": attention_pass,
                "attention_override_applied": attention_override_applied,
                "attention_override_reason": attention_override_reason,
                "postquestionnaire_present": post is not None,
                "include": include,
                "exclusion_reasons": "; ".join(blockers),
            }
        )
        for row, evaluation in zip(attention_rows, evaluations):
            attention_audit.append(
                {
                    "list_id": list_id,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "item_order": row.get("item_order", ""),
                    "item_id": row.get("item_id", ""),
                    "question": row.get("question", ""),
                    **evaluation,
                    "attention_override_applied": attention_override_applied,
                    "attention_override_reason": attention_override_reason,
                    "source_file": row.get("source_file", ""),
                    "source_row": row.get("source_row", ""),
                }
            )

    for key in sorted(set(attention_override_index) - set(sessions)):
        _issue(
            issues,
            "WARNING",
            "ATTENTION_OVERRIDE_NOT_FOUND",
            "Configured attention override has no matching main participant/session.",
            participant_id=key[0],
            session_id=key[1],
        )
    for key in sorted(set(incomplete_override_index) - set(sessions)):
        _issue(
            issues,
            "WARNING",
            "INCOMPLETE_OVERRIDE_NOT_FOUND",
            "Configured incomplete-session override has no matching main participant/session.",
            participant_id=key[0],
            session_id=key[1],
        )
    session_keys_with_eligibility_overrides = {
        (participant_id, session_id)
        for participant_id, session_id, _field in eligibility_override_index
    }
    for key in sorted(session_keys_with_eligibility_overrides - set(sessions)):
        _issue(
            issues,
            "WARNING",
            "ELIGIBILITY_OVERRIDE_NOT_FOUND",
            "Configured eligibility-field override has no matching main participant/session.",
            participant_id=key[0],
            session_id=key[1],
        )

    for source_name, index in (
        ("prescreen", prescreen_index),
        ("sound_check", sound_index),
        ("postquestionnaire", questionnaire_index),
    ):
        for key in sorted(set(index) - set(sessions)):
            row = index[key]
            _issue(
                issues,
                "WARNING",
                "ORPHAN_SOURCE_ROW",
                f"{source_name} row has no matching main participant/session.",
                row.get("list_id", ""),
                key[0],
                key[1],
            )

    expected_participants = config.get("expected_participants_per_list")
    if expected_participants is not None:
        for list_id in config["list_ids"]:
            count = sum(
                row["include"] and str(row["list_id"]) == str(list_id)
                for row in participant_qc
            )
            if count != int(expected_participants):
                _issue(
                    issues,
                    "ERROR",
                    "ELIGIBLE_PARTICIPANT_COUNT",
                    f"Expected {expected_participants} eligible participants; found {count}.",
                    list_id,
                )

    experimental_rows = [row for row in main if not _is_attention(row, config)]
    if main and "is_attention_check" not in main_mapping and "trial_type" not in main_mapping:
        _issue(
            issues,
            "WARNING",
            "ATTENTION_COLUMN_NOT_FOUND",
            "No attention-check marker column was found; configure an alias if checks exist.",
        )
    duplicates = _duplicate_rows(
        experimental_rows,
        [
            "list_id",
            "participant_id",
            "session_id",
            "item_id",
            "speaker_id",
            "condition",
        ],
    )
    if duplicates:
        _issue(
            issues,
            "ERROR",
            "DUPLICATE_MOS_RATINGS",
            f"Found {len(duplicates)} rows belonging to duplicate MOS response keys.",
        )

    rating_audit: list[dict[str, Any]] = []
    for row in experimental_rows:
        parsed = _parse_rating(row.get("mos_rating"), allowed_ratings)
        valid = parsed is not None
        rating_audit.append(
            {
                "list_id": row.get("list_id", ""),
                "participant_id": row.get("participant_id", ""),
                "session_id": row.get("session_id", ""),
                "item_id": row.get("item_id", ""),
                "condition": row.get("condition", ""),
                "mos_rating": row.get("mos_rating", ""),
                "valid_rating": valid,
                "validation_status": "VALID" if valid else "INVALID_OR_OUT_OF_RANGE",
                "source_file": row.get("source_file", ""),
                "source_row": row.get("source_row", ""),
            }
        )
        if not valid:
            _issue(
                issues,
                "ERROR",
                "INVALID_MOS_RATING",
                f"MOS rating must be an integer from 1 to 5; found {row.get('mos_rating', '')!r}.",
                row.get("list_id", ""),
                str(row.get("participant_id", "")),
                str(row.get("session_id", "")),
            )

    eligible_before_exclusions = [
        row for row in experimental_rows if _participant_key(row) in included_keys
    ]
    exclusion_audit: list[dict[str, Any]] = []
    excluded_locations: set[tuple[str, str]] = set()
    exclusion_keys: set[tuple[str, str, str, str, str]] = set()
    for override in config.get("technical_exclusion_overrides", []):
        required = (
            "list_id",
            "participant_id",
            "session_id",
            "item_id",
            "speaker_id",
            "condition",
            "reason",
        )
        missing = [field for field in required if not str(override.get(field, "")).strip()]
        if missing:
            raise ValueError(
                f"Technical exclusion is missing required fields: {', '.join(missing)}."
            )
        key = tuple(str(override[field]).strip() for field in required[:-1])
        if key in exclusion_keys:
            raise ValueError(f"Duplicate technical exclusion: {key}.")
        exclusion_keys.add(key)
        matches = [
            row
            for row in experimental_rows
            if all(
                str(row.get(field, "")) == str(override[field]).strip()
                for field in required[:-1]
            )
        ]
        participant_included = (
            str(override["participant_id"]).strip(),
            str(override["session_id"]).strip(),
        ) in included_keys
        applied = len(matches) == 1 and participant_included
        if len(matches) != 1:
            _issue(
                issues,
                "ERROR",
                "TECHNICAL_EXCLUSION_MATCH_COUNT",
                f"Technical exclusion matched {len(matches)} rows; expected exactly one.",
                override["list_id"],
                str(override["participant_id"]),
                str(override["session_id"]),
            )
        elif participant_included:
            excluded_locations.add(
                (str(matches[0].get("source_file", "")), str(matches[0].get("source_row", "")))
            )
            _issue(
                issues,
                "WARNING",
                "TECHNICAL_MOS_RATING_EXCLUDED",
                str(override["reason"]).strip(),
                override["list_id"],
                str(override["participant_id"]),
                str(override["session_id"]),
            )
        matched = matches[0] if len(matches) == 1 else {}
        exclusion_audit.append(
            {
                "list_id": override["list_id"],
                "participant_id": override["participant_id"],
                "session_id": override["session_id"],
                "item_id": override["item_id"],
                "speaker_id": override["speaker_id"],
                "condition": override["condition"],
                "mos_rating": matched.get("mos_rating", ""),
                "matched_rows": len(matches),
                "participant_included": participant_included,
                "exclusion_applied": applied,
                "reviewed_at": override.get("reviewed_at", ""),
                "reason": override["reason"],
                "source_file": matched.get("source_file", ""),
                "source_row": matched.get("source_row", ""),
            }
        )

    eligible_rows = [
        row
        for row in eligible_before_exclusions
        if (str(row.get("source_file", "")), str(row.get("source_row", "")))
        not in excluded_locations
    ]
    if not eligible_rows:
        _issue(
            issues,
            "ERROR",
            "NO_ELIGIBLE_RATINGS",
            "No eligible experimental MOS ratings remain after participant QC and exclusions.",
        )

    expected_conditions = [str(value) for value in config.get("expected_conditions", [])]
    list_condition_counts = Counter(
        (str(row.get("list_id", "")), str(row.get("condition", "")))
        for row in eligible_rows
    )
    list_condition_coverage: list[dict[str, Any]] = []
    for list_id in config["list_ids"]:
        observed = {
            condition
            for current_list, condition in list_condition_counts
            if current_list == str(list_id)
        }
        for condition in expected_conditions:
            count = list_condition_counts[(str(list_id), condition)]
            list_condition_coverage.append(
                {
                    "list_id": list_id,
                    "condition": condition,
                    "rating_count": count,
                    "present": count > 0,
                }
            )
            if count == 0 and observed:
                _issue(
                    issues,
                    "ERROR",
                    "MISSING_LIST_CONDITION",
                    f"Expected condition is absent: {condition}",
                    list_id,
                )
        for condition in sorted(observed - set(expected_conditions)):
            _issue(
                issues,
                "ERROR",
                "UNEXPECTED_CONDITION",
                f"Unexpected condition found: {condition}",
                list_id,
            )

    stimulus_condition_groups: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in eligible_rows:
        stimulus_condition_groups[
            (
                str(row.get("item_id", "")),
                str(row.get("speaker_id", "")),
                str(row.get("condition", "")),
            )
        ].append(row)
    expected_cell_count = config.get("expected_ratings_per_stimulus_condition")
    stimulus_condition_coverage: list[dict[str, Any]] = []
    for (item_id, speaker_id, condition), rows in sorted(
        stimulus_condition_groups.items()
    ):
        participants = {str(row.get("participant_id", "")) for row in rows}
        sessions_in_group = {str(row.get("session_id", "")) for row in rows}
        list_ids_in_group = {str(row.get("list_id", "")) for row in rows}
        complete = (
            expected_cell_count is None
            or (
                len(rows) == int(expected_cell_count)
                and len(participants) == int(expected_cell_count)
                and len(sessions_in_group) == int(expected_cell_count)
            )
        )
        approved_missing_sessions: list[str] = []
        if expected_cell_count is not None and len(list_ids_in_group) == 1:
            group_list_id = next(iter(list_ids_in_group))
            for incomplete_key in sorted(incomplete_override_applied_keys):
                incomplete_rows = sessions[incomplete_key]
                if not incomplete_rows or str(incomplete_rows[0].get("list_id", "")) != group_list_id:
                    continue
                has_rating = any(
                    not _is_attention(row, config)
                    and str(row.get("item_id", "")) == item_id
                    and str(row.get("speaker_id", "")) == speaker_id
                    and str(row.get("condition", "")) == condition
                    for row in incomplete_rows
                )
                if not has_rating:
                    approved_missing_sessions.append(
                        f"{incomplete_key[0]} / {incomplete_key[1]}"
                    )
        approved_shortfall = bool(
            not complete
            and expected_cell_count is not None
            and len(rows) < int(expected_cell_count)
            and len(rows) + len(approved_missing_sessions) == int(expected_cell_count)
            and len(participants) + len(approved_missing_sessions)
            == int(expected_cell_count)
            and len(sessions_in_group) + len(approved_missing_sessions)
            == int(expected_cell_count)
        )
        stimulus_condition_coverage.append(
            {
                "list_ids": " | ".join(sorted(list_ids_in_group)),
                "item_id": item_id,
                "speaker_id": speaker_id,
                "condition": condition,
                "observed_ratings": len(rows),
                "expected_ratings": expected_cell_count if expected_cell_count is not None else "",
                "distinct_participants": len(participants),
                "distinct_sessions": len(sessions_in_group),
                "complete": complete,
                "approved_shortfall": approved_shortfall,
                "approved_missing_sessions": " | ".join(
                    approved_missing_sessions
                ),
            }
        )
        if approved_shortfall:
            _issue(
                issues,
                "WARNING",
                "STIMULUS_CONDITION_SHORTFALL_OVERRIDDEN",
                f"{item_id} / {speaker_id} / {condition}: expected {expected_cell_count} "
                f"independent ratings; found {len(rows)} because of an approved "
                "incomplete session.",
                next(iter(list_ids_in_group)),
            )
        elif not complete:
            _issue(
                issues,
                "ERROR",
                "STIMULUS_CONDITION_COUNT",
                f"{item_id} / {speaker_id} / {condition}: expected {expected_cell_count} independent ratings; found {len(rows)}.",
            )

    item_metadata_audit: list[dict[str, Any]] = []
    for item_id in sorted({str(row.get("item_id", "")) for row in eligible_rows}):
        rows = [row for row in eligible_rows if str(row.get("item_id", "")) == item_id]
        speakers = sorted({str(row.get("speaker_id", "")) for row in rows})
        references = sorted(
            {str(row.get("reference_text", "")) for row in rows if str(row.get("reference_text", ""))}
        )
        consistent = len(references) <= 1
        item_metadata_audit.append(
            {
                "item_id": item_id,
                "distinct_speakers": len(speakers),
                "speakers": " | ".join(speakers),
                "distinct_reference_texts": len(references),
                "consistent": consistent,
            }
        )
        if not consistent:
            _issue(
                issues,
                "ERROR",
                "ITEM_METADATA_CONFLICT",
                f"Item {item_id} has conflicting reference texts across speakers or conditions.",
            )

    timing_columns = {"response_time_sec", "study_elapsed_sec"} & set(main_mapping)
    timing_audit = (
        _timing_audit(sessions, included_keys, config.get("timing_qc", {}))
        if timing_columns
        else []
    )
    if not timing_columns:
        _issue(
            issues,
            "WARNING",
            "TIMING_COLUMNS_NOT_FOUND",
            "No response-time or elapsed-time column was found; timing QC was skipped.",
        )
    for row in timing_audit:
        if row["included_in_analysis"] and row["timing_qc_status"] == "REVIEW":
            _issue(
                issues,
                "WARNING",
                "SESSION_TIMING_REVIEW",
                str(row["timing_qc_reasons"]),
                row["list_id"],
                str(row["participant_id"]),
                str(row["session_id"]),
            )

    structural_codes = {
        "MISSING_LIST_DIRECTORY",
        "MISSING_SOURCE_ROWS",
        "MAIN_MISSING_COLUMNS",
        "MAIN_BLANK_REQUIRED_VALUE",
        "PRESCREEN_MISSING_COLUMNS",
        "SOUND_CHECK_MISSING_COLUMNS",
        "REPORTED_LIST_ID_MISMATCH",
        "PRESCREEN_MISSING_KEY",
        "SOUND_CHECK_MISSING_KEY",
        "POSTQUESTIONNAIRE_MISSING_KEY",
        "PRESCREEN_DUPLICATE",
        "SOUND_CHECK_DUPLICATE",
        "POSTQUESTIONNAIRE_DUPLICATE",
        "PARTICIPANT_ACROSS_LISTS_OR_SESSIONS",
        "ELIGIBLE_PARTICIPANT_COUNT",
        "DUPLICATE_MOS_RATINGS",
        "INVALID_MOS_RATING",
        "INCOMPLETE_OVERRIDE_COUNT_MISMATCH",
        "TECHNICAL_EXCLUSION_MATCH_COUNT",
        "NO_ELIGIBLE_RATINGS",
        "MISSING_LIST_CONDITION",
        "UNEXPECTED_CONDITION",
        "STIMULUS_CONDITION_COUNT",
        "ITEM_METADATA_CONFLICT",
    }
    structural_errors = [
        row
        for row in issues
        if row["severity"] == "ERROR" and row["code"] in structural_codes
    ]
    status = (
        "BLOCKED"
        if structural_errors
        else "PASSED_WITH_EXCLUSIONS"
        if any(not row["include"] for row in participant_qc)
        else "PASSED"
    )

    qc_root = output_root / "01_qc"
    _write_csv(qc_root / "source_file_inventory.csv", inventory)
    _write_csv(qc_root / "participant_qc.csv", participant_qc)
    _write_csv(qc_root / "qc_issues.csv", issues)
    _write_csv(qc_root / "rating_value_audit.csv", rating_audit)
    _write_csv(qc_root / "duplicate_ratings.csv", duplicates)
    _write_csv(qc_root / "attention_check_audit.csv", attention_audit)
    _write_csv(
        qc_root / "incomplete_session_override_audit.csv",
        incomplete_override_audit,
    )
    _write_csv(
        qc_root / "eligibility_field_override_audit.csv",
        eligibility_override_audit,
    )
    _write_csv(qc_root / "technical_exclusion_audit.csv", exclusion_audit)
    _write_csv(qc_root / "session_timing_qc.csv", timing_audit)
    _write_csv(qc_root / "list_condition_coverage.csv", list_condition_coverage)
    _write_csv(
        qc_root / "stimulus_condition_coverage.csv",
        stimulus_condition_coverage,
    )
    _write_csv(qc_root / "item_metadata_audit.csv", item_metadata_audit)

    applied_attention_overrides = sum(
        bool(row["attention_override_applied"]) for row in participant_qc
    )
    applied_incomplete_overrides = sum(
        bool(row["incomplete_session_override_applied"])
        for row in participant_qc
    )
    applied_eligibility_overrides = sum(
        bool(row["override_applied"]) for row in eligibility_override_audit
    )
    applied_exclusions = sum(bool(row["exclusion_applied"]) for row in exclusion_audit)
    summary = _qc_summary(
        status,
        participant_qc,
        len(eligible_rows),
        applied_attention_overrides,
        applied_incomplete_overrides,
        applied_eligibility_overrides,
        applied_exclusions,
        structural_errors,
    )
    (qc_root / "qc_summary.txt").write_text(summary + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root.resolve()),
        "output_root": str(output_root.resolve()),
        "status": status,
        "qc_summary": summary,
        "eligible_participants": len(included_keys),
        "analysis_ready_rows": 0 if status == "BLOCKED" else len(eligible_rows),
        "error_count": sum(row["severity"] == "ERROR" for row in issues),
        "warning_count": sum(row["severity"] == "WARNING" for row in issues),
        "attention_overrides_applied": applied_attention_overrides,
        "incomplete_session_overrides_applied": applied_incomplete_overrides,
        "eligibility_field_overrides_applied": applied_eligibility_overrides,
        "technical_exclusions_applied": applied_exclusions,
        "resolved_columns": {
            "main": main_mapping,
            "prescreen": prescreen_mapping,
            "sound_check": sound_mapping,
            "postquestionnaire": questionnaire_mapping,
        },
    }
    _write_json(output_root / "run_manifest.json", manifest)

    if status == "BLOCKED":
        raise MOSQCError(f"{summary} Details: {qc_root / 'qc_issues.csv'}")

    output_columns = [
        "participant_id",
        "session_id",
        "list_id",
        "item_order",
        "item_id",
        "audio_file",
        "condition",
        "speaker_id",
        "gender",
        "language",
        "reference_text",
        "question",
        "mos_rating",
        "plays_used",
        "response_time_sec",
        "study_elapsed_sec",
        "shown_at",
        "submitted_at",
        "source_file",
        "source_row",
    ]
    analysis_ready: list[dict[str, Any]] = []
    for row in eligible_rows:
        prepared = {column: row.get(column, "") for column in output_columns}
        prepared["mos_rating"] = _parse_rating(row.get("mos_rating"), allowed_ratings)
        analysis_ready.append(prepared)
    analysis_ready.sort(
        key=lambda row: (
            str(row["list_id"]),
            str(row["participant_id"]),
            str(row["session_id"]),
            _number(row["item_order"]) or 0,
        )
    )
    _write_csv(
        output_root / "02_preprocessed" / "mos_analysis_ready.csv",
        analysis_ready,
        output_columns,
    )
    return manifest
