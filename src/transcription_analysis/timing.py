from __future__ import annotations

from datetime import datetime
from typing import Any


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, float, int]:
        order = number(row.get("item_order"))
        source_row = number(row.get("source_row")) or 0
        return (0 if order is not None else 1, order if order is not None else source_row, int(source_row))

    return sorted(rows, key=key)


def _percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def session_timing_qc(
    sessions: dict[tuple[str, str], list[dict[str, Any]]],
    included_keys: set[tuple[str, str]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    minimum_duration = number(settings.get("minimum_session_duration_sec"))
    maximum_duration = number(settings.get("maximum_session_duration_sec"))
    maximum_response = number(settings.get("maximum_trial_response_time_sec"))
    for (participant_id, session_id), session_rows in sorted(sessions.items()):
        rows = _ordered(session_rows)
        response_times = [number(row.get("response_time_sec")) for row in rows]
        elapsed = [number(row.get("study_elapsed_sec")) for row in rows]
        valid_elapsed = [value for value in elapsed if value is not None]
        submitted = [timestamp(row.get("submitted_at")) for row in rows]
        valid_submitted = [value for value in submitted if value is not None]
        negative_response_rows = sum(value is not None and value < 0 for value in response_times)
        excessive_response_rows = sum(value is not None and maximum_response is not None and value > maximum_response for value in response_times)
        nonmonotonic_elapsed_steps = sum(
            current is not None and previous is not None and current < previous
            for previous, current in zip(elapsed, elapsed[1:])
        )
        nonmonotonic_submitted_steps = sum(
            current is not None and previous is not None and current < previous
            for previous, current in zip(submitted, submitted[1:])
        )
        timestamp_span = ""
        if len(valid_submitted) >= 2:
            timestamp_span = (max(valid_submitted) - min(valid_submitted)).total_seconds()
        if valid_elapsed:
            duration = max(valid_elapsed)
            duration_source = "max_study_elapsed_sec"
        elif timestamp_span != "":
            duration = timestamp_span
            duration_source = "submitted_at_span"
        else:
            duration = None
            duration_source = "unavailable"
        below_minimum = duration is not None and minimum_duration is not None and duration < minimum_duration
        above_maximum = duration is not None and maximum_duration is not None and duration > maximum_duration
        problems: list[str] = []
        if any(value is None for value in response_times):
            problems.append("missing_or_invalid_response_time")
        if negative_response_rows:
            problems.append("negative_response_time")
        if excessive_response_rows:
            problems.append("response_time_above_configured_maximum")
        if nonmonotonic_elapsed_steps:
            problems.append("nonmonotonic_study_elapsed_sec")
        if nonmonotonic_submitted_steps:
            problems.append("nonmonotonic_submitted_at")
        if below_minimum:
            problems.append("session_below_configured_minimum")
        if above_maximum:
            problems.append("session_above_configured_maximum")
        if duration is None:
            problems.append("session_duration_unavailable")
        output.append(
            {
                "list_id": rows[0].get("list_id", "") if rows else "",
                "participant_id": participant_id,
                "session_id": session_id,
                "included_in_analysis": (participant_id, session_id) in included_keys,
                "trial_rows": len(rows),
                "missing_or_invalid_response_time_rows": sum(value is None for value in response_times),
                "negative_response_time_rows": negative_response_rows,
                "response_time_above_configured_maximum_rows": excessive_response_rows,
                "nonmonotonic_study_elapsed_steps": nonmonotonic_elapsed_steps,
                "nonmonotonic_submitted_at_steps": nonmonotonic_submitted_steps,
                "max_study_elapsed_sec": max(valid_elapsed) if valid_elapsed else "",
                "submitted_at_span_sec": timestamp_span,
                "session_duration_sec": duration if duration is not None else "",
                "session_duration_source": duration_source,
                "below_configured_minimum": below_minimum,
                "above_configured_maximum": above_maximum,
                "relative_duration_outlier": False,
                "relative_duration_lower_bound_sec": "",
                "relative_duration_upper_bound_sec": "",
                "timing_qc_status": "REVIEW" if problems else "OK",
                "timing_qc_reasons": "; ".join(problems),
            }
        )
    durations = [float(row["session_duration_sec"]) for row in output if row["session_duration_sec"] != ""]
    if len(durations) >= 4:
        q1 = _percentile(durations, 0.25)
        q3 = _percentile(durations, 0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            for row in output:
                row["relative_duration_lower_bound_sec"] = lower_bound
                row["relative_duration_upper_bound_sec"] = upper_bound
                duration = number(row["session_duration_sec"])
                if duration is not None and (duration < lower_bound or duration > upper_bound):
                    row["relative_duration_outlier"] = True
                    row["timing_qc_status"] = "REVIEW"
                    reasons = [value for value in str(row["timing_qc_reasons"]).split("; ") if value]
                    reasons.append("relative_session_duration_outlier")
                    row["timing_qc_reasons"] = "; ".join(reasons)
    return output
