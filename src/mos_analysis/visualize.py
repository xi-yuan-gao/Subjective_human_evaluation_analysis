from __future__ import annotations

import csv
import hashlib
import html
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .summarize import (
    listener_mean_t_ci,
    summarize_conditions,
    summarize_rating_distribution,
)


BLUE = "#0072B2"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
TEXT = "#1F2937"
BACKGROUND = "#FFFFFF"
RATING_COLORS = {
    1: "#B2182B",
    2: "#EF8A62",
    3: "#D1D5DB",
    4: "#67A9CF",
    5: "#2166AC",
}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _text(
    x: float,
    y: float,
    value: Any,
    *,
    anchor: str = "start",
    size: int = 13,
    weight: int = 400,
    color: str = TEXT,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{_escape(value)}</text>'
    )


def _svg_document(
    width: int,
    height: int,
    title: str,
    description: str,
    body: list[str],
) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{_escape(title)}</title>',
            f'<desc id="desc">{_escape(description)}</desc>',
            f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in materialized:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _scale(value: float, start: float, width: float) -> float:
    """Map a MOS value from the fixed 1-5 scale into plot coordinates."""
    return start + (max(1.0, min(5.0, value)) - 1.0) / 4.0 * width


def _label_fits_segment(segment_width: float, label: str) -> bool:
    """Return whether a percentage label fits safely inside a bar segment."""
    estimated_text_width = len(label) * 6.5
    horizontal_padding = 10.0
    return segment_width >= estimated_text_width + horizontal_padding


def _ordered_conditions(
    rows: list[dict[str, Any]],
    expected_conditions: list[str],
    settings: dict[str, Any],
) -> list[str]:
    observed = {str(row.get("condition", "")) for row in rows}
    requested = [
        str(value)
        for value in settings.get("condition_order", expected_conditions)
    ]
    ordered = [value for value in requested if value in observed]
    ordered.extend(sorted(observed - set(ordered) - {""}))
    return ordered


def _condition_data(
    rows: list[dict[str, Any]],
    conditions: list[str],
    display_names: dict[str, str],
    confidence_level: float,
) -> list[dict[str, Any]]:
    output = summarize_conditions(
        rows,
        condition_order=conditions,
        confidence_level=confidence_level,
    )
    for row in output:
        condition = str(row["condition"])
        row["condition_display_name"] = display_names.get(condition, condition)
    return output


def _distribution_data(
    rows: list[dict[str, Any]],
    conditions: list[str],
    display_names: dict[str, str],
    rating_scale: list[int],
) -> list[dict[str, Any]]:
    output = summarize_rating_distribution(
        rows,
        condition_order=conditions,
        rating_scale=rating_scale,
    )
    for row in output:
        condition = str(row["condition"])
        row["condition_display_name"] = display_names.get(condition, condition)
    return output


def _listener_condition_data(
    rows: list[dict[str, Any]],
    conditions: list[str],
    display_names: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    list_ids: dict[tuple[str, str, str], str] = {}
    for row in rows:
        key = (
            str(row.get("participant_id", "")),
            str(row.get("session_id", "")),
            str(row.get("condition", "")),
        )
        grouped[key].append(float(row["mos_rating"]))
        list_ids[key] = str(row.get("list_id", ""))
    condition_rank = {condition: index for index, condition in enumerate(conditions)}
    output: list[dict[str, Any]] = []
    for (participant, session, condition), ratings in grouped.items():
        if condition not in condition_rank:
            continue
        output.append(
            {
                "participant_id": participant,
                "session_id": session,
                "list_id": list_ids[(participant, session, condition)],
                "condition": condition,
                "condition_display_name": display_names.get(condition, condition),
                "n_ratings": len(ratings),
                "listener_mean_mos": mean(ratings),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            condition_rank[str(row["condition"])],
            str(row["participant_id"]),
            str(row["session_id"]),
        ),
    )


def _speaker_condition_data(
    rows: list[dict[str, Any]],
    conditions: list[str],
    condition_display_names: dict[str, str],
    speaker_order: list[str],
    speaker_display_names: dict[str, str],
    confidence_level: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("speaker_id", "")), str(row.get("condition", "")))
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for speaker in speaker_order:
        for condition in conditions:
            group = grouped.get((speaker, condition), [])
            if not group:
                continue
            interval = listener_mean_t_ci(group, confidence_level=confidence_level)
            output.append(
                {
                    "speaker_id": speaker,
                    "speaker_display_name": speaker_display_names.get(speaker, speaker),
                    "condition": condition,
                    "condition_display_name": condition_display_names.get(condition, condition),
                    "n_ratings": len(group),
                    "n_listener_means": interval["n_listener_means"],
                    "mean_mos": interval["mean_mos"],
                    "mos_ci_low": interval["mos_ci_low"],
                    "mos_ci_high": interval["mos_ci_high"],
                    "confidence_level": confidence_level,
                    "ci_method": "listener_mean_student_t",
                }
            )
    return output


def _draw_mos_by_condition(data: list[dict[str, Any]], path: Path) -> None:
    width, height = 1000, 420
    left, right, top, bottom = 260.0, 110.0, 105.0, 65.0
    chart_width = width - left - right
    chart_height = height - top - bottom
    row_step = chart_height / max(1, len(data))
    body = [
        _text(18, 34, "Mean opinion score by condition", size=21, weight=500),
        _text(18, 60, "Points are listener-level means; whiskers are Student-t 95% confidence intervals. Higher is better.", size=12, color=GRAY),
    ]
    for rating in range(1, 6):
        x = _scale(float(rating), left, chart_width)
        body.append(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{top + chart_height:.1f}" stroke="{LIGHT_GRAY}" stroke-width="1"/>')
        body.append(_text(x, height - 38, rating, anchor="middle", size=12, color=GRAY))
    for index, row in enumerate(data):
        y = top + (index + 0.5) * row_step
        low = _scale(float(row["mos_ci_low"]), left, chart_width)
        high = _scale(float(row["mos_ci_high"]), left, chart_width)
        point = _scale(float(row["mean_mos"]), left, chart_width)
        body.extend(
            [
                _text(left - 18, y + 5, row["condition_display_name"], anchor="end", size=13),
                f'<line x1="{low:.1f}" y1="{y:.1f}" x2="{high:.1f}" y2="{y:.1f}" stroke="{BLUE}" stroke-width="3"/>',
                f'<line x1="{low:.1f}" y1="{y - 7:.1f}" x2="{low:.1f}" y2="{y + 7:.1f}" stroke="{BLUE}" stroke-width="2"/>',
                f'<line x1="{high:.1f}" y1="{y - 7:.1f}" x2="{high:.1f}" y2="{y + 7:.1f}" stroke="{BLUE}" stroke-width="2"/>',
                f'<circle cx="{point:.1f}" cy="{y:.1f}" r="6" fill="{BLUE}"/>',
                _text(high + 10, y + 5, f'{float(row["mean_mos"]):.2f}', size=12, weight=500),
            ]
        )
    body.append(_text(left + chart_width / 2, height - 10, "Mean opinion score (1-5)", anchor="middle", size=12, color=GRAY))
    path.write_text(
        _svg_document(width, height, "Mean opinion score by condition", "Mean MOS and listener-level Student-t 95 percent confidence interval for each audio condition.", body),
        encoding="utf-8",
    )


def _draw_rating_distribution(
    data: list[dict[str, Any]], path: Path, conditions: list[str]
) -> None:
    width, height = 1000, 455
    left, right, top = 260.0, 55.0, 150.0
    chart_width = width - left - right
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data:
        grouped[str(row["condition"])].append(row)
    row_step = 62.0
    body = [
        _text(18, 34, "MOS rating distribution by condition", size=21, weight=500),
        _text(18, 60, "Each bar shows the percentage of ratings from 1 (Bad) to 5 (Excellent).", size=12, color=GRAY),
    ]
    for rating in range(1, 6):
        x = left + (rating - 1) * 112
        body.append(f'<rect x="{x:.1f}" y="84" width="16" height="16" fill="{RATING_COLORS[rating]}"/>')
        body.append(_text(x + 22, 97, str(rating), size=12))
    for index, condition in enumerate(conditions):
        rows = sorted(grouped.get(condition, []), key=lambda row: int(row["mos_rating"]))
        if not rows:
            continue
        y = top + index * row_step
        body.append(_text(left - 18, y + 20, rows[0]["condition_display_name"], anchor="end", size=13))
        cursor = left
        for row in rows:
            rating = int(row["mos_rating"])
            proportion = float(row["rating_proportion"])
            segment_width = chart_width * proportion
            body.append(f'<rect x="{cursor:.1f}" y="{y:.1f}" width="{segment_width:.1f}" height="30" fill="{RATING_COLORS[rating]}"/>')
            percentage_label = f"{proportion:.0%}"
            if _label_fits_segment(segment_width, percentage_label):
                label_color = TEXT if rating == 3 else BACKGROUND
                body.append(_text(cursor + segment_width / 2, y + 20, percentage_label, anchor="middle", size=11, weight=500, color=label_color))
            cursor += segment_width
    for percentage in range(0, 101, 20):
        x = left + chart_width * percentage / 100
        body.append(_text(x, height - 36, f"{percentage}%", anchor="middle", size=12, color=GRAY))
    body.append(_text(left + chart_width / 2, height - 8, "Share of ratings", anchor="middle", size=12, color=GRAY))
    path.write_text(
        _svg_document(width, height, "MOS rating distribution by condition", "One hundred percent stacked bars show rating proportions from one through five for each condition.", body),
        encoding="utf-8",
    )


def _jitter(participant: str, session: str, condition: str) -> float:
    digest = hashlib.sha256(f"{participant}|{session}|{condition}".encode()).digest()
    return (int.from_bytes(digest[:2], "big") / 65535.0 - 0.5) * 50.0


def _draw_listener_distribution(
    listener_data: list[dict[str, Any]],
    condition_data: list[dict[str, Any]],
    path: Path,
    conditions: list[str],
) -> None:
    width, height = 1000, 600
    left, right, top, bottom = 90.0, 45.0, 95.0, 105.0
    chart_width = width - left - right
    chart_height = height - top - bottom
    condition_x = {
        condition: left + (index + 0.5) * chart_width / len(conditions)
        for index, condition in enumerate(conditions)
    }
    condition_lookup = {str(row["condition"]): row for row in condition_data}
    body = [
        _text(18, 34, "Listener-level MOS by condition", size=21, weight=500),
        _text(18, 60, "Each small point is one listener's condition mean; diamonds and whiskers show the overall mean and 95% CI.", size=12, color=GRAY),
    ]
    for rating in range(1, 6):
        y = top + (5 - rating) / 4 * chart_height
        body.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + chart_width:.1f}" y2="{y:.1f}" stroke="{LIGHT_GRAY}" stroke-width="1"/>')
        body.append(_text(left - 14, y + 4, rating, anchor="end", size=12, color=GRAY))
    for row in listener_data:
        condition = str(row["condition"])
        x = condition_x[condition] + _jitter(str(row["participant_id"]), str(row["session_id"]), condition)
        y = top + (5 - float(row["listener_mean_mos"])) / 4 * chart_height
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{GRAY}" fill-opacity="0.42"/>')
    for condition in conditions:
        row = condition_lookup[condition]
        x = condition_x[condition]
        low_y = top + (5 - float(row["mos_ci_low"])) / 4 * chart_height
        high_y = top + (5 - float(row["mos_ci_high"])) / 4 * chart_height
        point_y = top + (5 - float(row["mean_mos"])) / 4 * chart_height
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{high_y:.1f}" x2="{x:.1f}" y2="{low_y:.1f}" stroke="{BLUE}" stroke-width="3"/>',
                f'<line x1="{x - 8:.1f}" y1="{high_y:.1f}" x2="{x + 8:.1f}" y2="{high_y:.1f}" stroke="{BLUE}" stroke-width="2"/>',
                f'<line x1="{x - 8:.1f}" y1="{low_y:.1f}" x2="{x + 8:.1f}" y2="{low_y:.1f}" stroke="{BLUE}" stroke-width="2"/>',
                f'<path d="M {x:.1f} {point_y - 8:.1f} L {x + 8:.1f} {point_y:.1f} L {x:.1f} {point_y + 8:.1f} L {x - 8:.1f} {point_y:.1f} Z" fill="{BLUE}"/>',
                _text(x, height - 66, row["condition_display_name"], anchor="middle", size=12),
            ]
        )
    axis_y = top + chart_height / 2
    body.append(
        f'<text x="24" y="{axis_y:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 24 {axis_y:.1f})" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="12" '
        f'fill="{GRAY}">Listener mean MOS</text>'
    )
    path.write_text(
        _svg_document(width, height, "Listener-level MOS by condition", "Listener condition means are shown as jittered points with condition means and Student-t confidence intervals.", body),
        encoding="utf-8",
    )


def _draw_speaker_condition(
    data: list[dict[str, Any]],
    path: Path,
    speakers: list[str],
    conditions: list[str],
) -> None:
    width = 1200
    columns = 2
    rows_count = (len(speakers) + columns - 1) // columns
    panel_width = 580.0
    panel_height = 270.0
    start_y = 100.0
    grouped = {(str(row["speaker_id"]), str(row["condition"])): row for row in data}
    body = [
        _text(18, 34, "Mean opinion score by speaker and condition", size=21, weight=500),
        _text(18, 60, "Points are listener-level means; whiskers are Student-t 95% confidence intervals. All panels use the same 1-5 scale.", size=12, color=GRAY),
    ]
    for speaker_index, speaker in enumerate(speakers):
        column = speaker_index % columns
        row_index = speaker_index // columns
        panel_x = 18.0 + column * panel_width
        panel_y = start_y + row_index * panel_height
        label_width = 185.0
        chart_x = panel_x + label_width
        chart_width = panel_width - label_width - 30.0
        chart_top = panel_y + 42.0
        row_step = 40.0
        first = next((grouped[(speaker, c)] for c in conditions if (speaker, c) in grouped), None)
        speaker_name = first["speaker_display_name"] if first else speaker
        body.append(_text(panel_x, panel_y + 18, speaker_name, size=14, weight=500))
        for rating in range(1, 6):
            x = _scale(float(rating), chart_x, chart_width)
            body.append(f'<line x1="{x:.1f}" y1="{chart_top - 12:.1f}" x2="{x:.1f}" y2="{chart_top + row_step * len(conditions) - 12:.1f}" stroke="{LIGHT_GRAY}" stroke-width="1"/>')
            body.append(_text(x, chart_top + row_step * len(conditions) + 5, rating, anchor="middle", size=11, color=GRAY))
        for condition_index, condition in enumerate(conditions):
            item = grouped.get((speaker, condition))
            if item is None:
                continue
            y = chart_top + condition_index * row_step
            low = _scale(float(item["mos_ci_low"]), chart_x, chart_width)
            high = _scale(float(item["mos_ci_high"]), chart_x, chart_width)
            point = _scale(float(item["mean_mos"]), chart_x, chart_width)
            body.extend(
                [
                    _text(chart_x - 12, y + 4, item["condition_display_name"], anchor="end", size=11),
                    f'<line x1="{low:.1f}" y1="{y:.1f}" x2="{high:.1f}" y2="{y:.1f}" stroke="{BLUE}" stroke-width="2.5"/>',
                    f'<line x1="{low:.1f}" y1="{y - 5:.1f}" x2="{low:.1f}" y2="{y + 5:.1f}" stroke="{BLUE}" stroke-width="1.5"/>',
                    f'<line x1="{high:.1f}" y1="{y - 5:.1f}" x2="{high:.1f}" y2="{y + 5:.1f}" stroke="{BLUE}" stroke-width="1.5"/>',
                    f'<circle cx="{point:.1f}" cy="{y:.1f}" r="5" fill="{BLUE}"/>',
                ]
            )
        body.append(_text(chart_x + chart_width / 2, chart_top + row_step * len(conditions) + 30, "Mean opinion score", anchor="middle", size=11, color=GRAY))
    figure_height = int(start_y + rows_count * panel_height + 30)
    path.write_text(
        _svg_document(width, figure_height, "Mean opinion score by speaker and condition", "Small multiples compare condition MOS with listener-level Student-t intervals for every speaker.", body),
        encoding="utf-8",
    )


def generate_mos_figures(
    scored: list[dict[str, Any]],
    output_root: Path,
    expected_conditions: list[str],
    rating_scale: list[int],
    confidence_level: float,
    settings: dict[str, Any],
) -> list[str]:
    """Generate report-ready MOS SVGs and the exact data used by each plot."""
    if not settings.get("enabled", True):
        return []
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = _ordered_conditions(scored, expected_conditions, settings)
    condition_names = {
        str(key): str(value)
        for key, value in settings.get("condition_display_names", {}).items()
    }
    speakers_in_data = {str(row.get("speaker_id", "")) for row in scored} - {""}
    configured_speakers = [str(value) for value in settings.get("speaker_order", [])]
    speakers = [value for value in configured_speakers if value in speakers_in_data]
    speakers.extend(sorted(speakers_in_data - set(speakers)))
    speaker_names = {
        str(key): str(value)
        for key, value in settings.get("speaker_display_names", {}).items()
    }

    condition_data = _condition_data(scored, conditions, condition_names, confidence_level)
    distribution_data = _distribution_data(scored, conditions, condition_names, rating_scale)
    listener_data = _listener_condition_data(scored, conditions, condition_names)
    speaker_data = _speaker_condition_data(scored, conditions, condition_names, speakers, speaker_names, confidence_level)

    _write_csv(output_root / "mos_condition_plot_data.csv", condition_data)
    _write_csv(output_root / "mos_rating_distribution_plot_data.csv", distribution_data)
    _write_csv(output_root / "mos_listener_condition_plot_data.csv", listener_data)
    _write_csv(output_root / "mos_speaker_condition_plot_data.csv", speaker_data)
    _draw_mos_by_condition(condition_data, output_root / "mos_by_condition.svg")
    _draw_rating_distribution(distribution_data, output_root / "mos_rating_distribution.svg", conditions)
    _draw_listener_distribution(listener_data, condition_data, output_root / "mos_by_listener_condition.svg", conditions)
    _draw_speaker_condition(speaker_data, output_root / "mos_by_speaker_condition.svg", speakers, conditions)
    return [
        "mos_by_condition.svg",
        "mos_rating_distribution.svg",
        "mos_by_listener_condition.svg",
        "mos_by_speaker_condition.svg",
        "mos_condition_plot_data.csv",
        "mos_rating_distribution_plot_data.csv",
        "mos_listener_condition_plot_data.csv",
        "mos_speaker_condition_plot_data.csv",
    ]
