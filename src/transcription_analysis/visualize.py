from __future__ import annotations

import html
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .io_utils import write_csv


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B61A8"
SKY_BLUE = "#56B4E9"
PINK = "#CC79A7"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
TEXT = "#1F2937"
BACKGROUND = "#FFFFFF"


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile from an empty list.")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _listener_bootstrap_ci(
    rows: list[dict[str, Any]],
    metric: str,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Return the mean and a listener-clustered percentile bootstrap interval."""
    observed = [
        value for value in (_number(row.get(metric)) for row in rows)
        if value is not None
    ]
    if not observed:
        return (math.nan, math.nan, math.nan)
    clusters: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row.get(metric))
        if value is None:
            continue
        key = (
            str(row.get("participant_id", "")),
            str(row.get("session_id", "")),
        )
        clusters[key].append(value)
    cluster_values = list(clusters.values())
    if len(cluster_values) < 2 or samples < 2:
        point = mean(observed)
        return (point, point, point)
    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        resampled: list[float] = []
        for _ in cluster_values:
            resampled.extend(generator.choice(cluster_values))
        estimates.append(mean(resampled))
    return (mean(observed), _quantile(estimates, 0.025), _quantile(estimates, 0.975))


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


def _svg_document(width: int, height: int, title: str, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{_escape(title)}</title>',
            '<desc id="desc">Lower values indicate fewer transcription errors or fewer explicit unintelligible spans.</desc>',
            f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def _nice_max(values: list[float], minimum: float = 0.10) -> float:
    maximum = max(values, default=minimum)
    padded = max(minimum, maximum * 1.12)
    step = 0.05 if padded <= 0.50 else 0.10
    return math.ceil(padded / step) * step


def _group_condition_error_data(
    scored: list[dict[str, Any]],
    group_key: str,
    group_order: list[str],
    group_display_names: dict[str, str],
    conditions: list[str],
    condition_display_names: dict[str, str],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Summarize WER/CER for every requested group-by-condition cell."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        group_value = str(row.get(group_key, "")).strip()
        condition = str(row.get("condition", "")).strip()
        if group_value and condition:
            grouped[(group_value, condition)].append(row)
    samples = int(settings.get("bootstrap_samples", 5000))
    seed = int(settings.get("random_seed", 20261016))
    output: list[dict[str, Any]] = []
    cell_index = 0
    for group_value in group_order:
        for condition in conditions:
            group = grouped.get((group_value, condition), [])
            if not group:
                continue
            wer, wer_low, wer_high = _listener_bootstrap_ci(
                group, "wer", samples, seed + cell_index * 2
            )
            cer, cer_low, cer_high = _listener_bootstrap_ci(
                group, "cer", samples, seed + cell_index * 2 + 1
            )
            output.append(
                {
                    group_key: group_value,
                    "group_display_name": group_display_names.get(
                        group_value, group_value
                    ),
                    "condition": condition,
                    "condition_display_name": condition_display_names.get(
                        condition, condition
                    ),
                    "n_transcriptions": len(group),
                    "mean_wer": wer,
                    "wer_ci_low": wer_low,
                    "wer_ci_high": wer_high,
                    "mean_cer": cer,
                    "cer_ci_low": cer_low,
                    "cer_ci_high": cer_high,
                }
            )
            cell_index += 1
    return output


def _condition_plot_data(
    scored: list[dict[str, Any]],
    conditions: list[str],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_condition[str(row.get("condition", ""))].append(row)
    samples = int(settings.get("bootstrap_samples", 5000))
    seed = int(settings.get("random_seed", 20261016))
    output: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        group = by_condition.get(condition, [])
        if not group:
            continue
        wer, wer_low, wer_high = _listener_bootstrap_ci(
            group, "wer", samples, seed + index * 2
        )
        cer, cer_low, cer_high = _listener_bootstrap_ci(
            group, "cer", samples, seed + index * 2 + 1
        )
        def mean_component_rate(count_field: str, length_field: str) -> float:
            rates = [
                int(row.get(count_field, 0) or 0) / int(row.get(length_field, 0) or 0)
                for row in group
                if int(row.get(length_field, 0) or 0) > 0
            ]
            return mean(rates) if rates else 0.0
        output.append(
            {
                "condition": condition,
                "display_name": settings.get("condition_display_names", {}).get(
                    condition, condition
                ),
                "n_transcriptions": len(group),
                "mean_wer": wer,
                "wer_ci_low": wer_low,
                "wer_ci_high": wer_high,
                "mean_cer": cer,
                "cer_ci_low": cer_low,
                "cer_ci_high": cer_high,
                "substitution_rate": mean_component_rate("word_substitutions", "reference_words"),
                "deletion_rate": mean_component_rate("word_deletions", "reference_words"),
                "insertion_rate": mean_component_rate("word_insertions", "reference_words"),
                "character_substitution_rate": mean_component_rate("character_substitutions", "reference_characters"),
                "character_deletion_rate": mean_component_rate("character_deletions", "reference_characters"),
                "character_insertion_rate": mean_component_rate("character_insertions", "reference_characters"),
                "xxx_response_rate": sum(
                    int(row.get("contains_xxx", 0)) for row in group
                )
                / len(group),
                "whole_response_xxx_rate": sum(
                    int(row.get("whole_utterance_xxx", 0)) for row in group
                )
                / len(group),
                "total_xxx_tokens": sum(
                    int(row.get("xxx_token_count", 0)) for row in group
                ),
            }
        )
    return output


def _draw_error_rate_figure(
    data: list[dict[str, Any]],
    path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    width, height = 1180, 520
    top, bottom = 88, 65
    panel_width = 475
    label_width = 205
    panels = [
        (35, "Word error rate", "mean_wer", "wer_ci_low", "wer_ci_high", BLUE),
        (620, "Character error rate", "mean_cer", "cer_ci_low", "cer_ci_high", ORANGE),
    ]
    body = [
        _text(35, 34, title, size=20, weight=500),
        _text(
            35,
            58,
            subtitle,
            size=12,
            color=GRAY,
        ),
    ]
    row_gap = (height - top - bottom) / max(1, len(data))
    for panel_x, title, value_key, low_key, high_key, color in panels:
        chart_x = panel_x + label_width
        chart_width = panel_width - label_width
        maximum = _nice_max([float(row[high_key]) for row in data])
        body.append(_text(panel_x, 82, title, size=15, weight=500))
        for tick_index in range(5):
            tick = maximum * tick_index / 4
            x = chart_x + chart_width * tick / maximum
            body.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="{LIGHT_GRAY}" stroke-width="1"/>'
            )
            body.append(
                _text(x, height - 38, f"{tick:.0%}", anchor="middle", size=11, color=GRAY)
            )
        for row_index, row in enumerate(data):
            y = top + row_gap * (row_index + 0.5)
            value = float(row[value_key])
            low = float(row[low_key])
            high = float(row[high_key])
            x = chart_x + chart_width * value / maximum
            x_low = chart_x + chart_width * low / maximum
            x_high = chart_x + chart_width * high / maximum
            body.append(_text(panel_x, y + 4, row["display_name"], size=12))
            body.append(
                f'<line x1="{x_low:.1f}" y1="{y:.1f}" x2="{x_high:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="3"/>'
            )
            body.append(
                f'<line x1="{x_low:.1f}" y1="{y-6:.1f}" x2="{x_low:.1f}" y2="{y+6:.1f}" stroke="{color}" stroke-width="2"/>'
            )
            body.append(
                f'<line x1="{x_high:.1f}" y1="{y-6:.1f}" x2="{x_high:.1f}" y2="{y+6:.1f}" stroke="{color}" stroke-width="2"/>'
            )
            body.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
            )
            label_x = min(chart_x + chart_width - 2, x_high + 8)
            anchor = "end" if label_x >= chart_x + chart_width - 4 else "start"
            body.append(
                _text(label_x, y + 4, f"{value:.1%}", anchor=anchor, size=11, weight=500)
            )
        body.append(
            _text(
                chart_x + chart_width / 2,
                height - 13,
                "Error rate",
                anchor="middle",
                size=12,
                color=GRAY,
            )
        )
    path.write_text(
        _svg_document(width, height, title, body),
        encoding="utf-8",
    )


def _draw_group_condition_error_figure(
    data: list[dict[str, Any]],
    path: Path,
    *,
    group_key: str,
    group_order: list[str],
    conditions: list[str],
    title: str,
) -> None:
    """Draw WER/CER for each group, with one estimate per condition."""
    present_groups = [
        value for value in group_order
        if any(str(row.get(group_key, "")) == value for row in data)
    ]
    condition_colors = [BLUE, ORANGE, GREEN, PURPLE, PINK, SKY_BLUE]
    colors = {
        condition: condition_colors[index % len(condition_colors)]
        for index, condition in enumerate(conditions)
    }
    width = 1380
    top = 150
    bottom = 66
    group_gap = 118
    height = top + bottom + group_gap * max(1, len(present_groups))
    panel_width = 620
    label_width = 170
    chart_width = panel_width - label_width
    panels = [
        (35, "Word error rate", "mean_wer", "wer_ci_low", "wer_ci_high"),
        (710, "Character error rate", "mean_cer", "cer_ci_low", "cer_ci_high"),
    ]
    body = [
        _text(35, 34, title, size=20, weight=500),
        _text(
            35,
            58,
            "Each colored point is one condition; whiskers are listener-clustered 95% bootstrap intervals.",
            size=12,
            color=GRAY,
        ),
    ]
    legend_x = 35
    legend_y = 88
    condition_names = {
        str(row["condition"]): str(row["condition_display_name"]) for row in data
    }
    for condition in conditions:
        if condition not in condition_names:
            continue
        body.append(
            f'<circle cx="{legend_x + 5}" cy="{legend_y - 4}" r="5" fill="{colors[condition]}"/>'
        )
        body.append(_text(legend_x + 16, legend_y, condition_names[condition], size=11))
        legend_x += max(115, 16 + len(condition_names[condition]) * 7)

    condition_count = max(1, len(conditions))
    point_gap = min(17, 70 / condition_count)
    offsets = {
        condition: (index - (condition_count - 1) / 2) * point_gap
        for index, condition in enumerate(conditions)
    }
    for panel_x, panel_title, value_key, low_key, high_key in panels:
        chart_x = panel_x + label_width
        maximum = _nice_max([float(row[high_key]) for row in data])
        body.append(_text(panel_x, 126, panel_title, size=15, weight=500))
        for tick_index in range(5):
            tick = maximum * tick_index / 4
            x = chart_x + chart_width * tick / maximum
            body.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="{LIGHT_GRAY}" stroke-width="1"/>'
            )
            body.append(
                _text(x, height - 39, f"{tick:.0%}", anchor="middle", size=11, color=GRAY)
            )
        for group_index, group_value in enumerate(present_groups):
            center_y = top + group_gap * (group_index + 0.5)
            group_rows = [
                row for row in data if str(row.get(group_key, "")) == group_value
            ]
            if group_rows:
                body.append(
                    _text(panel_x, center_y + 4, group_rows[0]["group_display_name"], size=12)
                )
            if group_index:
                separator_y = top + group_gap * group_index
                body.append(
                    f'<line x1="{panel_x}" y1="{separator_y:.1f}" x2="{chart_x + chart_width}" y2="{separator_y:.1f}" stroke="{LIGHT_GRAY}" stroke-width="1"/>'
                )
            for row in group_rows:
                condition = str(row["condition"])
                y = center_y + offsets.get(condition, 0)
                value = float(row[value_key])
                low = float(row[low_key])
                high = float(row[high_key])
                x = chart_x + chart_width * value / maximum
                x_low = chart_x + chart_width * low / maximum
                x_high = chart_x + chart_width * high / maximum
                color = colors[condition]
                body.append(
                    f'<line x1="{x_low:.1f}" y1="{y:.1f}" x2="{x_high:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2"/>'
                )
                body.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
                )
        body.append(
            _text(
                chart_x + chart_width / 2,
                height - 13,
                "Error rate",
                anchor="middle",
                size=12,
                color=GRAY,
            )
        )
    path.write_text(_svg_document(width, height, title, body), encoding="utf-8")


def _draw_error_composition_figure(data: list[dict[str, Any]], path: Path) -> None:
    width, height = 1380, 520
    top, bottom = 112, 60
    row_gap = (height - top - bottom) / max(1, len(data))
    body = [
        _text(35, 34, "Error composition by condition", size=20, weight=500),
        _text(
            35,
            58,
            "Each panel shows mean response-level substitutions, deletions, and insertions. Lower is better.",
            size=12,
            color=GRAY,
        ),
    ]
    legend = [(BLUE, "Substitutions"), (ORANGE, "Deletions"), (GREEN, "Insertions")]
    legend_x = 890
    for color, label in legend:
        body.append(
            f'<rect x="{legend_x}" y="47" width="12" height="12" fill="{color}"/>'
        )
        body.append(_text(legend_x + 18, 58, label, size=11))
        legend_x += 115
    panels = [
        (35, "Word error rate", "Error rate", ""),
        (710, "Character error rate", "Error rate", "character_"),
    ]
    panel_width, label_width, chart_width = 635, 215, 420
    for panel_x, panel_title, axis_title, prefix in panels:
        body.append(_text(panel_x, 92, panel_title, size=15, weight=500))
        maximum = _nice_max(
            [
                float(row[f"{prefix}substitution_rate"])
                + float(row[f"{prefix}deletion_rate"])
                + float(row[f"{prefix}insertion_rate"])
                for row in data
            ]
        )
        chart_x = panel_x + label_width
        for tick_index in range(5):
            tick = maximum * tick_index / 4
            x = chart_x + chart_width * tick / maximum
            body.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="{LIGHT_GRAY}"/>'
            )
            body.append(_text(x, height - 34, f"{tick:.0%}", anchor="middle", size=11, color=GRAY))
        for index, row in enumerate(data):
            y = top + row_gap * (index + 0.5)
            body.append(_text(chart_x - 12, y + 4, row["display_name"], anchor="end", size=12))
            current_x = chart_x
            total = 0.0
            for key, color in (
                (f"{prefix}substitution_rate", BLUE),
                (f"{prefix}deletion_rate", ORANGE),
                (f"{prefix}insertion_rate", GREEN),
            ):
                value = float(row[key])
                segment_width = chart_width * value / maximum
                body.append(f'<rect x="{current_x:.1f}" y="{y-10:.1f}" width="{segment_width:.1f}" height="20" fill="{color}"/>')
                current_x += segment_width
                total += value
            body.append(_text(min(panel_x + panel_width - 2, current_x + 8), y + 4, f"{total:.1%}", anchor="end" if current_x + 8 >= panel_x + panel_width else "start", size=11, weight=500))
        body.append(
            _text(chart_x + chart_width / 2, height - 10, axis_title, anchor="middle", size=12, color=GRAY)
        )
    path.write_text(
        _svg_document(width, height, "Word and character error composition by condition", body),
        encoding="utf-8",
    )


def _draw_xxx_figure(data: list[dict[str, Any]], path: Path) -> None:
    width, height = 920, 520
    left, right, top, bottom = 245, 75, 112, 60
    chart_width = width - left - right
    row_gap = (height - top - bottom) / max(1, len(data))
    maximum = _nice_max([float(row["xxx_response_rate"]) for row in data])
    body = [
        _text(35, 34, "Explicit unintelligibility markers by condition", size=20, weight=500),
        _text(
            35,
            58,
            "Any xxx shows partial uncertainty; whole-response xxx is complete uncertainty.",
            size=12,
            color=GRAY,
        ),
        f'<circle cx="505" cy="82" r="5" fill="{PURPLE}"/>',
        _text(517, 86, "Any xxx", size=11),
        f'<rect x="605" y="77" width="10" height="10" fill="{ORANGE}"/>',
        _text(622, 86, "Whole response xxx", size=11),
    ]
    for tick_index in range(5):
        tick = maximum * tick_index / 4
        x = left + chart_width * tick / maximum
        body.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="{LIGHT_GRAY}"/>'
        )
        body.append(
            _text(x, height - 34, f"{tick:.0%}", anchor="middle", size=11, color=GRAY)
        )
    for index, row in enumerate(data):
        y = top + row_gap * (index + 0.5)
        any_rate = float(row["xxx_response_rate"])
        whole_rate = float(row["whole_response_xxx_rate"])
        x_any = left + chart_width * any_rate / maximum
        x_whole = left + chart_width * whole_rate / maximum
        body.append(
            _text(left - 12, y + 4, row["display_name"], anchor="end", size=12)
        )
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{x_any:.1f}" y2="{y:.1f}" stroke="{LIGHT_GRAY}" stroke-width="8"/>'
        )
        body.append(
            f'<circle cx="{x_any:.1f}" cy="{y:.1f}" r="6" fill="{PURPLE}"/>'
        )
        body.append(
            f'<rect x="{x_whole-5:.1f}" y="{y-5:.1f}" width="10" height="10" fill="{ORANGE}"/>'
        )
        body.append(
            _text(min(width - right + 50, x_any + 9), y + 4, f"{any_rate:.1%}", size=11, weight=500)
        )
    body.append(
        _text(
            left + chart_width / 2,
            height - 10,
            "Share of transcription responses",
            anchor="middle",
            size=12,
            color=GRAY,
        )
    )
    path.write_text(
        _svg_document(width, height, "Explicit unintelligibility markers by condition", body),
        encoding="utf-8",
    )


def _consistency_plot_data(
    rows: list[dict[str, Any]],
    conditions: list[str],
    condition_display_names: dict[str, str],
) -> list[dict[str, Any]]:
    """Return complete utterance-condition consistency rows in plot order."""
    order = {condition: index for index, condition in enumerate(conditions)}
    output: list[dict[str, Any]] = []
    for row in rows:
        condition = str(row.get("condition", ""))
        complete = str(row.get("complete_listener_set", "")).lower() == "true"
        if condition not in order or not complete:
            continue
        wer_sd = _number(row.get("listener_sd_wer"))
        cer_sd = _number(row.get("listener_sd_cer"))
        if wer_sd is None or cer_sd is None:
            continue
        output.append(
            {
                "utterance_id": row.get("utterance_id", ""),
                "condition": condition,
                "condition_display_name": condition_display_names.get(
                    condition, condition
                ),
                "n_responses": row.get("n_responses", ""),
                "listener_sd_wer": wer_sd,
                "listener_sd_cer": cer_sd,
            }
        )
    return sorted(
        output,
        key=lambda row: (order[str(row["condition"])], str(row["utterance_id"])),
    )


def _draw_consistency_figure(
    data: list[dict[str, Any]],
    path: Path,
    conditions: list[str],
) -> None:
    """Draw box-and-dot distributions of within-item listener WER/CER SD."""
    width, height = 1260, 555
    top, bottom = 112, 66
    panel_width = 570
    label_width = 205
    chart_width = panel_width - label_width
    panels = [
        (35, "Word error rate variation", "listener_sd_wer", BLUE),
        (655, "Character error rate variation", "listener_sd_cer", ORANGE),
    ]
    present_conditions = [
        condition
        for condition in conditions
        if any(str(row["condition"]) == condition for row in data)
    ]
    row_gap = (height - top - bottom) / max(1, len(present_conditions))
    body = [
        _text(35, 34, "Inter-listener consistency by condition", size=20, weight=500),
        _text(
            35,
            58,
            "Each dot is one complete utterance-condition group; boxes show the middle 50% and median. Lower variation means greater consistency.",
            size=12,
            color=GRAY,
        ),
    ]
    for panel_x, panel_title, metric, color in panels:
        chart_x = panel_x + label_width
        all_values = [float(row[metric]) for row in data]
        maximum = _nice_max(all_values, minimum=0.20)
        body.append(_text(panel_x, 90, panel_title, size=15, weight=500))
        for tick_index in range(5):
            tick = maximum * tick_index / 4
            x = chart_x + chart_width * tick / maximum
            body.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="{LIGHT_GRAY}" stroke-width="1"/>'
            )
            body.append(
                _text(x, height - 38, f"{tick:.0%}", anchor="middle", size=11, color=GRAY)
            )
        for row_index, condition in enumerate(present_conditions):
            group = [row for row in data if str(row["condition"]) == condition]
            values = sorted(float(row[metric]) for row in group)
            if not values:
                continue
            center_y = top + row_gap * (row_index + 0.5)
            display_name = str(group[0]["condition_display_name"])
            body.append(_text(panel_x, center_y + 4, display_name, size=12))
            q1 = _quantile(values, 0.25)
            median_value = _quantile(values, 0.50)
            q3 = _quantile(values, 0.75)
            iqr = q3 - q1
            lower_candidates = [value for value in values if value >= q1 - 1.5 * iqr]
            upper_candidates = [value for value in values if value <= q3 + 1.5 * iqr]
            whisker_low = min(lower_candidates) if lower_candidates else min(values)
            whisker_high = max(upper_candidates) if upper_candidates else max(values)
            scale = lambda value: chart_x + chart_width * value / maximum
            jitter = (-10, -5, 0, 5, 10)
            for value_index, value in enumerate(values):
                dot_y = center_y + jitter[value_index % len(jitter)]
                body.append(
                    f'<circle cx="{scale(value):.1f}" cy="{dot_y:.1f}" r="3.2" fill="{GRAY}" fill-opacity="0.48"/>'
                )
            body.append(
                f'<line x1="{scale(whisker_low):.1f}" y1="{center_y:.1f}" x2="{scale(whisker_high):.1f}" y2="{center_y:.1f}" stroke="{color}" stroke-width="2"/>'
            )
            body.append(
                f'<rect x="{scale(q1):.1f}" y="{center_y-12:.1f}" width="{max(1.0, scale(q3)-scale(q1)):.1f}" height="24" fill="{color}" fill-opacity="0.24" stroke="{color}" stroke-width="2"/>'
            )
            body.append(
                f'<line x1="{scale(median_value):.1f}" y1="{center_y-14:.1f}" x2="{scale(median_value):.1f}" y2="{center_y+14:.1f}" stroke="{color}" stroke-width="3"/>'
            )
            body.append(
                _text(
                    chart_x + chart_width + 8,
                    center_y + 4,
                    f"{median_value:.1%}",
                    size=11,
                    weight=500,
                    color=color,
                )
            )
        body.append(
            _text(
                chart_x + chart_width / 2,
                height - 12,
                "Within-item listener SD",
                anchor="middle",
                size=12,
                color=GRAY,
            )
        )
    path.write_text(
        _svg_document(width, height, "Inter-listener consistency by condition", body),
        encoding="utf-8",
    )


def generate_figures(
    scored: list[dict[str, Any]],
    output_root: Path,
    expected_conditions: list[str],
    settings: dict[str, Any],
    consistency_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Generate report-ready SVG figures and their auditable condition data."""
    if not settings.get("enabled", True):
        return []
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = [
        condition
        for condition in settings.get("condition_order", expected_conditions)
        if any(str(row.get("condition", "")) == condition for row in scored)
    ]
    condition_data = _condition_plot_data(scored, conditions, settings)
    genders = [
        str(value) for value in settings.get("gender_order", ["F", "M"])
        if any(str(row.get("gender", "")).strip() == str(value) for row in scored)
    ]
    speakers_in_data = {str(row.get("speaker_id", "")).strip() for row in scored}
    configured_speakers = [str(value) for value in settings.get("speaker_order", [])]
    speakers = [value for value in configured_speakers if value in speakers_in_data]
    speakers.extend(sorted(speakers_in_data - set(speakers) - {""}))
    gender_data = _group_condition_error_data(
        scored,
        "gender",
        genders,
        settings.get("gender_display_names", {}),
        conditions,
        settings.get("condition_display_names", {}),
        settings,
    )
    speaker_data = _group_condition_error_data(
        scored,
        "speaker_id",
        speakers,
        settings.get("speaker_display_names", {}),
        conditions,
        settings.get("condition_display_names", {}),
        settings,
    )
    write_csv(output_root / "condition_plot_data.csv", condition_data)
    write_csv(output_root / "gender_condition_plot_data.csv", gender_data)
    write_csv(output_root / "speaker_condition_plot_data.csv", speaker_data)
    consistency_data = _consistency_plot_data(
        consistency_rows or [],
        conditions,
        settings.get("condition_display_names", {}),
    )
    write_csv(
        output_root / "inter_listener_consistency_plot_data.csv",
        consistency_data,
    )
    _draw_error_rate_figure(
        condition_data,
        output_root / "condition_error_rates.svg",
        title="Transcription error rates by condition",
        subtitle="Points are means; whiskers are listener-clustered 95% bootstrap intervals. Lower is better.",
    )
    _draw_group_condition_error_figure(
        gender_data,
        output_root / "gender_error_rates.svg",
        group_key="gender",
        group_order=genders,
        conditions=conditions,
        title="Error rates by speaker gender and condition",
    )
    _draw_group_condition_error_figure(
        speaker_data,
        output_root / "speaker_error_rates.svg",
        group_key="speaker_id",
        group_order=speakers,
        conditions=conditions,
        title="Error rates by speaker and condition",
    )
    _draw_error_composition_figure(condition_data, output_root / "error_composition.svg")
    _draw_xxx_figure(condition_data, output_root / "xxx_rates.svg")
    figures = [
        "condition_error_rates.svg",
        "gender_error_rates.svg",
        "speaker_error_rates.svg",
        "error_composition.svg",
        "xxx_rates.svg",
    ]
    if consistency_data:
        _draw_consistency_figure(
            consistency_data,
            output_root / "inter_listener_consistency.svg",
            conditions,
        )
        figures.append("inter_listener_consistency.svg")
    return figures
