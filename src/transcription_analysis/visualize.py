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


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite(row: dict[str, Any], *keys: str) -> bool:
    """True when every key holds a finite number (used to decide if a CI can be drawn)."""
    for key in keys:
        value = _number(row.get(key))
        if value is None or math.isnan(value):
            return False
    return True


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _ci_text(low: float, high: float, digits: int = 1) -> str:
    return f"95% CI {low * 100:.{digits}f}–{high * 100:.{digits}f}%"


def _bracket_ci(low: float, high: float, digits: int = 1) -> str:
    return f"[{low * 100:.{digits}f}–{high * 100:.{digits}f}]"


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


def _box_summary(values: list[float]) -> dict[str, float]:
    """Quartiles, Tukey (1.5 x IQR) whiskers and mean for a box plot."""
    q1 = _quantile(values, 0.25)
    median_value = _quantile(values, 0.50)
    q3 = _quantile(values, 0.75)
    iqr = q3 - q1
    lower_candidates = [value for value in values if value >= q1 - 1.5 * iqr]
    upper_candidates = [value for value in values if value <= q3 + 1.5 * iqr]
    return {
        "q1": q1,
        "median": median_value,
        "q3": q3,
        "whisker_low": min(lower_candidates) if lower_candidates else min(values),
        "whisker_high": max(upper_candidates) if upper_candidates else max(values),
        "mean": mean(values),
    }


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


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------


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


def _line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"/>'
    )


def _ci_whisker(
    x_low: float, x_high: float, y: float, color: str, *, cap: float = 5, width: float = 1.8
) -> list[str]:
    """Horizontal confidence interval with end caps."""
    return [
        _line(x_low, y, x_high, y, color, width),
        _line(x_low, y - cap, x_low, y + cap, color, width),
        _line(x_high, y - cap, x_high, y + cap, color, width),
    ]


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


_TICK_STEPS = (0.01, 0.02, 0.025, 0.05, 0.10, 0.20, 0.25)


def _tick_step(maximum: float) -> float:
    """Smallest 'round' step that splits the axis into at most 6 intervals."""
    for step in _TICK_STEPS:
        if maximum / step <= 6 + 1e-9:
            return step
    return _TICK_STEPS[-1]


def _nice_max(values: list[float], minimum: float = 0.10, cap: float | None = None) -> float:
    """Axis maximum rounded up to a round tick step (optionally capped, e.g. at 100%)."""
    maximum = max(values, default=minimum)
    padded = max(minimum, maximum * 1.12)
    if cap is not None:
        padded = min(padded, cap)
    step = _tick_step(padded)
    return math.ceil(padded / step - 1e-9) * step


def _axis(
    body: list[str],
    chart_x: float,
    chart_width: float,
    maximum: float,
    top: float,
    bottom_y: float,
    title: str,
) -> None:
    """Vertical grid lines, percentage tick labels and an axis title."""
    step = _tick_step(maximum)
    decimals = 0 if abs(step * 100 - round(step * 100)) < 1e-9 else 1
    for tick_index in range(int(round(maximum / step)) + 1):
        tick = step * tick_index
        x = chart_x + chart_width * tick / maximum
        body.append(_line(x, top, x, bottom_y, LIGHT_GRAY))
        body.append(_text(x, bottom_y + 18, f"{tick * 100:.{decimals}f}%", anchor="middle", size=11, color=GRAY))
    body.append(
        _text(chart_x + chart_width / 2, bottom_y + 44, title, anchor="middle", size=12, color=GRAY)
    )


def _draw_box(
    body: list[str],
    values: list[float],
    stats: dict[str, float],
    scale: Any,
    center_y: float,
    color: str,
    *,
    dot_radius: float,
    dot_opacity: float,
    jitter: tuple[int, ...],
) -> None:
    for value_index, value in enumerate(values):
        dot_y = center_y + jitter[value_index % len(jitter)]
        body.append(
            f'<circle cx="{scale(value):.1f}" cy="{dot_y:.1f}" r="{dot_radius}" fill="{GRAY}" fill-opacity="{dot_opacity}"/>'
        )
    body.extend(
        _ci_whisker(scale(stats["whisker_low"]), scale(stats["whisker_high"]), center_y, color, cap=7, width=2)
    )
    body.append(
        f'<rect x="{scale(stats["q1"]):.1f}" y="{center_y-12:.1f}" width="{max(1.0, scale(stats["q3"])-scale(stats["q1"])):.1f}" height="24" fill="{color}" fill-opacity="0.24" stroke="{color}" stroke-width="2"/>'
    )
    body.append(_line(scale(stats["median"]), center_y - 14, scale(stats["median"]), center_y + 14, color, 3))


# ---------------------------------------------------------------------------
# Plot data
# ---------------------------------------------------------------------------


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


def _rate_rows(
    group: list[dict[str, Any]], count_field: str, length_field: str
) -> list[dict[str, Any]]:
    """Per-response component rate (e.g. substitutions / reference words) for bootstrapping."""
    rows: list[dict[str, Any]] = []
    for row in group:
        length = int(row.get(length_field, 0) or 0)
        if length <= 0:
            continue
        rows.append(
            {
                "participant_id": row.get("participant_id", ""),
                "session_id": row.get("session_id", ""),
                "rate": int(row.get(count_field, 0) or 0) / length,
            }
        )
    return rows


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
    components = [
        ("substitution_rate", "word_substitutions", "reference_words"),
        ("deletion_rate", "word_deletions", "reference_words"),
        ("insertion_rate", "word_insertions", "reference_words"),
        ("character_substitution_rate", "character_substitutions", "reference_characters"),
        ("character_deletion_rate", "character_deletions", "reference_characters"),
        ("character_insertion_rate", "character_insertions", "reference_characters"),
    ]
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
        record: dict[str, Any] = {
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
        }
        # Error components, each with its own listener-clustered bootstrap CI.
        for component_index, (name, count_field, length_field) in enumerate(components):
            rate_rows = _rate_rows(group, count_field, length_field)
            if rate_rows:
                value, low, high = _listener_bootstrap_ci(
                    rate_rows, "rate", samples, seed + 1000 + index * 20 + component_index
                )
            else:
                value, low, high = 0.0, 0.0, 0.0
            record[name] = value
            record[f"{name}_ci_low"] = low
            record[f"{name}_ci_high"] = high
        # Explicit unintelligibility (xxx) markers: counts, rates and CIs.
        xxx_rows = [
            {
                "participant_id": row.get("participant_id", ""),
                "session_id": row.get("session_id", ""),
                "contains_xxx": int(row.get("contains_xxx", 0) or 0),
                "whole_utterance_xxx": int(row.get("whole_utterance_xxx", 0) or 0),
            }
            for row in group
        ]
        any_rate, any_low, any_high = _listener_bootstrap_ci(
            xxx_rows, "contains_xxx", samples, seed + 2000 + index * 2
        )
        whole_rate, whole_low, whole_high = _listener_bootstrap_ci(
            xxx_rows, "whole_utterance_xxx", samples, seed + 2000 + index * 2 + 1
        )
        record.update(
            {
                "xxx_response_count": sum(row["contains_xxx"] for row in xxx_rows),
                "xxx_response_rate": any_rate,
                "xxx_response_rate_ci_low": any_low,
                "xxx_response_rate_ci_high": any_high,
                "whole_response_xxx_count": sum(row["whole_utterance_xxx"] for row in xxx_rows),
                "whole_response_xxx_rate": whole_rate,
                "whole_response_xxx_rate_ci_low": whole_low,
                "whole_response_xxx_rate_ci_high": whole_high,
                "total_xxx_tokens": sum(
                    int(row.get("xxx_token_count", 0) or 0) for row in group
                ),
            }
        )
        output.append(record)
    return output


def _error_distribution_plot_data(
    scored: list[dict[str, Any]],
    conditions: list[str],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the response-level WER/CER values used in distribution figures."""
    condition_set = set(conditions)
    display_names = settings.get("condition_display_names", {})
    output: list[dict[str, Any]] = []
    for row in scored:
        condition = str(row.get("condition", "")).strip()
        wer = _number(row.get("wer"))
        cer = _number(row.get("cer"))
        if condition not in condition_set or wer is None or cer is None:
            continue
        output.append(
            {
                "condition": condition,
                "condition_display_name": display_names.get(condition, condition),
                "participant_id": row.get("participant_id", ""),
                "session_id": row.get("session_id", ""),
                "utterance_id": row.get("utterance_id", ""),
                "wer": wer,
                "cer": cer,
            }
        )
    condition_order = {condition: index for index, condition in enumerate(conditions)}
    return sorted(
        output,
        key=lambda row: (
            condition_order[str(row["condition"])],
            str(row["participant_id"]),
            str(row["session_id"]),
            str(row["utterance_id"]),
        ),
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


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _draw_group_condition_error_figure(
    data: list[dict[str, Any]],
    path: Path,
    *,
    group_key: str,
    group_order: list[str],
    conditions: list[str],
    title: str,
) -> None:
    """Mean WER/CER per group and condition, with 95% CI whiskers and value labels."""
    present_groups = [
        value for value in group_order
        if any(str(row.get(group_key, "")) == value for row in data)
    ]
    condition_colors = [BLUE, ORANGE, GREEN, PURPLE, PINK, SKY_BLUE]
    colors = {
        condition: condition_colors[index % len(condition_colors)]
        for index, condition in enumerate(conditions)
    }
    condition_count = max(1, len(conditions))
    point_gap = 26
    group_gap = 50 + point_gap * condition_count
    top, bottom = 150, 70
    height = int(top + bottom + group_gap * max(1, len(present_groups)))
    label_width, chart_width, value_space = 140, 400, 175
    panel_width = label_width + chart_width + value_space
    width = 35 + panel_width * 2 + 35
    panels = [
        (35, "Word error rate", "mean_wer", "wer_ci_low", "wer_ci_high"),
        (35 + panel_width + 20, "Character error rate", "mean_cer", "cer_ci_low", "cer_ci_high"),
    ]
    width += 20
    body = [
        _text(35, 34, title, size=20, weight=500),
        _text(
            35,
            58,
            "Points are mean error rates; whiskers and [brackets] are listener-clustered 95% bootstrap CIs; "
            "n = transcriptions per cell. Lower is better.",
            size=12,
            color=GRAY,
        ),
    ]
    condition_names = {
        str(row["condition"]): str(row["condition_display_name"]) for row in data
    }
    legend_x = 35
    for condition in conditions:
        if condition not in condition_names:
            continue
        body.append(f'<circle cx="{legend_x + 5}" cy="84" r="5" fill="{colors[condition]}"/>')
        body.append(_text(legend_x + 16, 88, condition_names[condition], size=12))
        legend_x += 30 + len(condition_names[condition]) * 7
    offsets = {
        condition: (index - (condition_count - 1) / 2) * point_gap
        for index, condition in enumerate(conditions)
    }
    for panel_x, panel_title, value_key, low_key, high_key in panels:
        chart_x = panel_x + label_width
        maximum = _nice_max([float(row[high_key]) for row in data])
        scale = lambda value, cx=chart_x, m=maximum: cx + chart_width * value / m
        body.append(_text(panel_x, 126, panel_title, size=15, weight=500))
        _axis(body, chart_x, chart_width, maximum, top, height - bottom, "Mean error rate")
        for group_index, group_value in enumerate(present_groups):
            center_y = top + group_gap * (group_index + 0.5)
            group_rows = [row for row in data if str(row.get(group_key, "")) == group_value]
            if group_rows:
                body.append(_text(panel_x, center_y + 4, group_rows[0]["group_display_name"], size=12, weight=500))
            if group_index:
                separator_y = top + group_gap * group_index
                body.append(_line(panel_x, separator_y, chart_x + chart_width + value_space - 10, separator_y, LIGHT_GRAY))
            for row in group_rows:
                condition = str(row["condition"])
                y = center_y + offsets.get(condition, 0)
                value = float(row[value_key])
                color = colors[condition]
                label = f"{_pct(value)}"
                label_x = scale(value)
                if _finite(row, low_key, high_key):
                    low, high = float(row[low_key]), float(row[high_key])
                    body.extend(_ci_whisker(scale(low), scale(high), y, color, cap=4, width=2))
                    label += f" {_bracket_ci(low, high)}"
                    label_x = scale(high)
                body.append(f'<circle cx="{scale(value):.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
                n = row.get("n_transcriptions")
                if n not in (None, ""):
                    label += f"  n={int(float(n))}"
                body.append(_text(label_x + 8, y + 4, label, size=11, weight=500, color=color))
    path.write_text(_svg_document(width, height, title, body), encoding="utf-8")


def _draw_error_distribution_figure(
    data: list[dict[str, Any]],
    path: Path,
    conditions: list[str],
    summary_data: list[dict[str, Any]] | None = None,
) -> None:
    """Response-level WER/CER box-and-dot plots with mean, 95% CI, median, IQR and % error-free."""
    summaries = {str(row["condition"]): row for row in (summary_data or [])}
    present_conditions = [
        condition for condition in conditions
        if any(str(row["condition"]) == condition for row in data)
    ]
    top, bottom, row_gap = 120, 70, 165
    height = int(top + bottom + row_gap * max(1, len(present_conditions)))
    label_width, chart_width = 170, 450
    panel_width = label_width + chart_width + 50
    width = 35 + panel_width * 2 + 35
    panels = [
        (35, "Word error rate", "wer", "mean_wer", "wer_ci_low", "wer_ci_high", BLUE),
        (35 + panel_width, "Character error rate", "cer", "mean_cer", "cer_ci_low", "cer_ci_high", ORANGE),
    ]
    maximum = _nice_max(
        [float(row[metric]) for row in data for metric in ("wer", "cer")],
        minimum=0.50,
        cap=1.0,
    )
    body = [
        _text(35, 34, "Transcription error rates by condition", size=20, weight=500),
        _text(
            35,
            58,
            "Dots = responses; box = middle 50% with median line; whiskers = 1.5×IQR; open circle = mean; "
            "black bar above the box = listener-clustered 95% CI of the mean. Lower is better.",
            size=12,
            color=GRAY,
        ),
    ]
    for panel_x, panel_title, metric, mean_key, low_key, high_key, color in panels:
        chart_x = panel_x + label_width
        scale = lambda value, cx=chart_x: cx + chart_width * value / maximum
        body.append(_text(panel_x, 92, panel_title, size=15, weight=500))
        _axis(body, chart_x, chart_width, maximum, top, height - bottom, "Response-level error rate")
        for row_index, condition in enumerate(present_conditions):
            group = [row for row in data if str(row["condition"]) == condition]
            values = sorted(float(row[metric]) for row in group)
            if not values:
                continue
            center_y = top + row_gap * (row_index + 0.4)
            body.append(_text(panel_x, center_y + 4, str(group[0]["condition_display_name"]), size=12, weight=500))
            stats = _box_summary(values)
            _draw_box(body, values, stats, scale, center_y, color, dot_radius=2.3, dot_opacity=0.35, jitter=(-8, -4, 0, 4, 8))
            summary = summaries.get(condition, {})
            mean_value = float(summary[mean_key]) if _finite(summary, mean_key) else stats["mean"]
            mean_label = f"Mean {_pct(mean_value)}"
            if _finite(summary, low_key, high_key):
                low, high = float(summary[low_key]), float(summary[high_key])
                body.extend(_ci_whisker(scale(low), scale(high), center_y - 24, TEXT, cap=4, width=1.8))
                mean_label += f" ({_ci_text(low, high)})"
            body.append(
                f'<circle cx="{scale(mean_value):.1f}" cy="{center_y:.1f}" r="5" fill="{BACKGROUND}" stroke="{TEXT}" stroke-width="2"/>'
            )
            zero_count = sum(1 for value in values if value == 0)
            body.append(_text(chart_x, center_y + 40, mean_label, size=12, weight=600, color=color))
            body.append(
                _text(
                    chart_x,
                    center_y + 57,
                    f"Median {_pct(stats['median'])} · IQR {_pct(stats['q1'])}–{_pct(stats['q3'])} · "
                    f"error-free {zero_count}/{len(values)} ({zero_count / len(values):.0%}) · max {_pct(max(values))}",
                    size=11,
                    color=GRAY,
                )
            )
    path.write_text(
        _svg_document(width, height, "Transcription error rates by condition", body),
        encoding="utf-8",
    )


def _draw_error_composition_figure(data: list[dict[str, Any]], path: Path) -> None:
    """Stacked substitution/deletion/insertion bars with segment values, totals and 95% CIs."""
    top, bottom, row_gap = 120, 70, 175
    height = int(top + bottom + row_gap * max(1, len(data)))
    label_width, chart_width, value_space = 170, 400, 200
    panel_width = label_width + chart_width + value_space
    width = 35 + panel_width * 2 + 35
    body = [
        _text(35, 34, "Error composition by condition", size=20, weight=500),
        _text(
            35,
            58,
            "Bars stack mean substitution, deletion and insertion rates (they sum to the mean error rate). "
            "Black bar above = 95% CI of the total; brackets = listener-clustered 95% CIs. Lower is better.",
            size=12,
            color=GRAY,
        ),
    ]
    components = [("substitution", "Substitutions", BLUE), ("deletion", "Deletions", ORANGE), ("insertion", "Insertions", GREEN)]
    legend_x = 35
    for _, label, color in components:
        body.append(f'<rect x="{legend_x}" y="76" width="12" height="12" fill="{color}"/>')
        body.append(_text(legend_x + 18, 87, label, size=12))
        legend_x += 120
    panels = [
        (35, "Word error rate", "", "mean_wer", "wer_ci_low", "wer_ci_high"),
        (35 + panel_width, "Character error rate", "character_", "mean_cer", "cer_ci_low", "cer_ci_high"),
    ]
    for panel_x, panel_title, prefix, total_key, total_low_key, total_high_key in panels:
        chart_x = panel_x + label_width
        totals = [sum(float(row[f"{prefix}{name}_rate"]) for name, _, _ in components) for row in data]
        highs = [float(row[total_high_key]) for row in data if _finite(row, total_high_key)]
        maximum = _nice_max(totals + highs)
        scale = lambda value, cx=chart_x, m=maximum: cx + chart_width * value / m
        body.append(_text(panel_x, 112, panel_title, size=15, weight=500))
        _axis(body, chart_x, chart_width, maximum, top + 10, height - bottom, "Mean error rate")
        for index, row in enumerate(data):
            bar_y = top + row_gap * index + 55
            body.append(_text(panel_x, bar_y + 4, row["display_name"], size=12, weight=500))
            current_x = chart_x
            total = totals[index]
            for name, _, color in components:
                value = float(row[f"{prefix}{name}_rate"])
                segment_width = chart_width * value / maximum
                body.append(f'<rect x="{current_x:.1f}" y="{bar_y-12:.1f}" width="{segment_width:.1f}" height="24" fill="{color}"/>')
                if segment_width >= 40:
                    body.append(_text(current_x + segment_width / 2, bar_y + 4, _pct(value), anchor="middle", size=11, weight=600, color=BACKGROUND))
                current_x += segment_width
            label_x = current_x
            # Prefer the bootstrapped mean so the label always matches its CI.
            shown_total = float(row[total_key]) if _finite(row, total_key) else total
            total_label = f"Total {_pct(shown_total)}"
            if _finite(row, total_low_key, total_high_key):
                low, high = float(row[total_low_key]), float(row[total_high_key])
                body.extend(_ci_whisker(scale(low), scale(high), bar_y - 22, TEXT, cap=4, width=1.8))
                total_label += f" {_bracket_ci(low, high)}"
            body.append(_text(label_x + 8, bar_y + 4, total_label, size=12, weight=600))
            # Breakdown lines under the bar: value, CI and share of all errors.
            for line_index, (name, label, color) in enumerate(components):
                key = f"{prefix}{name}_rate"
                value = float(row[key])
                line_y = bar_y + 34 + line_index * 17
                body.append(f'<rect x="{chart_x}" y="{line_y-9:.1f}" width="10" height="10" fill="{color}"/>')
                text = f"{label} {_pct(value)}"
                if _finite(row, f"{key}_ci_low", f"{key}_ci_high"):
                    text += f" {_bracket_ci(float(row[f'{key}_ci_low']), float(row[f'{key}_ci_high']))}"
                if total > 0:
                    text += f" · {value / total:.0%} of errors"
                body.append(_text(chart_x + 16, line_y, text, size=11, color=TEXT))
    path.write_text(
        _svg_document(width, height, "Word and character error composition by condition", body),
        encoding="utf-8",
    )


def _draw_xxx_figure(data: list[dict[str, Any]], path: Path) -> None:
    """Share of responses with any / whole-response xxx, with counts, token totals and 95% CIs."""
    top, bottom, row_gap = 120, 70, 130
    height = int(top + bottom + row_gap * max(1, len(data)))
    left, chart_width, value_space = 230, 420, 390
    width = left + chart_width + value_space
    measures = [
        ("xxx_response_rate", "xxx_response_count", "Any xxx", PURPLE, -14),
        ("whole_response_xxx_rate", "whole_response_xxx_count", "Whole-response xxx", ORANGE, 14),
    ]
    candidates: list[float] = []
    for row in data:
        for rate_key, *_ in measures:
            candidates.append(float(row[rate_key]))
            if _finite(row, f"{rate_key}_ci_high"):
                candidates.append(float(row[f"{rate_key}_ci_high"]))
    maximum = _nice_max(candidates)
    scale = lambda value: left + chart_width * value / maximum
    body = [
        _text(35, 34, "Explicit unintelligibility markers by condition", size=20, weight=500),
        _text(
            35,
            58,
            "Share of responses containing any xxx (partial) or consisting only of xxx (complete unintelligibility). "
            "Whiskers = listener-clustered 95% CIs.",
            size=12,
            color=GRAY,
        ),
        f'<circle cx="41" cy="84" r="6" fill="{PURPLE}"/>',
        _text(53, 88, "Any xxx", size=12),
        f'<rect x="135" y="78" width="11" height="11" fill="{ORANGE}"/>',
        _text(153, 88, "Whole-response xxx", size=12),
    ]
    _axis(body, left, chart_width, maximum, top, height - bottom, "Share of transcription responses")
    for index, row in enumerate(data):
        center_y = top + row_gap * (index + 0.5)
        n = int(float(row.get("n_transcriptions", 0) or 0))
        body.append(_text(35, center_y - 2, row["display_name"], size=12, weight=500))
        tokens = row.get("total_xxx_tokens")
        if tokens not in (None, ""):
            body.append(_text(35, center_y + 15, f"{int(float(tokens))} xxx tokens in {n} responses", size=11, color=GRAY))
        for rate_key, count_key, label, color, offset in measures:
            y = center_y + offset
            rate = float(row[rate_key])
            count = row.get(count_key)
            count = int(float(count)) if count not in (None, "") else round(rate * n)
            text = f"{label}: {count}/{n} ({_pct(rate)}"
            label_x = scale(rate)
            body.append(_line(left, y, scale(rate), y, LIGHT_GRAY, 8))
            if _finite(row, f"{rate_key}_ci_low", f"{rate_key}_ci_high"):
                low, high = float(row[f"{rate_key}_ci_low"]), float(row[f"{rate_key}_ci_high"])
                body.extend(_ci_whisker(scale(low), scale(high), y, color, cap=4, width=1.8))
                label_x = max(label_x, scale(high))
                text += f", {_ci_text(low, high)}"
            text += ")"
            if label.startswith("Any"):
                body.append(f'<circle cx="{scale(rate):.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
            else:
                body.append(f'<rect x="{scale(rate)-5.5:.1f}" y="{y-5.5:.1f}" width="11" height="11" fill="{color}"/>')
            body.append(_text(label_x + 10, y + 4, text, size=11, weight=500, color=color))
    path.write_text(
        _svg_document(width, height, "Explicit unintelligibility markers by condition", body),
        encoding="utf-8",
    )


def _draw_consistency_figure(
    data: list[dict[str, Any]],
    path: Path,
    conditions: list[str],
) -> None:
    """Within-utterance listener SD as box-and-dot plots, with summary numbers and labelled outliers."""
    present_conditions = [
        condition for condition in conditions
        if any(str(row["condition"]) == condition for row in data)
    ]
    top, bottom, row_gap = 120, 70, 165
    height = int(top + bottom + row_gap * max(1, len(present_conditions)))
    label_width, chart_width = 170, 450
    panel_width = label_width + chart_width + 50
    width = 35 + panel_width * 2 + 35
    panels = [
        (35, "Word error rate variation", "listener_sd_wer", BLUE),
        (35 + panel_width, "Character error rate variation", "listener_sd_cer", ORANGE),
    ]
    body = [
        _text(35, 34, "Inter-listener consistency by condition", size=20, weight=500),
        _text(
            35,
            58,
            "Each dot is one utterance: SD of the error rate across its listeners. Box = middle 50% with median line; "
            "whiskers = 1.5×IQR; open circle = mean; outliers labelled with utterance ID. Lower = more consistent.",
            size=12,
            color=GRAY,
        ),
    ]
    for panel_x, panel_title, metric, color in panels:
        chart_x = panel_x + label_width
        maximum = _nice_max([float(row[metric]) for row in data], minimum=0.20)
        scale = lambda value, cx=chart_x, m=maximum: cx + chart_width * value / m
        body.append(_text(panel_x, 92, panel_title, size=15, weight=500))
        _axis(body, chart_x, chart_width, maximum, top, height - bottom, "Within-utterance listener SD")
        for row_index, condition in enumerate(present_conditions):
            group = sorted(
                (row for row in data if str(row["condition"]) == condition),
                key=lambda row: float(row[metric]),
            )
            values = [float(row[metric]) for row in group]
            if not values:
                continue
            center_y = top + row_gap * (row_index + 0.4)
            body.append(_text(panel_x, center_y + 4, str(group[0]["condition_display_name"]), size=12, weight=500))
            stats = _box_summary(values)
            jitter = (-10, -5, 0, 5, 10)
            _draw_box(body, values, stats, scale, center_y, color, dot_radius=3.2, dot_opacity=0.45, jitter=jitter)
            body.append(
                f'<circle cx="{scale(stats["mean"]):.1f}" cy="{center_y:.1f}" r="5" fill="{BACKGROUND}" stroke="{TEXT}" stroke-width="2"/>'
            )
            outlier_index = 0
            for value_index, row in enumerate(group):
                value = values[value_index]
                if value <= stats["whisker_high"]:
                    continue
                dot_y = center_y + jitter[value_index % len(jitter)]
                label_y = dot_y - 9 if outlier_index % 2 == 0 else dot_y + 18
                short_id = str(row["utterance_id"]).split("_")[0]
                body.append(_text(scale(value), label_y, f"{short_id}: {_pct(value)}", anchor="middle", size=10, color=TEXT))
                outlier_index += 1
            zero_count = sum(1 for value in values if value == 0)
            body.append(
                _text(
                    chart_x,
                    center_y + 40,
                    f"Median {_pct(stats['median'])} · IQR {_pct(stats['q1'])}–{_pct(stats['q3'])} · Mean {_pct(stats['mean'])}",
                    size=12,
                    weight=600,
                    color=color,
                )
            )
            body.append(
                _text(
                    chart_x,
                    center_y + 57,
                    f"Full agreement (SD = 0): {zero_count}/{len(values)} utterances ({zero_count / len(values):.0%}) · "
                    f"max {_pct(max(values))}",
                    size=11,
                    color=GRAY,
                )
            )
    path.write_text(
        _svg_document(width, height, "Inter-listener consistency by condition", body),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    distribution_data = _error_distribution_plot_data(scored, conditions, settings)
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
    write_csv(output_root / "error_rate_distribution_plot_data.csv", distribution_data)
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
    _draw_error_distribution_figure(
        distribution_data,
        output_root / "error_rate_distributions.svg",
        conditions,
        summary_data=condition_data,
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
        "error_rate_distributions.svg",
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
