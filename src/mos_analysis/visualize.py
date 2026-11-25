from __future__ import annotations

import csv
import hashlib
import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Iterable

from .summarize import (
    listener_mean_t_ci,
    summarize_conditions,
    summarize_rating_distribution,
)


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B61A8"
PINK = "#CC79A7"
SKY_BLUE = "#56B4E9"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
TEXT = "#1F2937"
BACKGROUND = "#FFFFFF"
CONDITION_COLORS = [BLUE, ORANGE, GREEN, PURPLE, PINK, SKY_BLUE]
RATING_COLORS = {
    1: "#B2182B",
    2: "#EF8A62",
    3: "#D1D5DB",
    4: "#67A9CF",
    5: "#2166AC",
}
RATING_LABELS = {1: "Bad", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite(row: dict[str, Any], *keys: str) -> bool:
    """True when every key holds a finite number."""
    for key in keys:
        value = _number(row.get(key))
        if value is None or math.isnan(value):
            return False
    return True


def _mos(value: float) -> str:
    return f"{value:.2f}"


def _mos_ci(low: float, high: float) -> str:
    return f"[{low:.2f}–{high:.2f}]"


def _signed_ci(low: float, high: float) -> str:
    return f"[{low:+.2f} to {high:+.2f}]"


def _rating(value: Any) -> str:
    """Format a rating-scale value without a trailing .0 (3.0 -> '3', 2.5 -> '2.5')."""
    number = float(value)
    return f"{number:g}"


def _condition_colors(conditions: list[str]) -> dict[str, str]:
    return {
        condition: CONDITION_COLORS[index % len(CONDITION_COLORS)]
        for index, condition in enumerate(conditions)
    }


# --- Student-t critical value (no SciPy dependency) -------------------------


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        result *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return result


def _regularized_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _t_cdf(t: float, degrees_of_freedom: float) -> float:
    x = degrees_of_freedom / (degrees_of_freedom + t * t)
    tail = 0.5 * _regularized_beta(degrees_of_freedom / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def _t_critical(degrees_of_freedom: int, confidence_level: float) -> float:
    """Two-sided Student-t critical value, e.g. df=19, 95% -> 2.093."""
    target = 1.0 - (1.0 - confidence_level) / 2.0
    low, high = 0.0, 1000.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if _t_cdf(middle, degrees_of_freedom) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


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
    halo: bool = False,
) -> str:
    # A white outline behind the glyphs keeps labels readable where they cross grid lines.
    halo_attr = (
        f' stroke="{BACKGROUND}" stroke-width="4" stroke-linejoin="round" paint-order="stroke"'
        if halo else ""
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}"{halo_attr}>{_escape(value)}</text>'
    )


def _line(
    x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1, dash: str = ""
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr}/>'
    )


def _h_whisker(x_low: float, x_high: float, y: float, color: str, cap: float = 6, width: float = 2.5) -> list[str]:
    return [
        _line(x_low, y, x_high, y, color, width),
        _line(x_low, y - cap, x_low, y + cap, color, 1.8),
        _line(x_high, y - cap, x_high, y + cap, color, 1.8),
    ]


def _v_whisker(x: float, y_low: float, y_high: float, color: str, cap: float = 7, width: float = 2.5) -> list[str]:
    return [
        _line(x, y_low, x, y_high, color, width),
        _line(x - cap, y_low, x + cap, y_low, color, 1.8),
        _line(x - cap, y_high, x + cap, y_high, color, 1.8),
    ]


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


def _mos_axis(body: list[str], chart_x: float, chart_width: float, top: float, bottom: float, title: str, size: int = 12) -> None:
    for rating in range(1, 6):
        x = _scale(float(rating), chart_x, chart_width)
        body.append(_line(x, top, x, bottom, LIGHT_GRAY))
        body.append(_text(x, bottom + 18, f"{rating} {RATING_LABELS[rating]}", anchor="middle", size=size - 1, color=GRAY))
    body.append(_text(chart_x + chart_width / 2, bottom + 42, title, anchor="middle", size=size, color=GRAY))


def _label_fits_segment(segment_width: float, label: str) -> bool:
    """Return whether a percentage label fits safely inside a bar segment."""
    estimated_text_width = len(label) * 6.5
    horizontal_padding = 10.0
    return segment_width >= estimated_text_width + horizontal_padding


# ---------------------------------------------------------------------------
# Plot data
# ---------------------------------------------------------------------------


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


def _item_key(row: dict[str, Any]) -> str:
    """Unique stimulus key: different speakers can share the same item_id (e.g. sentence number)."""
    speaker = str(row.get("speaker_id", "")).strip()
    item = str(row.get("item_id", "")).strip()
    return f"{speaker}|{item}" if speaker else item


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
    items_by_condition: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        items_by_condition[str(row.get("condition", ""))].add(_item_key(row))
    for row in output:
        condition = str(row["condition"])
        row["condition_display_name"] = display_names.get(condition, condition)
        # Recount items by speaker + item_id; summarize_conditions counts item_id alone,
        # which merges recordings of the same sentence by different speakers.
        row["n_items"] = len(items_by_condition.get(condition, set()))
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


def _paired_difference_data(
    rows: list[dict[str, Any]],
    conditions: list[str],
    display_names: dict[str, str],
    confidence_level: float,
    *,
    group_key: str | None = None,
    group_order: list[str] | None = None,
    group_display_names: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Difference of each condition from the first (reference) condition, paired by listener.

    Each listener's mean rating per condition (within group, if given) is computed first;
    the CI is a Student-t interval over the per-listener differences, so listeners who
    rate everything high or low do not inflate the uncertainty of the comparison.
    """
    if len(conditions) < 2:
        return []
    reference = conditions[0]
    ratings: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row.get("mos_rating"))
        if value is None:
            continue
        group = str(row.get(group_key, "")) if group_key else ""
        key = (group, str(row.get("participant_id", "")), str(row.get("session_id", "")), str(row.get("condition", "")))
        ratings[key].append(value)
    listener_means: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for (group, participant, session, condition), values in ratings.items():
        listener_means[(group, participant, session)][condition] = mean(values)
    groups = group_order if group_key else [""]
    output: list[dict[str, Any]] = []
    for group in groups or []:
        for condition in conditions[1:]:
            differences = [
                means[condition] - means[reference]
                for (listener_group, _, _), means in listener_means.items()
                if listener_group == group and condition in means and reference in means
            ]
            n = len(differences)
            if n == 0:
                continue
            mean_difference = mean(differences)
            if n >= 2:
                sd_difference = stdev(differences)
                margin = _t_critical(n - 1, confidence_level) * sd_difference / math.sqrt(n)
            else:
                sd_difference, margin = math.nan, math.nan
            record: dict[str, Any] = {}
            if group_key:
                record[group_key] = group
                record["group_display_name"] = (group_display_names or {}).get(group, group)
            record.update(
                {
                    "condition": condition,
                    "condition_display_name": display_names.get(condition, condition),
                    "reference_condition": reference,
                    "reference_display_name": display_names.get(reference, reference),
                    "n_paired_listeners": n,
                    "mean_difference": mean_difference,
                    "sd_difference": sd_difference,
                    "difference_ci_low": mean_difference - margin,
                    "difference_ci_high": mean_difference + margin,
                    "n_listeners_higher": sum(1 for value in differences if value > 0),
                    "n_listeners_lower": sum(1 for value in differences if value < 0),
                    "n_listeners_tied": sum(1 for value in differences if value == 0),
                    "confidence_level": confidence_level,
                    "ci_method": "paired_listener_mean_student_t",
                }
            )
            output.append(record)
    return output


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _draw_mos_by_condition(
    data: list[dict[str, Any]],
    path: Path,
    difference_data: list[dict[str, Any]] | None = None,
) -> None:
    """Mean MOS per condition with CI and summary numbers, plus paired differences."""
    conditions = [str(row["condition"]) for row in data]
    colors = _condition_colors(conditions)
    width = 1060
    left, chart_width = 230.0, 520.0
    top, row_step = 112.0, 82.0
    main_bottom = top + row_step * max(1, len(data))
    body = [
        _text(18, 34, "Mean opinion score by condition", size=21, weight=500),
        _text(
            18,
            60,
            "Points are means of listener-level means; whiskers and [brackets] are Student-t 95% confidence intervals. Higher is better.",
            size=12,
            color=GRAY,
        ),
    ]
    _mos_axis(body, left, chart_width, top - 20, main_bottom - 10, "Mean opinion score (1–5)")
    for index, row in enumerate(data):
        condition = str(row["condition"])
        color = colors[condition]
        y = top + index * row_step + 18
        point = _scale(float(row["mean_mos"]), left, chart_width)
        body.append(_text(left - 18, y + 5, row["condition_display_name"], anchor="end", size=13, weight=500))
        label = _mos(float(row["mean_mos"]))
        label_x = point
        if _finite(row, "mos_ci_low", "mos_ci_high"):
            low, high = float(row["mos_ci_low"]), float(row["mos_ci_high"])
            body.extend(_h_whisker(_scale(low, left, chart_width), _scale(high, left, chart_width), y, color))
            label += f" {_mos_ci(low, high)}"
            label_x = _scale(high, left, chart_width)
        body.append(f'<circle cx="{point:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        body.append(_text(label_x + 10, y + 5, label, size=13, weight=600, color=color, halo=True))
        details = []
        if _finite(row, "sd_mos"):
            details.append(f"SD {float(row['sd_mos']):.2f}")
        if _finite(row, "median_mos", "q1_mos", "q3_mos"):
            details.append(f"median {_rating(row['median_mos'])} (IQR {_rating(row['q1_mos'])}–{_rating(row['q3_mos'])})")
        counts = []
        if _finite(row, "n_ratings"):
            counts.append(f"{int(float(row['n_ratings']))} ratings")
        if _finite(row, "n_participants"):
            counts.append(f"{int(float(row['n_participants']))} listeners")
        if _finite(row, "n_items"):
            counts.append(f"{int(float(row['n_items']))} items")
        if counts:
            details.append(", ".join(counts))
        if details:
            body.append(_text(left, y + 30, " · ".join(details), size=11, color=GRAY, halo=True))

    height = main_bottom + 60
    if difference_data:
        reference_name = str(difference_data[0]["reference_display_name"])
        section_top = main_bottom + 85
        bounds = [abs(float(row[key])) for row in difference_data for key in ("difference_ci_low", "difference_ci_high", "mean_difference") if _finite(row, key)]
        step = 0.25 if max(bounds, default=0) * 1.15 <= 1.0 else 0.5
        limit = max(0.5, math.ceil(max(bounds, default=0) * 1.15 / step) * step)
        diff_scale = lambda value: left + chart_width * (value + limit) / (2 * limit)
        body.append(_text(18, section_top, f"Difference from {reference_name} (paired by listener)", size=15, weight=500))
        body.append(
            _text(
                18,
                section_top + 20,
                "Each listener's mean for the condition minus their mean for the reference; whisker = Student-t 95% CI of the mean difference. Right of 0 = rated higher.",
                size=12,
                color=GRAY,
            )
        )
        diff_top = section_top + 45
        diff_step = 64.0
        diff_bottom = diff_top + diff_step * len(difference_data)
        tick_count = int(round(2 * limit / step))
        for tick_index in range(tick_count + 1):
            tick = -limit + tick_index * step
            x = diff_scale(tick)
            body.append(_line(x, diff_top, x, diff_bottom - 10, LIGHT_GRAY))
            body.append(_text(x, diff_bottom + 8, f"{tick:+.2f}" if tick else "0", anchor="middle", size=11, color=GRAY))
        body.append(_line(diff_scale(0), diff_top, diff_scale(0), diff_bottom - 10, GRAY, 1.5, "4 3"))
        body.append(_text(left + chart_width / 2, diff_bottom + 32, "Difference in mean opinion score", anchor="middle", size=12, color=GRAY))
        for index, row in enumerate(difference_data):
            condition = str(row["condition"])
            color = colors.get(condition, BLUE)
            y = diff_top + index * diff_step + 18
            value = float(row["mean_difference"])
            body.append(_text(left - 18, y + 5, row["condition_display_name"], anchor="end", size=13, weight=500))
            label = f"{value:+.2f}"
            label_x = diff_scale(value)
            if _finite(row, "difference_ci_low", "difference_ci_high"):
                low, high = float(row["difference_ci_low"]), float(row["difference_ci_high"])
                body.extend(_h_whisker(diff_scale(low), diff_scale(high), y, color))
                label += f" {_signed_ci(low, high)}"
                label_x = diff_scale(high)
            body.append(f'<circle cx="{diff_scale(value):.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
            body.append(_text(label_x + 10, y + 5, label, size=13, weight=600, color=color, halo=True))
            n = int(float(row["n_paired_listeners"]))
            body.append(
                _text(
                    left,
                    y + 30,
                    f"Rated higher by {int(float(row['n_listeners_higher']))}/{n} listeners, lower by "
                    f"{int(float(row['n_listeners_lower']))}, tied {int(float(row['n_listeners_tied']))}"
                    + (f" · SD of differences {float(row['sd_difference']):.2f}" if _finite(row, "sd_difference") else ""),
                    size=11,
                    color=GRAY,
                    halo=True,
                )
            )
        height = diff_bottom + 50
    path.write_text(
        _svg_document(int(width), int(height), "Mean opinion score by condition", "Mean MOS and listener-level Student-t 95 percent confidence interval for each audio condition, with paired differences.", body),
        encoding="utf-8",
    )


def _draw_rating_distribution(
    data: list[dict[str, Any]],
    path: Path,
    conditions: list[str],
    condition_data: list[dict[str, Any]] | None = None,
) -> None:
    """100% stacked bars with every percentage, counts, and top-2 / bottom-2 box summaries."""
    width = 1180
    left, chart_width = 200.0, 600.0
    stats_x = left + chart_width + 30
    top, row_step = 150.0, 112.0
    summaries = {str(row["condition"]): row for row in (condition_data or [])}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data:
        grouped[str(row["condition"])].append(row)
    present = [condition for condition in conditions if grouped.get(condition)]
    height = top + row_step * max(1, len(present)) + 40
    body = [
        _text(18, 34, "MOS rating distribution by condition", size=21, weight=500),
        _text(
            18,
            60,
            "Each bar shows the share of ratings at each score; counts below. Right: mean (Student-t 95% CI), median, and share of high (4–5) and low (1–2) ratings.",
            size=12,
            color=GRAY,
        ),
    ]
    legend_x = left
    for rating in range(1, 6):
        body.append(f'<rect x="{legend_x:.1f}" y="86" width="14" height="14" fill="{RATING_COLORS[rating]}"/>')
        body.append(_text(legend_x + 20, 98, f"{rating} {RATING_LABELS[rating]}", size=12))
        legend_x += 105
    grid_bottom = top + row_step * len(present) - 40
    for percentage in range(0, 101, 20):
        x = left + chart_width * percentage / 100
        body.append(_line(x, top - 14, x, grid_bottom, LIGHT_GRAY))
        body.append(_text(x, grid_bottom + 16, f"{percentage}%", anchor="middle", size=11, color=GRAY))
    body.append(_text(left + chart_width / 2, grid_bottom + 38, "Share of ratings", anchor="middle", size=12, color=GRAY))
    for index, condition in enumerate(present):
        rows = sorted(grouped[condition], key=lambda row: int(row["mos_rating"]))
        y = top + index * row_step
        body.append(_text(left - 18, y + 20, rows[0]["condition_display_name"], anchor="end", size=13, weight=500))
        cursor = left
        small_labels: list[tuple[float, str, int]] = []
        proportions = {int(row["mos_rating"]): float(row["rating_proportion"]) for row in rows}
        counts = {int(row["mos_rating"]): int(float(row["rating_count"])) for row in rows}
        total = int(float(rows[0].get("n_ratings", sum(counts.values())) or sum(counts.values())))
        for row in rows:
            rating = int(row["mos_rating"])
            proportion = proportions[rating]
            segment_width = chart_width * proportion
            body.append(f'<rect x="{cursor:.1f}" y="{y:.1f}" width="{segment_width:.1f}" height="30" fill="{RATING_COLORS[rating]}"/>')
            percentage_label = f"{proportion:.0%}"
            if _label_fits_segment(segment_width, percentage_label):
                label_color = TEXT if rating == 3 else BACKGROUND
                body.append(_text(cursor + segment_width / 2, y + 20, percentage_label, anchor="middle", size=11, weight=600, color=label_color))
            elif proportion > 0:
                small_labels.append((cursor + segment_width / 2, f"{proportion:.0%}", rating))
            cursor += segment_width
        for label_index, (x, label, rating) in enumerate(small_labels):
            body.append(_text(x, y - 6 - 11 * (label_index % 2), label, anchor="middle", size=10, weight=600, color=RATING_COLORS[rating]))
        count_text = " · ".join(f"{rating}: {counts[rating]}" for rating in sorted(counts))
        body.append(_text(left, y + 50, f"Counts — {count_text}  (n = {total})", size=11, color=GRAY))
        # Summary column.
        rating_mean = sum(rating * proportion for rating, proportion in proportions.items())
        cumulative, median_rating = 0.0, None
        for rating in sorted(proportions):
            cumulative += proportions[rating]
            if median_rating is None and cumulative >= 0.5:
                median_rating = rating
        summary = summaries.get(condition, {})
        mean_line = f"Mean {_mos(float(summary['mean_mos']) if _finite(summary, 'mean_mos') else rating_mean)}"
        if _finite(summary, "mos_ci_low", "mos_ci_high"):
            mean_line += f" {_mos_ci(float(summary['mos_ci_low']), float(summary['mos_ci_high']))}"
        shown_median = _rating(summary["median_mos"]) if _finite(summary, "median_mos") else median_rating
        mean_line += f" · median {shown_median}"
        high_share = sum(proportions.get(rating, 0.0) for rating in (4, 5))
        low_share = sum(proportions.get(rating, 0.0) for rating in (1, 2))
        body.append(_text(stats_x, y + 6, mean_line, size=12, weight=600))
        body.append(_text(stats_x, y + 24, f"Good or excellent (4–5): {high_share:.0%}", size=12, color=RATING_COLORS[5]))
        body.append(_text(stats_x, y + 42, f"Poor or bad (1–2): {low_share:.0%}", size=12, color=RATING_COLORS[1]))
    path.write_text(
        _svg_document(width, int(height), "MOS rating distribution by condition", "One hundred percent stacked bars show rating proportions from one through five for each condition.", body),
        encoding="utf-8",
    )


def _jitter(participant: str, session: str) -> float:
    """Stable horizontal offset per listener (same in every condition, so pairs line up)."""
    digest = hashlib.sha256(f"{participant}|{session}".encode()).digest()
    return (int.from_bytes(digest[:2], "big") / 65535.0 - 0.5) * 50.0


def _draw_listener_distribution(
    listener_data: list[dict[str, Any]],
    condition_data: list[dict[str, Any]],
    path: Path,
    conditions: list[str],
    difference_data: list[dict[str, Any]] | None = None,
) -> None:
    """Listener means per condition, linked across conditions, with mean, CI and spread numbers."""
    colors = _condition_colors(conditions)
    width, height = 1000, 680
    left, right, top, bottom = 90.0, 45.0, 120.0, 150.0
    chart_width = width - left - right
    chart_height = height - top - bottom
    condition_x = {
        condition: left + (index + 0.5) * chart_width / len(conditions)
        for index, condition in enumerate(conditions)
    }
    y_of = lambda value: top + (5 - value) / 4 * chart_height
    condition_lookup = {str(row["condition"]): row for row in condition_data}
    body = [
        _text(18, 34, "Listener-level MOS by condition", size=21, weight=500),
        _text(
            18,
            60,
            "Small points are one listener's mean per condition; gray lines link the same listener across conditions. "
            "Diamond and whisker = overall mean and Student-t 95% CI.",
            size=12,
            color=GRAY,
        ),
    ]
    if difference_data:
        lines = []
        for row in difference_data:
            n = int(float(row["n_paired_listeners"]))
            text = (
                f"{row['condition_display_name']} vs {row['reference_display_name']}: "
                f"{float(row['mean_difference']):+.2f}"
            )
            if _finite(row, "difference_ci_low", "difference_ci_high"):
                text += f" {_signed_ci(float(row['difference_ci_low']), float(row['difference_ci_high']))}"
            text += (
                f"; rated higher by {int(float(row['n_listeners_higher']))}/{n} listeners, "
                f"lower by {int(float(row['n_listeners_lower']))}, tied {int(float(row['n_listeners_tied']))}"
            )
            lines.append(text)
        for line_index, text in enumerate(lines):
            body.append(_text(18, 84 + line_index * 17, text, size=12, weight=600))
    for rating in range(1, 6):
        y = y_of(rating)
        body.append(_line(left, y, left + chart_width, y, LIGHT_GRAY))
        body.append(_text(left - 14, y + 4, rating, anchor="end", size=12, color=GRAY))
    # Lines linking the same listener in adjacent conditions.
    by_listener: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in listener_data:
        by_listener[(str(row["participant_id"]), str(row["session_id"]))][str(row["condition"])] = float(row["listener_mean_mos"])
    for (participant, session), means in by_listener.items():
        offset = _jitter(participant, session)
        for first, second in zip(conditions, conditions[1:]):
            if first in means and second in means:
                body.append(
                    _line(condition_x[first] + offset, y_of(means[first]), condition_x[second] + offset, y_of(means[second]), GRAY, 1)
                    .replace("/>", ' stroke-opacity="0.25"/>')
                )
    for row in listener_data:
        condition = str(row["condition"])
        if condition not in condition_x:
            continue
        x = condition_x[condition] + _jitter(str(row["participant_id"]), str(row["session_id"]))
        body.append(f'<circle cx="{x:.1f}" cy="{y_of(float(row["listener_mean_mos"])):.1f}" r="3.5" fill="{GRAY}" fill-opacity="0.5"/>')
    for condition in conditions:
        row = condition_lookup.get(condition)
        if row is None:
            continue
        color = colors[condition]
        x = condition_x[condition] + 48
        point_y = y_of(float(row["mean_mos"]))
        if _finite(row, "mos_ci_low", "mos_ci_high"):
            body.extend(_v_whisker(x, y_of(float(row["mos_ci_low"])), y_of(float(row["mos_ci_high"])), color))
        body.append(
            f'<path d="M {x:.1f} {point_y - 8:.1f} L {x + 8:.1f} {point_y:.1f} L {x:.1f} {point_y + 8:.1f} L {x - 8:.1f} {point_y:.1f} Z" fill="{color}"/>'
        )
        body.append(_text(x + 14, point_y - 2, f"Mean {_mos(float(row['mean_mos']))}", size=12, weight=600, color=color))
        if _finite(row, "mos_ci_low", "mos_ci_high"):
            body.append(_text(x + 14, point_y + 14, _mos_ci(float(row["mos_ci_low"]), float(row["mos_ci_high"])), size=11, color=color))
        values = [
            float(item["listener_mean_mos"]) for item in listener_data if str(item["condition"]) == condition
        ]
        center = condition_x[condition]
        body.append(_text(center, height - 112, row["condition_display_name"], anchor="middle", size=13, weight=500))
        if values:
            spread = f"SD {stdev(values):.2f} · " if len(values) > 1 else ""
            body.append(
                _text(center, height - 93, f"{len(values)} listeners · median {median(values):.2f}", anchor="middle", size=11, color=GRAY)
            )
            body.append(
                _text(center, height - 77, f"{spread}range {min(values):.2f}–{max(values):.2f}", anchor="middle", size=11, color=GRAY)
            )
    axis_y = top + chart_height / 2
    body.append(
        f'<text x="24" y="{axis_y:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 24 {axis_y:.1f})" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="12" '
        f'fill="{GRAY}">Listener mean MOS</text>'
    )
    path.write_text(
        _svg_document(width, height, "Listener-level MOS by condition", "Listener condition means are shown as linked points with condition means and Student-t confidence intervals.", body),
        encoding="utf-8",
    )


def _draw_speaker_condition(
    data: list[dict[str, Any]],
    path: Path,
    speakers: list[str],
    conditions: list[str],
    difference_data: list[dict[str, Any]] | None = None,
) -> None:
    """Small multiples per speaker: mean, CI and n per condition, plus the paired difference."""
    colors = _condition_colors(conditions)
    columns = 2
    rows_count = (len(speakers) + columns - 1) // columns
    panel_width = 600.0
    width = int(18 + panel_width * columns + 10)
    row_step = 40.0
    differences = {
        (str(row["speaker_id"]), str(row["condition"])): row for row in (difference_data or [])
    }
    n_difference_lines = max(0, len(conditions) - 1) if difference_data else 0
    panel_height = 78 + row_step * len(conditions) + 50 + 18 * n_difference_lines + 5
    start_y = 112.0
    grouped = {(str(row["speaker_id"]), str(row["condition"])): row for row in data}
    body = [
        _text(18, 34, "Mean opinion score by speaker and condition", size=21, weight=500),
        _text(
            18,
            60,
            "Points are means of listener-level means; whiskers and [brackets] are Student-t 95% CIs. "
            "Difference lines are paired by listener. All panels use the same 1–5 scale.",
            size=12,
            color=GRAY,
        ),
    ]
    legend_x = 18.0
    for condition in conditions:
        name = next((str(row["condition_display_name"]) for row in data if str(row["condition"]) == condition), condition)
        body.append(f'<circle cx="{legend_x + 5:.1f}" cy="83" r="5" fill="{colors[condition]}"/>')
        body.append(_text(legend_x + 16, 87, name, size=12))
        legend_x += 30 + len(name) * 7
    for speaker_index, speaker in enumerate(speakers):
        column = speaker_index % columns
        row_index = speaker_index // columns
        panel_x = 18.0 + column * panel_width
        panel_y = start_y + row_index * panel_height
        label_width = 150.0
        chart_x = panel_x + label_width
        chart_width = 290.0
        chart_top = panel_y + 66.0
        chart_bottom = chart_top + row_step * len(conditions) - 12
        first = next((grouped[(speaker, c)] for c in conditions if (speaker, c) in grouped), None)
        speaker_name = first["speaker_display_name"] if first else speaker
        body.append(_text(panel_x, panel_y + 18, speaker_name, size=14, weight=600))
        if first is not None:
            body.append(
                _text(
                    panel_x,
                    panel_y + 35,
                    f"{int(float(first['n_ratings']))} ratings from {int(float(first['n_listener_means']))} listeners per condition",
                    size=11,
                    color=GRAY,
                )
            )
        for rating in range(1, 6):
            x = _scale(float(rating), chart_x, chart_width)
            body.append(_line(x, chart_top - 12, x, chart_bottom, LIGHT_GRAY))
            body.append(_text(x, chart_bottom + 16, rating, anchor="middle", size=11, color=GRAY))
        body.append(_text(chart_x + chart_width / 2, chart_bottom + 34, "Mean opinion score", anchor="middle", size=11, color=GRAY))
        for condition_index, condition in enumerate(conditions):
            item = grouped.get((speaker, condition))
            if item is None:
                continue
            color = colors[condition]
            y = chart_top + condition_index * row_step
            point = _scale(float(item["mean_mos"]), chart_x, chart_width)
            body.append(_text(chart_x - 12, y + 4, item["condition_display_name"], anchor="end", size=11))
            label = _mos(float(item["mean_mos"]))
            label_x = point
            if _finite(item, "mos_ci_low", "mos_ci_high"):
                low, high = float(item["mos_ci_low"]), float(item["mos_ci_high"])
                body.extend(_h_whisker(_scale(low, chart_x, chart_width), _scale(high, chart_x, chart_width), y, color, cap=5, width=2.2))
                label += f" {_mos_ci(low, high)}"
                label_x = _scale(high, chart_x, chart_width)
            body.append(f'<circle cx="{point:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
            body.append(_text(label_x + 8, y + 4, label, size=11, weight=600, color=color, halo=True))
        line_y = chart_bottom + 58
        for condition in conditions[1:]:
            row = differences.get((speaker, condition))
            if row is None:
                continue
            value = float(row["mean_difference"])
            text = f"{row['condition_display_name']} − {row['reference_display_name']}: {value:+.2f}"
            if _finite(row, "difference_ci_low", "difference_ci_high"):
                low, high = float(row["difference_ci_low"]), float(row["difference_ci_high"])
                text += f" {_signed_ci(low, high)}"
            text += f" · higher for {int(float(row['n_listeners_higher']))}/{int(float(row['n_paired_listeners']))} listeners"
            body.append(_text(panel_x, line_y, text, size=11, weight=600, color=TEXT))
            line_y += 18
    figure_height = int(start_y + rows_count * panel_height + 10)
    path.write_text(
        _svg_document(width, figure_height, "Mean opinion score by speaker and condition", "Small multiples compare condition MOS with listener-level Student-t intervals for every speaker.", body),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    difference_data = _paired_difference_data(scored, conditions, condition_names, confidence_level)
    speaker_difference_data = _paired_difference_data(
        scored,
        conditions,
        condition_names,
        confidence_level,
        group_key="speaker_id",
        group_order=speakers,
        group_display_names=speaker_names,
    )

    _write_csv(output_root / "mos_condition_plot_data.csv", condition_data)
    _write_csv(output_root / "mos_rating_distribution_plot_data.csv", distribution_data)
    _write_csv(output_root / "mos_listener_condition_plot_data.csv", listener_data)
    _write_csv(output_root / "mos_speaker_condition_plot_data.csv", speaker_data)
    _write_csv(output_root / "mos_condition_difference_plot_data.csv", difference_data)
    _write_csv(output_root / "mos_speaker_difference_plot_data.csv", speaker_difference_data)
    _draw_mos_by_condition(condition_data, output_root / "mos_by_condition.svg", difference_data)
    _draw_rating_distribution(distribution_data, output_root / "mos_rating_distribution.svg", conditions, condition_data)
    _draw_listener_distribution(listener_data, condition_data, output_root / "mos_by_listener_condition.svg", conditions, difference_data)
    _draw_speaker_condition(speaker_data, output_root / "mos_by_speaker_condition.svg", speakers, conditions, speaker_difference_data)
    return [
        "mos_by_condition.svg",
        "mos_rating_distribution.svg",
        "mos_by_listener_condition.svg",
        "mos_by_speaker_condition.svg",
        "mos_condition_plot_data.csv",
        "mos_rating_distribution_plot_data.csv",
        "mos_listener_condition_plot_data.csv",
        "mos_speaker_condition_plot_data.csv",
        "mos_condition_difference_plot_data.csv",
        "mos_speaker_difference_plot_data.csv",
    ]
