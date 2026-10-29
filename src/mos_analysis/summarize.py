from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median, stdev
from typing import Any


def _number(value: Any) -> float | None:
    """Return a finite number or None."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _quantile(values: list[float], probability: float) -> float:
    """Calculate a linearly interpolated sample quantile."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile from an empty list.")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _continued_beta_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the incomplete beta function."""
    maximum_iterations = 200
    epsilon = 3.0e-14
    minimum = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < minimum:
        d = minimum
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + even) * (a + even)
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        result *= d * c

        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + even) * (qap + even)
        )
        d = 1.0 + coefficient * d
        if abs(d) < minimum:
            d = minimum
        c = 1.0 + coefficient / c
        if abs(c) < minimum:
            c = minimum
        d = 1.0 / d
        change = d * c
        result *= change
        if abs(change - 1.0) < epsilon:
            return result
    raise ArithmeticError("Incomplete beta calculation did not converge.")


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Return the regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    scale = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return scale * _continued_beta_fraction(a, b, x) / a
    return 1.0 - scale * _continued_beta_fraction(b, a, 1.0 - x) / b


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    """Return the cumulative probability of Student's t distribution."""
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be at least 1.")
    if value == 0.0:
        return 0.5
    ratio = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        ratio, degrees_of_freedom / 2.0, 0.5
    )
    return 1.0 - tail if value > 0.0 else tail


def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Invert Student's t CDF by deterministic bisection."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1.")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -_student_t_quantile(1.0 - probability, degrees_of_freedom)
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _student_t_cdf(midpoint, degrees_of_freedom) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def listener_mean_t_ci(
    rows: list[dict[str, Any]],
    *,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    """Calculate a t-based CI across equal-weighted listener condition means."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    listeners: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        rating = _number(row.get("mos_rating"))
        if rating is not None:
            listeners[
                (
                    str(row.get("participant_id", "")),
                    str(row.get("session_id", "")),
                )
            ].append(rating)
    listener_means = [mean(values) for values in listeners.values() if values]
    if len(listener_means) < 2:
        raise ValueError("At least two listeners are required for a t interval.")
    point = mean(listener_means)
    listener_sd = stdev(listener_means)
    standard_error = listener_sd / math.sqrt(len(listener_means))
    degrees_of_freedom = len(listener_means) - 1
    critical_value = _student_t_quantile(
        0.5 + confidence_level / 2.0, degrees_of_freedom
    )
    margin = critical_value * standard_error
    return {
        "mean_mos": point,
        "mos_ci_low": point - margin,
        "mos_ci_high": point + margin,
        "listener_mean_sd": listener_sd,
        "listener_mean_standard_error": standard_error,
        "n_listener_means": len(listener_means),
        "degrees_of_freedom": degrees_of_freedom,
        "t_critical_value": critical_value,
    }


def _group_rows(
    rows: list[dict[str, Any]], group_keys: list[str]
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in group_keys)].append(row)
    return groups


def summarize_mos(
    rows: list[dict[str, Any]], group_keys: list[str]
) -> list[dict[str, Any]]:
    """Calculate conventional descriptive MOS statistics for each group."""
    output: list[dict[str, Any]] = []
    for values, group in sorted(_group_rows(rows, group_keys).items()):
        ratings = [float(row["mos_rating"]) for row in group]
        plays = [
            value
            for value in (_number(row.get("plays_used")) for row in group)
            if value is not None
        ]
        response_times = [
            value
            for value in (
                _number(row.get("response_time_sec")) for row in group
            )
            if value is not None
        ]
        result: dict[str, Any] = {
            key: value for key, value in zip(group_keys, values)
        }
        result.update(
            {
                "n_ratings": len(ratings),
                "n_participants": len(
                    {
                        (
                            str(row.get("participant_id", "")),
                            str(row.get("session_id", "")),
                        )
                        for row in group
                    }
                ),
                "n_items": len({str(row.get("item_id", "")) for row in group}),
                "n_speakers": len(
                    {str(row.get("speaker_id", "")) for row in group}
                ),
                "mean_mos": mean(ratings),
                "sd_mos": stdev(ratings) if len(ratings) >= 2 else 0.0,
                "median_mos": median(ratings),
                "q1_mos": _quantile(ratings, 0.25),
                "q3_mos": _quantile(ratings, 0.75),
                "minimum_mos": min(ratings),
                "maximum_mos": max(ratings),
                "mean_plays_used": mean(plays) if plays else "",
                "median_plays_used": median(plays) if plays else "",
                "mean_response_time_sec": (
                    mean(response_times) if response_times else ""
                ),
                "median_response_time_sec": (
                    median(response_times) if response_times else ""
                ),
            }
        )
        output.append(result)
    return output


def summarize_conditions(
    rows: list[dict[str, Any]],
    *,
    condition_order: list[str],
    confidence_level: float,
) -> list[dict[str, Any]]:
    """Summarize conditions with listener-level t confidence intervals."""
    grouped = _group_rows(rows, ["condition"])
    output: list[dict[str, Any]] = []
    observed_conditions = sorted(key[0] for key in grouped)
    ordered_conditions = [
        condition for condition in condition_order if (condition,) in grouped
    ]
    ordered_conditions.extend(
        condition
        for condition in observed_conditions
        if condition not in ordered_conditions
    )
    for condition in ordered_conditions:
        group = grouped[(condition,)]
        summary = summarize_mos(group, ["condition"])[0]
        response_weighted_mean = summary["mean_mos"]
        interval = listener_mean_t_ci(group, confidence_level=confidence_level)
        summary.update(
            {
                "response_weighted_mean_mos": response_weighted_mean,
                **interval,
                "confidence_level": confidence_level,
                "ci_method": "listener_mean_student_t",
            }
        )
        output.append(summary)
    return output


def summarize_rating_distribution(
    rows: list[dict[str, Any]],
    *,
    condition_order: list[str],
    rating_scale: list[int],
) -> list[dict[str, Any]]:
    """Return counts and proportions for every rating within each condition."""
    grouped = _group_rows(rows, ["condition"])
    output: list[dict[str, Any]] = []
    for condition in condition_order:
        group = grouped.get((condition,), [])
        if not group:
            continue
        counts = Counter(int(row["mos_rating"]) for row in group)
        for rating in rating_scale:
            output.append(
                {
                    "condition": condition,
                    "mos_rating": rating,
                    "rating_count": counts[rating],
                    "rating_proportion": counts[rating] / len(group),
                    "n_ratings": len(group),
                }
            )
    return output
