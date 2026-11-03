"""SVG figures and plot-data CSVs for the human-ASR correlation study.

Called by ``run_human_asr_correlation.py`` using the ``visualization`` section of
``config/human_asr_correlation_config.json`` (the same pattern as ``visualize.py``
for the transcription analysis). It only reads finished results, so figures can
also be redrawn for an existing run without recomputing statistics:

    PYTHONPATH=src python -m transcription_analysis.correlation_visualize \
        --run-dir analysis_output/human_asr_correlation
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

from .io_utils import write_csv
from .visualize import (
    BACKGROUND, BLUE, GRAY, GREEN, LIGHT_GRAY, ORANGE, PINK, PURPLE, TEXT,
    _escape, _quantile, _text,
)

CONDITION_COLORS = [BLUE, ORANGE, GREEN, PURPLE, PINK]
DEFAULT_ANCHORS = ["spoken_audio", "whisper_audio"]


# --------------------------------------------------------------------------- helpers
def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _svg(width: float, height: float, title: str, description: str, body: list[str]) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{_escape(title)}</title>',
        f'<desc id="desc">{_escape(description)}</desc>',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="{BACKGROUND}"/>',
        *body, "</svg>", "",
    ])


def _axis_max(values: list[float], minimum: float = 0.2) -> float:
    padded = max([minimum, *values]) * 1.05
    step = 0.1 if padded <= 1.0 else 0.25
    return math.ceil(padded / step) * step


def _fit(x: list[float], y: list[float]) -> tuple[float, float] | None:
    """Least-squares line y = intercept + slope * x; None if x is constant."""
    if len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    sxx = math.fsum((v - mx) ** 2 for v in x)
    if sxx == 0:
        return None
    slope = math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx
    return slope, my - slope * mx


class _Style:
    """Condition/ASR order, display names and colours taken from the config."""

    def __init__(self, tables: dict[str, list[dict]], settings: dict[str, Any]):
        rows = next(iter(tables.values()))
        present = {r["condition"] for r in rows}
        order = [c for c in settings.get("condition_order", []) if c in present]
        self.conditions = order + sorted(present - set(order))
        self.condition_names = settings.get("condition_display_names", {})
        self.anchors = [c for c in settings.get("anchor_conditions", DEFAULT_ANCHORS) if c in present]
        asr_order = [a for a in settings.get("asr_order", []) if a in tables]
        self.asrs = asr_order + sorted(set(tables) - set(asr_order))
        self.asr_names = settings.get("asr_display_names", {})
        self.colors = {c: CONDITION_COLORS[i % len(CONDITION_COLORS)] for i, c in enumerate(self.conditions)}

    def cname(self, condition: str) -> str:
        return self.condition_names.get(condition, condition)

    def aname(self, asr: str) -> str:
        return self.asr_names.get(asr, asr)

    def marker(self, condition: str, cx: float, cy: float, r: float, opacity: float = 0.8) -> str:
        color = self.colors[condition]
        if condition in self.anchors:  # anchors: hollow markers
            return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{BACKGROUND}" '
                    f'stroke="{color}" stroke-width="1.6" stroke-opacity="{opacity}"/>')
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{color}" fill-opacity="{opacity}"/>'

    def legend(self, x: float, y: float, conditions: list[str]) -> list[str]:
        body = []
        for condition in conditions:
            label = self.cname(condition) + (" (anchor)" if condition in self.anchors else "")
            body.append(self.marker(condition, x + 5, y - 4, 5, 1))
            body.append(_text(x + 15, y, label, size=12))
            x += 34 + 6.3 * len(label)
        return body


def _panel_axes(body: list[str], x0: float, y0: float, size: float, maximum: float,
                x_label: str, y_label: str, title: str, clip_id: str) -> None:
    """Square axes from 0 to maximum with grid, ticks, labels and the y = x diagonal."""
    body.append(_text(x0, y0 - 12, title, size=14, weight=500))
    step = 0.1 if maximum <= 0.5 else 0.25
    for k in range(round(maximum / step) + 1):
        value = k * step
        offset = size * value / maximum
        body.append(f'<line x1="{x0 + offset:.1f}" y1="{y0}" x2="{x0 + offset:.1f}" y2="{y0 + size}" stroke="{LIGHT_GRAY}"/>')
        body.append(f'<line x1="{x0}" y1="{y0 + size - offset:.1f}" x2="{x0 + size}" y2="{y0 + size - offset:.1f}" stroke="{LIGHT_GRAY}"/>')
        body.append(_text(x0 + offset, y0 + size + 15, f"{value:.0%}", anchor="middle", size=10, color=GRAY))
        body.append(_text(x0 - 6, y0 + size - offset + 3, f"{value:.0%}", anchor="end", size=10, color=GRAY))
    body.append(f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" fill="none" stroke="{GRAY}" stroke-width="0.8"/>')
    body.append(f'<clipPath id="{clip_id}"><rect x="{x0}" y="{y0}" width="{size}" height="{size}"/></clipPath>')
    body.append(f'<line x1="{x0}" y1="{y0 + size}" x2="{x0 + size}" y2="{y0}" stroke="{GRAY}" '
                f'stroke-width="1.2" stroke-dasharray="5 4" clip-path="url(#{clip_id})"/>')
    body.append(_text(x0 + size / 2, y0 + size + 32, x_label, anchor="middle", size=11, color=GRAY))
    cx, cy = x0 - 38, y0 + size / 2
    body.append(f'<text x="{cx}" y="{cy}" transform="rotate(-90 {cx} {cy})" text-anchor="middle" '
                f'font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{GRAY}">{_escape(y_label)}</text>')


def _grid(n: int, columns: int = 4) -> tuple[int, int]:
    return columns, math.ceil(n / columns)


# --------------------------------------------------------------------------- scatter
def _scatter_data(tables, style: _Style, conditions: list[str]) -> list[dict]:
    data = []
    for asr in style.asrs:
        for condition in conditions:
            group = [r for r in tables[asr] if r["condition"] == condition and r["asr_wer"] is not None]
            x = [r["human_mean_wer"] for r in group]
            y = [r["asr_wer"] for r in group]
            line = _fit(x, y)
            for row in group:
                data.append({
                    "asr_system": asr, "asr_display_name": style.aname(asr),
                    "condition": condition, "condition_display_name": style.cname(condition),
                    "is_anchor": condition in style.anchors,
                    "utterance_id": row["utterance_id"],
                    "human_mean_wer": row["human_mean_wer"], "asr_wer": row["asr_wer"],
                    "condition_fit_slope": line[0] if line else None,
                    "condition_fit_intercept": line[1] if line else None,
                })
    return data


def _draw_scatter(data: list[dict], style: _Style, conditions: list[str], path: Path, title: str) -> None:
    columns, rows = _grid(len(style.asrs))
    size, panel_w, panel_h, left, top = 240, 330, 330, 30, 132
    width, height = left + columns * panel_w, top + rows * panel_h
    maximum = _axis_max([v for d in data for v in (d["human_mean_wer"], d["asr_wer"])])
    body = [
        _text(left, 34, title, size=20, weight=500),
        _text(left, 58, "Each point is one recording (one utterance in one condition); x = mean WER of its listeners, y = ASR WER.", size=12, color=GRAY),
        _text(left, 76, "Dashed diagonal: ASR WER = human WER. Points above it: the ASR counts more errors than listeners; below it: fewer.", size=12, color=GRAY),
        _text(left, 94, "Coloured lines: least-squares fit within each condition", size=12, color=GRAY),
        *style.legend(left, 120, conditions),
    ]
    for k, asr in enumerate(style.asrs):
        x0 = left + 55 + (k % columns) * panel_w
        y0 = top + 30 + (k // columns) * panel_h
        clip = f"clip{k}"
        _panel_axes(body, x0, y0, size, maximum, "Human mean WER", "ASR WER", style.aname(asr), clip)
        sx = lambda v: x0 + size * v / maximum
        sy = lambda v: y0 + size - size * v / maximum
        points = [d for d in data if d["asr_system"] == asr]
        for d in points:
            body.append(style.marker(d["condition"], sx(d["human_mean_wer"]), sy(d["asr_wer"]), 3.2, 0.7))
        for condition in conditions:
            group = [d for d in points if d["condition"] == condition]
            if not group or group[0]["condition_fit_slope"] is None:
                continue
            slope, intercept = group[0]["condition_fit_slope"], group[0]["condition_fit_intercept"]
            lo = min(d["human_mean_wer"] for d in group)
            hi = max(d["human_mean_wer"] for d in group)
            body.append(f'<line x1="{sx(lo):.1f}" y1="{sy(intercept + slope * lo):.1f}" x2="{sx(hi):.1f}" '
                        f'y2="{sy(intercept + slope * hi):.1f}" stroke="{style.colors[condition]}" '
                        f'stroke-width="2" clip-path="url(#{clip})"/>')
    path.write_text(_svg(width, height, title, "Human versus ASR WER per recording, one panel per ASR.", body), encoding="utf-8")


# --------------------------------------------------------------------------- disagreement
def _disagreement_data(tables, style: _Style) -> list[dict]:
    data = []
    for asr in style.asrs:
        for condition in style.conditions:
            for row in tables[asr]:
                if row["condition"] == condition and row["asr_wer"] is not None:
                    data.append({
                        "asr_system": asr, "asr_display_name": style.aname(asr),
                        "condition": condition, "condition_display_name": style.cname(condition),
                        "is_anchor": condition in style.anchors,
                        "utterance_id": row["utterance_id"],
                        "human_mean_wer": row["human_mean_wer"], "asr_wer": row["asr_wer"],
                        "delta_wer": row["asr_wer"] - row["human_mean_wer"],
                    })
    return data


def _draw_disagreement(data: list[dict], style: _Style, path: Path) -> None:
    label_w, panel_w, gap, top, row_h, bottom = 150, 230, 24, 140, 46, 62
    n_rows = len(style.asrs)
    width = 30 + label_w + len(style.conditions) * (panel_w + gap)
    height = top + n_rows * row_h + bottom
    limit = _axis_max([abs(d["delta_wer"]) for d in data], minimum=0.2)
    title = "ASR − human WER disagreement by condition"
    body = [
        _text(30, 34, title, size=20, weight=500),
        _text(30, 58, "delta WER = ASR WER − human mean WER for each recording. Dashed line: zero disagreement.", size=12, color=GRAY),
        _text(30, 76, "Right of the line: ASR counts more errors than listeners; left: listeners find more errors than the ASR.", size=12, color=GRAY),
        _text(30, 94, "Dots are recordings; boxes show the middle 50% and median; open circles show the mean.", size=12, color=GRAY),
    ]
    for j, condition in enumerate(style.conditions):
        x0 = 30 + label_w + j * (panel_w + gap)
        scale = lambda v, x0=x0: x0 + panel_w * (v + limit) / (2 * limit)
        anchor = condition in style.anchors
        body.append(_text(x0 + panel_w / 2, top - 16, style.cname(condition) + (" (anchor)" if anchor else ""),
                          anchor="middle", size=13, weight=500, color=GRAY if anchor else TEXT))
        for k in range(5):
            value = -limit + 2 * limit * k / 4
            body.append(f'<line x1="{scale(value):.1f}" y1="{top}" x2="{scale(value):.1f}" y2="{top + n_rows * row_h}" stroke="{LIGHT_GRAY}"/>')
            body.append(_text(scale(value), top + n_rows * row_h + 16, f"{value * 100:+.0f}" if value else "0", anchor="middle", size=10, color=GRAY))
        body.append(f'<line x1="{scale(0):.1f}" y1="{top - 4}" x2="{scale(0):.1f}" y2="{top + n_rows * row_h}" stroke="{GRAY}" stroke-width="1.2" stroke-dasharray="5 4"/>')
        body.append(_text(x0 + panel_w / 2, top + n_rows * row_h + 36, "delta WER (percentage points)", anchor="middle", size=11, color=GRAY))
        color = style.colors[condition]
        for i, asr in enumerate(style.asrs):
            cy = top + row_h * (i + 0.5)
            if j == 0:
                body.append(_text(30, cy + 4, style.aname(asr), size=12))
            values = sorted(d["delta_wer"] for d in data if d["asr_system"] == asr and d["condition"] == condition)
            if not values:
                continue
            for n, value in enumerate(values):
                body.append(f'<circle cx="{scale(value):.1f}" cy="{cy + (-8, -4, 0, 4, 8)[n % 5]:.1f}" r="2" fill="{GRAY}" fill-opacity="0.35"/>')
            q1, q2, q3 = (_quantile(values, p) for p in (0.25, 0.5, 0.75))
            body.append(f'<rect x="{scale(q1):.1f}" y="{cy - 11:.1f}" width="{max(1.0, scale(q3) - scale(q1)):.1f}" height="22" '
                        f'fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="1.6"/>')
            body.append(f'<line x1="{scale(q2):.1f}" y1="{cy - 13:.1f}" x2="{scale(q2):.1f}" y2="{cy + 13:.1f}" stroke="{color}" stroke-width="2.5"/>')
            body.append(f'<circle cx="{scale(mean(values)):.1f}" cy="{cy:.1f}" r="4.5" fill="{BACKGROUND}" stroke="{color}" stroke-width="2"/>')
    path.write_text(_svg(width, height, title, "Distribution of ASR minus human mean WER per ASR and condition.", body), encoding="utf-8")


# --------------------------------------------------------------------------- model level
def _model_data(stats_dir: Path, style: _Style) -> list[dict]:
    scores = _read_csv(stats_dir / "model_level/01_condition_means_wer.csv")
    agreement = {r["asr_system"]: r for r in _read_csv(stats_dir / "model_level/02_correlations.csv") if r["metric"] == "wer"}
    data = []
    for asr in style.asrs:
        summary = agreement[asr]
        correct = int(summary["strict_pairs_correct"])
        compared = correct + int(summary["strict_pairs_reversed"]) + int(summary["asr_tied_strict_pairs"])
        by_condition = {r["condition"]: r for r in scores if r["asr_system"] == asr}
        for condition in style.conditions:
            row = by_condition[condition]
            data.append({
                "asr_system": asr, "asr_display_name": style.aname(asr),
                "condition": condition, "condition_display_name": style.cname(condition),
                "is_anchor": condition in style.anchors,
                "human_mean_wer": float(row["human_mean_wer"]), "asr_mean_wer": float(row["asr_mean_wer"]),
                "human_rank": float(row["human_rank"]), "asr_rank": float(row["asr_rank"]),
                "model_level_spearman_rho": _float(summary["spearman_rho"]),
                "model_level_pearson_r": _float(summary["pearson_r"]),
                "pairs_correct": correct, "pairs_compared": compared,
            })
    return data


def _draw_model_level(data: list[dict], style: _Style, path: Path) -> None:
    columns, rows = _grid(len(style.asrs))
    size, panel_w, panel_h, left, top = 240, 330, 330, 30, 132
    width, height = left + columns * panel_w, top + rows * panel_h
    maximum = _axis_max([v for d in data for v in (d["human_mean_wer"], d["asr_mean_wer"])], minimum=0.1)
    title = "Model-level comparison: condition mean WER"
    fmt = lambda v: "undefined" if v is None else f"{v:.2f}"
    body = [
        _text(left, 34, title, size=20, weight=500),
        _text(left, 58, "Each point is one condition: x = mean of its 30 recordings' human mean WER, y = mean ASR WER.", size=12, color=GRAY),
        _text(left, 76, "Dashed diagonal: equal human and ASR mean WER. Ranking agreement uses the five condition means only and is descriptive.", size=12, color=GRAY),
        _text(left, 94, "Pairs = condition pairs ordered the same way by ASR and humans (out of pairs not tied for humans).", size=12, color=GRAY),
        *style.legend(left, 120, style.conditions),
    ]
    for k, asr in enumerate(style.asrs):
        x0 = left + 55 + (k % columns) * panel_w
        y0 = top + 30 + (k // columns) * panel_h
        _panel_axes(body, x0, y0, size, maximum, "Human mean WER", "ASR mean WER", style.aname(asr), f"mclip{k}")
        points = [d for d in data if d["asr_system"] == asr]
        for d in points:
            body.append(style.marker(d["condition"], x0 + size * d["human_mean_wer"] / maximum,
                                     y0 + size - size * d["asr_mean_wer"] / maximum, 6, 0.9))
        first = points[0]
        for n, line in enumerate([f"Spearman ρ = {fmt(first['model_level_spearman_rho'])}",
                                  f"Pearson r = {fmt(first['model_level_pearson_r'])}",
                                  f"Pairs = {first['pairs_correct']}/{first['pairs_compared']}"]):
            body.append(_text(x0 + 8, y0 + 18 + 15 * n, line, size=11, color=TEXT))
    path.write_text(_svg(width, height, title, "Condition-level human versus ASR mean WER per ASR.", body), encoding="utf-8")


# --------------------------------------------------------------------------- entry points
def generate_correlation_figures(tables: dict[str, list[dict]], stats_dir: Path, output_dir: Path,
                                 settings: dict[str, Any]) -> list[str]:
    """Write the WER figures and the CSV data behind each one; return figure names."""
    if not settings.get("enabled", True):
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    style = _Style(tables, settings)
    generated = [c for c in style.conditions if c not in style.anchors]
    figures = []
    for name, conditions, title in [
        ("human_asr_wer_scatter_all_conditions", style.conditions, "Human vs ASR WER: all conditions"),
        ("human_asr_wer_scatter_generated_conditions", generated, "Human vs ASR WER: generated conditions only"),
    ]:
        data = _scatter_data(tables, style, conditions)
        write_csv(output_dir / f"{name}_plot_data.csv", data)
        _draw_scatter(data, style, conditions, output_dir / f"{name}.svg", title)
        figures.append(f"{name}.svg")
    data = _disagreement_data(tables, style)
    write_csv(output_dir / "wer_disagreement_plot_data.csv", data)
    _draw_disagreement(data, style, output_dir / "wer_disagreement.svg")
    figures.append("wer_disagreement.svg")
    data = _model_data(stats_dir, style)
    write_csv(output_dir / "model_level_wer_comparison_plot_data.csv", data)
    _draw_model_level(data, style, output_dir / "model_level_wer_comparison.svg")
    figures.append("model_level_wer_comparison.svg")
    return figures


def main() -> int:
    from .correlation import load_comparisons

    parser = argparse.ArgumentParser(description="Redraw figures for a finished human-ASR correlation run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing run folder containing run_manifest.json")
    parser.add_argument("--output", type=Path, help="Figure folder (default: <run-dir>/03_figures)")
    args = parser.parse_args()
    cfg = json.loads((args.run_dir / "run_manifest.json").read_text())["config"]
    base = Path(cfg["resolved_base_dir"])
    tables = load_comparisons(base / cfg["comparison_dir"])
    out = args.output or args.run_dir / "03_figures"
    settings = {"anchor_conditions": cfg.get("anchor_conditions", DEFAULT_ANCHORS), **cfg.get("visualization", {})}
    figures = generate_correlation_figures(tables, args.run_dir / "02_statistics", out, settings)
    print(f"Wrote {len(figures)} figures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
