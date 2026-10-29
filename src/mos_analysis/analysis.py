from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .summarize import (
    summarize_conditions,
    summarize_mos,
    summarize_rating_distribution,
)
from .visualize import generate_mos_figures


REQUIRED_FIELDS = (
    "participant_id",
    "session_id",
    "list_id",
    "item_id",
    "condition",
    "speaker_id",
    "mos_rating",
)

OUTPUT_GUIDE = """# MOS analysis output guide

This directory contains MOS QC, the validated analysis dataset, descriptive
summaries, and report-ready figures.

1. `03_summaries/mos_by_condition.csv` is the primary condition comparison.
   It reports conventional MOS statistics and listener-level Student t
   confidence intervals. Each listener contributes one condition mean, so
   listeners receive equal weight in the interval calculation.
2. The remaining summary files describe ratings by speaker, gender, listener,
   item, list, and condition combinations. The condition rating-distribution
   table contains the exact counts and proportions for scores 1 through 5.
3. `04_figures/` contains four report-ready SVG figures plus the exact plot-data
   CSV used for each figure.
4. `run_manifest.json` records the input, settings, validation counts, and all
   generated files.

The analysis never edits its input file and does not impute missing ratings.
"""


class MOSAnalysisError(RuntimeError):
    """Raised when an analysis-ready MOS input fails validation."""


def load_config(path: Path) -> dict[str, Any]:
    """Load a MOS JSON configuration."""
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("MOS configuration must contain a JSON object.")
    return config


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise MOSAnalysisError(f"Input CSV has no header: {path}")
        fieldnames = [str(name).strip() for name in reader.fieldnames]
        rows = [
            {str(key).strip(): (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
    return rows, fieldnames


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _parse_rating(value: Any, allowed: set[int]) -> int | None:
    text = str(value or "").strip()
    if not text.isdigit():
        return None
    rating = int(text)
    return rating if rating in allowed else None


def _validate_input(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    missing_fields = [field for field in REQUIRED_FIELDS if field not in fieldnames]
    if missing_fields:
        raise MOSAnalysisError(
            "Input is missing required columns: " + ", ".join(missing_fields)
        )
    if not rows:
        raise MOSAnalysisError("Input contains no MOS responses.")

    rating_scale = [
        int(value) for value in config.get("rating_scale", [1, 2, 3, 4, 5])
    ]
    allowed_ratings = set(rating_scale)
    scored: list[dict[str, Any]] = []
    invalid_rows: list[int] = []
    blank_key_rows: list[int] = []
    for row_number, row in enumerate(rows, start=2):
        if any(not str(row.get(field, "")).strip() for field in REQUIRED_FIELDS[:-1]):
            blank_key_rows.append(row_number)
        parsed = _parse_rating(row.get("mos_rating"), allowed_ratings)
        if parsed is None:
            invalid_rows.append(row_number)
            continue
        enriched: dict[str, Any] = dict(row)
        enriched["mos_rating"] = parsed
        scored.append(enriched)
    if blank_key_rows:
        raise MOSAnalysisError(
            "Required identifiers are blank on input rows: "
            + ", ".join(str(value) for value in blank_key_rows[:20])
        )
    if invalid_rows:
        raise MOSAnalysisError(
            "MOS ratings must be configured whole-number scale values. Invalid rows: "
            + ", ".join(str(value) for value in invalid_rows[:20])
        )

    response_keys = [
        (
            str(row["list_id"]),
            str(row["participant_id"]),
            str(row["session_id"]),
            str(row["item_id"]),
            str(row["condition"]),
        )
        for row in scored
    ]
    duplicates = [key for key, count in Counter(response_keys).items() if count > 1]
    if duplicates:
        raise MOSAnalysisError(
            f"Found {len(duplicates)} duplicate participant-item-condition response keys."
        )

    expected_conditions = [
        str(value) for value in config.get("expected_conditions", [])
    ]
    observed_conditions = sorted({str(row["condition"]) for row in scored})
    if expected_conditions and set(observed_conditions) != set(expected_conditions):
        missing = sorted(set(expected_conditions) - set(observed_conditions))
        unexpected = sorted(set(observed_conditions) - set(expected_conditions))
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise MOSAnalysisError(
            "Condition coverage differs from config (" + "; ".join(details) + ")."
        )

    participant_keys = {
        (str(row["participant_id"]), str(row["session_id"])) for row in scored
    }
    item_metadata: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in scored:
        item_metadata[str(row["item_id"])].add(
            (
                str(row.get("speaker_id", "")),
                str(row.get("gender", "")),
                str(row.get("reference_text", "")),
            )
        )
    inconsistent_items = [
        item for item, values in item_metadata.items() if len(values) > 1
    ]
    if inconsistent_items:
        raise MOSAnalysisError(
            f"Found {len(inconsistent_items)} item IDs with inconsistent speaker, gender, or reference metadata."
        )

    expected_participants = (
        len(config.get("list_ids", []))
        * int(config.get("expected_participants_per_list", 0) or 0)
    )
    if expected_participants and len(participant_keys) != expected_participants:
        raise MOSAnalysisError(
            f"Expected {expected_participants} participant sessions; found {len(participant_keys)}."
        )

    validation = {
        "input_rows": len(rows),
        "valid_score_rows": len(scored),
        "participant_sessions": len(participant_keys),
        "items": len(item_metadata),
        "conditions": observed_conditions,
        "rating_scale": rating_scale,
        "missing_ratings_imputed": 0,
    }
    return scored, validation


def run_mos_analysis(
    input_csv: Path,
    output_root: Path,
    config: dict[str, Any],
    *,
    summary_directory: str = "02_summaries",
    figure_directory: str = "03_figures",
    allow_existing_output: bool = False,
    write_root_files: bool = True,
) -> dict[str, Any]:
    """Validate a prepared MOS file and write descriptive summaries."""
    input_csv = input_csv.resolve()
    output_root = output_root.resolve()
    if not input_csv.is_file():
        raise MOSAnalysisError(f"Input file not found: {input_csv}")
    if not allow_existing_output and (
        output_root == input_csv.parent or output_root in input_csv.parents
    ):
        raise MOSAnalysisError("Output directory must not contain or replace the input file.")
    if (
        not allow_existing_output
        and output_root.exists()
        and any(output_root.iterdir())
    ):
        raise MOSAnalysisError(
            f"Output directory already exists and is not empty: {output_root}"
        )

    rows, input_fields = _read_csv(input_csv)
    scored, validation = _validate_input(rows, input_fields, config)
    settings = config.get("mos_analysis", {})
    confidence_level = float(settings.get("confidence_level", 0.95))
    condition_order = [
        str(value)
        for value in config.get(
            "expected_conditions",
            sorted({str(row["condition"]) for row in scored}),
        )
    ]
    rating_scale = [
        int(value) for value in config.get("rating_scale", [1, 2, 3, 4, 5])
    ]

    summaries: dict[str, list[dict[str, Any]]] = {
        "mos_overall.csv": summarize_mos(scored, []),
        "mos_by_condition.csv": summarize_conditions(
            scored,
            condition_order=condition_order,
            confidence_level=confidence_level,
        ),
        "mos_by_speaker.csv": summarize_mos(scored, ["speaker_id"]),
        "mos_by_gender.csv": summarize_mos(scored, ["gender"]),
        "mos_by_listener.csv": summarize_mos(
            scored, ["participant_id", "session_id", "list_id"]
        ),
        "mos_by_item.csv": summarize_mos(scored, ["item_id", "speaker_id"]),
        "mos_by_list.csv": summarize_mos(scored, ["list_id"]),
        "mos_by_condition_speaker.csv": summarize_mos(
            scored, ["condition", "speaker_id"]
        ),
        "mos_by_condition_gender.csv": summarize_mos(
            scored, ["condition", "gender"]
        ),
        "mos_rating_distribution_by_condition.csv": summarize_rating_distribution(
            scored,
            condition_order=condition_order,
            rating_scale=rating_scale,
        ),
    }
    summary_root = output_root / summary_directory
    for name, summary_rows in summaries.items():
        _write_csv(summary_root / name, summary_rows)

    figures = generate_mos_figures(
        scored,
        output_root / figure_directory,
        condition_order,
        rating_scale,
        confidence_level,
        config.get("visualization", {}),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    if write_root_files:
        (output_root / "OUTPUT_GUIDE.md").write_text(
            OUTPUT_GUIDE, encoding="utf-8"
        )
    generated_files = [
        *[str((summary_root / name).relative_to(output_root)) for name in summaries],
        *[str(Path(figure_directory) / name) for name in figures],
    ]
    if write_root_files:
        generated_files.extend(["OUTPUT_GUIDE.md", "run_manifest.json"])
    manifest = {
        "status": "PASSED",
        "analysis_mode": "analysis_only" if write_root_files else "full_pipeline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "output_root": str(output_root),
        **validation,
        "confidence_interval": {
            "method": "listener_mean_student_t",
            "confidence_level": confidence_level,
            "analysis_unit": "participant_id and session_id condition means",
        },
        "figures": [name for name in figures if name.endswith(".svg")],
        "generated_files": generated_files,
    }
    if write_root_files:
        _write_json(output_root / "run_manifest.json", manifest)
    return manifest
