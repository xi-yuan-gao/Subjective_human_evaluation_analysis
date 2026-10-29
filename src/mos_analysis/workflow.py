from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analysis import run_mos_analysis
from .pipeline import run_mos_pipeline


OUTPUT_GUIDE = """# MOS analysis output guide

This directory contains one complete MOS descriptive-analysis run.

1. Start with `01_qc/qc_summary.txt`. If QC is blocked, inspect
   `01_qc/qc_issues.csv` and the named audit files.
2. `02_preprocessed/mos_analysis_ready.csv` contains the exact eligible,
   non-attention ratings used by all downstream outputs. Raw result files are
   never edited.
3. `03_summaries/mos_by_condition.csv` is the primary descriptive condition
   comparison. Supporting summaries cover speakers, gender, listeners, items,
   lists, and rating distributions.
4. `04_figures/` contains report-ready SVG figures and the exact plot-data CSV
   used for each figure.
5. `run_manifest.json` records QC decisions, validation counts,
   confidence-interval settings, and generated files.

The separate `run_mos_clmm.py` command uses the analysis-ready CSV for ordinal
mixed-effects inference.
"""


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def run_full_mos_analysis(
    results_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run MOS QC, preprocessing, descriptive summaries, and figures."""
    preprocessing_manifest = run_mos_pipeline(results_root, output_root, config)
    analysis_ready = output_root / "02_preprocessed" / "mos_analysis_ready.csv"
    descriptive_manifest = run_mos_analysis(
        analysis_ready,
        output_root,
        config,
        summary_directory="03_summaries",
        figure_directory="04_figures",
        allow_existing_output=True,
        write_root_files=False,
    )

    combined_manifest = dict(preprocessing_manifest)
    combined_manifest.update(
        {
            "analysis_mode": "full_pipeline",
            "analysis_ready_csv": str(analysis_ready.resolve()),
            "valid_score_rows": descriptive_manifest["valid_score_rows"],
            "participant_sessions": descriptive_manifest["participant_sessions"],
            "items": descriptive_manifest["items"],
            "conditions": descriptive_manifest["conditions"],
            "rating_scale": descriptive_manifest["rating_scale"],
            "missing_ratings_imputed": descriptive_manifest[
                "missing_ratings_imputed"
            ],
            "confidence_interval": descriptive_manifest["confidence_interval"],
            "figures": descriptive_manifest["figures"],
        }
    )
    combined_manifest["generated_files"] = sorted(
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    ) + ["run_manifest.json"]
    (output_root / "OUTPUT_GUIDE.md").write_text(OUTPUT_GUIDE, encoding="utf-8")
    _write_json(output_root / "run_manifest.json", combined_manifest)
    return combined_manifest
