# MOS analysis output guide

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
5. `run_manifest.json` records QC decisions, item-ID harmonization, validation
   counts, confidence-interval settings, and generated files.

The separate `run_mos_clmm.py` command uses the analysis-ready CSV for ordinal
mixed-effects inference.
