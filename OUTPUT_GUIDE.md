# Analysis output guide

This directory contains one complete analysis run.

1. Start with `01_qc/qc_summary.txt`. If QC is blocked, inspect `01_qc/qc_issues.csv` and the relevant detailed audit. Approved playback failures are recorded in `01_qc/technical_exclusion_audit.csv`.
2. Review `02_preprocessed/normalized_transcriptions.csv` to verify text normalization and personal-name exclusions.
3. Use `03_scores/transcription_scores.csv` for response-level WER, CER, substitution, deletion, insertion, and `xxx` results.
4. Use `04_summaries/by_condition.csv` for the primary system comparison. When technical exclusions were applied, compare it with `04_summaries/technical_exclusion_sensitivity.csv`.
5. Review participant feedback separately under `05_questionnaire/`.
6. Use the vector figures in `06_figures/` for reports and presentations. The plot-data CSVs record the exact condition values and the gender-by-condition and speaker-by-condition values with confidence intervals. Gender and speaker comparisons are descriptive for this experiment's four speakers.

`run_manifest.json` records the run status, input location, resolved columns, and row counts.
