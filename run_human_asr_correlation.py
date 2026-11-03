#!/usr/bin/env python3
"""Human-ASR correlation study: statistics and figures from prepared comparison CSVs.

    python run_human_asr_correlation.py --config config/human_asr_correlation_config.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from transcription_analysis.correlation import analyze, load_comparisons
from transcription_analysis.correlation_visualize import generate_correlation_figures


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config_path: Path, output_override: Path | None = None) -> tuple[dict, Path]:
    # 1. Read the config; paths are relative to base_dir, which is relative to the config file.
    cfg = json.loads(config_path.read_text())
    base = (config_path.parent / cfg.get("base_dir", ".")).resolve()
    comparison_dir = (base / cfg["comparison_dir"]).resolve()
    out = Path(output_override).resolve() if output_override else (base / cfg["output_dir"]).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output must be a new/empty directory (existing results are preserved): {out}")
    anchors = cfg.get("anchor_conditions", [])

    # 2. Load the comparison CSVs, 3. calculate statistics, 4. draw figures.
    tables = load_comparisons(comparison_dir)
    info = analyze(tables, out / "02_statistics", cfg.get("bootstrap_samples", 2000),
                   cfg.get("random_seed", 20261102), cfg.get("allow_incomplete", False), anchors)
    figures = generate_correlation_figures(tables, out / "02_statistics", out / "03_figures",
                                           {"anchor_conditions": anchors, **cfg.get("visualization", {})})

    # 5. One record of what was run: settings, counts and exact input/code versions.
    code = [Path(__file__), *(ROOT / "src/transcription_analysis" / name
                              for name in ["correlation.py", "correlation_visualize.py"])]
    manifest = {
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        **info, "figures": figures,
        "config_file": str(config_path), "config_sha256": sha256(config_path),
        "config": {**cfg, "resolved_base_dir": str(base), "resolved_output_dir": str(out)},
        "input_sha256": {str(p): sha256(p) for p in sorted(comparison_dir.glob("comparison_*.csv"))},
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in code},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config/human_asr_correlation_config.json"))
    parser.add_argument("--output", type=Path, help="Override output_dir (relative to the current directory)")
    args = parser.parse_args()
    try:
        result, out = run(args.config.resolve(), args.output)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Complete: {result['n_asr_systems']} ASRs, {result['n_expected_recordings']} recordings -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
