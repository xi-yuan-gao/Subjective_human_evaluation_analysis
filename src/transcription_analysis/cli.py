from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .pipeline import QCError, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QC and analyze a five-list transcription experiment.")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Directory containing pilot_transcription_experiment_list1 through list5.")
    parser.add_argument("--output-root", type=Path, default=Path("analysis_output"), help="New directory for derived reports. Raw results are never modified.")
    parser.add_argument("--config", type=Path, default=Path("config/analysis_config.json"), help="JSON configuration file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_path = args.results_root.resolve()
    output_path = args.output_root.resolve()
    if output_path == results_path or results_path in output_path.parents or output_path in results_path.parents:
        print("ERROR: --output-root and --results-root must be separate, non-overlapping directories.", file=sys.stderr)
        return 2
    try:
        manifest = run_pipeline(args.results_root, args.output_root, load_config(args.config))
    except (QCError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(manifest["qc_summary"])
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
