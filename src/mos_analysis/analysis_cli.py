from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import MOSAnalysisError, load_config
from .pipeline import MOSQCError
from .workflow import run_full_mos_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MOS QC, preprocessing, summaries, and figures."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory containing pilot_mos_list1, pilot_mos_list2, and so on.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New or empty directory for the complete MOS analysis.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/mos_config.json"),
        help="MOS analysis JSON configuration.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_path = args.results_root.resolve()
    output_path = args.output_root.resolve()
    if (
        output_path == results_path
        or results_path in output_path.parents
        or output_path in results_path.parents
    ):
        print(
            "ERROR: --output-root and --results-root must be separate, "
            "non-overlapping directories.",
            file=sys.stderr,
        )
        return 2
    try:
        manifest = run_full_mos_analysis(
            args.results_root,
            args.output_root,
            load_config(args.config),
        )
    except (
        MOSAnalysisError,
        MOSQCError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "MOS analysis passed: "
        f"{manifest['valid_score_rows']} ratings from "
        f"{manifest['participant_sessions']} participant sessions."
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
