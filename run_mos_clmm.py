#!/usr/bin/env python3
"""Run the repository's frequentist ordinal MOS analysis in R."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REFERENCE = "mantis_v5_w2s_baseline_E2E_apt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a cumulative-link mixed model to the cleaned MOS ratings using R."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reference-condition",
        default=DEFAULT_REFERENCE,
        help="Condition used as the reference in focused pairwise tests.",
    )
    return parser


def build_command(
    rscript: str,
    analysis_script: Path,
    input_path: Path,
    output_root: Path,
    reference_condition: str,
) -> list[str]:
    """Build the explicit Rscript command used by the runner."""
    return [
        rscript,
        str(analysis_script),
        "--input",
        str(input_path),
        "--output-root",
        str(output_root),
        "--reference-condition",
        reference_condition,
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parent
    analysis_script = repository_root / "scripts" / "mos_ordinal_clmm.R"
    rscript = shutil.which("Rscript")

    if rscript is None:
        print(
            "ERROR: Rscript was not found. Install command-line R, then install "
            "the R packages 'ordinal' and 'emmeans'. RStudio is optional.",
            file=sys.stderr,
        )
        return 2
    if not analysis_script.is_file():
        print(f"ERROR: Analysis script not found: {analysis_script}", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"ERROR: MOS input not found: {args.input}", file=sys.stderr)
        return 2
    if args.output_root.exists() and any(args.output_root.iterdir()):
        print(
            f"ERROR: Output directory is not empty: {args.output_root}",
            file=sys.stderr,
        )
        return 2

    command = build_command(
        rscript,
        analysis_script,
        args.input,
        args.output_root,
        args.reference_condition,
    )
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:
        print(f"ERROR: Could not start Rscript: {exc}", file=sys.stderr)
        return 2
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

