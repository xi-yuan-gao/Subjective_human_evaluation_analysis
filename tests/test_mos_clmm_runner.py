from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run_mos_clmm import DEFAULT_REFERENCE, build_command, build_parser


class MOSCLMMRunnerTests(unittest.TestCase):
    def test_parser_uses_e2e_reference(self) -> None:
        args = build_parser().parse_args(
            ["--input", "ratings.csv", "--output-root", "clmm_output"]
        )
        self.assertEqual(args.reference_condition, DEFAULT_REFERENCE)

    def test_command_passes_paths_and_reference_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = build_command(
                "/usr/local/bin/Rscript",
                root / "analysis.R",
                root / "ratings.csv",
                root / "results",
                "e2e",
            )
        self.assertEqual(command[0], "/usr/local/bin/Rscript")
        self.assertEqual(command[2:4], ["--input", str(root / "ratings.csv")])
        self.assertEqual(command[-2:], ["--reference-condition", "e2e"])


if __name__ == "__main__":
    unittest.main()

