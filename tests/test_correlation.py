from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_human_asr_correlation import run
from transcription_analysis.correlation import (
    analyze, association, average_ranks, cluster_resamples, load_comparisons,
    pearson, ranking_summary, spearman, within_condition_association, write_table,
)


class CorrelationMathTests(unittest.TestCase):
    def test_coefficients_and_average_ties(self):
        self.assertAlmostEqual(pearson([0, 1, 2, 3], [1, 3, 5, 7]), 1)
        self.assertAlmostEqual(pearson([0, 1, 2, 3], [7, 5, 3, 1]), -1)
        self.assertEqual(average_ranks([0, 0, .5, 1, 1]), [1.5, 1.5, 3, 4.5, 4.5])
        self.assertAlmostEqual(spearman([1, 1, 2, 3], [1, 2, 2, 3]), 5 / 6)

    def test_float_noise_counts_as_a_tie(self):
        # Both are 1/6 in the real comparison data, stored as two different floats.
        a, b = 0.16666666666666666, 0.16666666666666669
        self.assertNotEqual(a, b)
        self.assertEqual(average_ranks([a, b, 0.9]), [1.5, 1.5, 3])
        self.assertAlmostEqual(spearman([a, b, 0.9, 0.0], [0.2, 0.1, 0.3, 0.0]),
                               spearman([b, b, 0.9, 0.0], [0.2, 0.1, 0.3, 0.0]))
        details, result = ranking_summary(
            [{"condition": c, "human_mean_wer": h, "asr_wer": v}
             for c, h, v in [("a", a, 0.1), ("b", b, 0.2), ("c", 0.9, 0.3)]], "system", "wer")
        self.assertEqual(result["human_tied_pairs"], 1)

    def test_constant_and_too_small_are_undefined(self):
        result = association([0, 0, 0], [0, 1, 2])
        self.assertIsNone(result["pearson_r"])
        self.assertIsNone(result["spearman_rho"])
        self.assertEqual(result["status"], "constant_human_scores")
        self.assertEqual(association([0, 1], [0, 1])["status"], "insufficient_pairs")
        self.assertIsNone(association([0, 1, 2], [0, 1, 2])["pearson_ci_low"])

    def test_cluster_bootstrap_preserves_conditions_and_seed(self):
        rows = [{"utterance_id": u, "condition": c}
                for u in ["u1", "u2", "u3", "u4"] for c in ["a", "b"]]
        samples = cluster_resamples(rows, 20, 42)
        self.assertEqual(samples, cluster_resamples(rows, 20, 42))
        for indices in samples:
            counts = Counter((rows[i]["utterance_id"], rows[i]["condition"]) for i in indices)
            for utterance in ["u1", "u2", "u3", "u4"]:
                self.assertEqual(counts[utterance, "a"], counts[utterance, "b"])
        result = association(list(range(len(rows))), list(range(len(rows))), samples)
        self.assertAlmostEqual(result["pearson_ci_low"], 1)
        self.assertAlmostEqual(result["spearman_ci_high"], 1)

    def test_condition_ranking_ties_and_pair_reversals(self):
        rows = [{"condition": c, "human_mean_wer": human, "asr_wer": asr}
                for c, human, asr in [("a", 0, 0), ("b", 1, 1), ("c", 2, 1)]]
        details, result = ranking_summary(rows, "system", "wer")
        self.assertEqual([row["asr_rank"] for row in details], [1, 2.5, 2.5])
        self.assertEqual(result["strict_pairs_correct"], 2)
        self.assertEqual(result["asr_tied_strict_pairs"], 1)
        self.assertAlmostEqual(result["strict_pair_agreement"], 2 / 3)


class WithinConditionTests(unittest.TestCase):
    @staticmethod
    def rows(spec):
        return [{"utterance_id": f"u{i}", "condition": c, "human_mean_wer": h, "asr_wer": a}
                for c, values in spec.items() for i, (h, a) in enumerate(values)]

    def test_condition_differences_cannot_create_correlation(self):
        # Within each condition human and ASR are exactly opposed; only the
        # condition means agree. Raw pooled r is high, within-condition r is -1.
        rows = self.rows({"easy": [(0.0, 0.2), (0.1, 0.1), (0.2, 0.0)],
                          "hard": [(0.8, 1.0), (0.9, 0.9), (1.0, 0.8)]})
        raw = pearson([r["human_mean_wer"] for r in rows], [r["asr_wer"] for r in rows])
        self.assertGreater(raw, 0.9)
        result = within_condition_association(rows, "wer")
        self.assertAlmostEqual(result["pearson_r"], -1)
        self.assertAlmostEqual(result["spearman_rho"], -1)
        self.assertEqual(result["n_conditions"], 2)

    def test_matches_manually_centred_pearson(self):
        rows = self.rows({"a": [(0.1, 0.3), (0.4, 0.2), (0.2, 0.5), (0.6, 0.7)],
                          "b": [(0.5, 0.4), (0.9, 0.8), (0.7, 0.9)]})
        x, y = [], []
        for c in ["a", "b"]:
            g = [r for r in rows if r["condition"] == c]
            hm = sum(r["human_mean_wer"] for r in g) / len(g)
            am = sum(r["asr_wer"] for r in g) / len(g)
            x += [r["human_mean_wer"] - hm for r in g]
            y += [r["asr_wer"] - am for r in g]
        self.assertAlmostEqual(within_condition_association(rows, "wer")["pearson_r"], pearson(x, y))

    def test_constant_condition_contributes_nothing_but_stays_defined(self):
        varied = [(0.1, 0.1), (0.3, 0.4), (0.5, 0.4), (0.7, 0.8)]
        alone = within_condition_association(self.rows({"a": varied}), "wer")
        with_constant = within_condition_association(
            self.rows({"a": varied, "zero": [(0.0, 0.0)] * 4}), "wer")
        self.assertEqual(with_constant["status"], "ok")
        self.assertAlmostEqual(with_constant["pearson_r"], alone["pearson_r"])
        all_constant = within_condition_association(self.rows({"zero": [(0.0, 0.0)] * 4}), "wer")
        self.assertIsNone(all_constant["pearson_r"])
        self.assertNotEqual(all_constant["status"], "ok")

    def test_bootstrap_recentres_and_is_reproducible(self):
        rows = self.rows({c: [(i / 10 + shift, i / 10 + shift + (0.05 if i % 2 else 0))
                              for i in range(6)] for c, shift in [("a", 0.0), ("b", 0.5)]})
        samples = cluster_resamples(rows, 50, 7)
        first = within_condition_association(rows, "wer", samples)
        second = within_condition_association(rows, "wer", samples)
        self.assertEqual(first, second)
        self.assertEqual(first["pearson_bootstrap_valid"], 50)
        self.assertLessEqual(first["pearson_ci_low"], first["pearson_r"])
        self.assertGreaterEqual(first["pearson_ci_high"], first["pearson_r"])


class CorrelationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.comparisons = self.base / "comparisons"
        for label, offset in [("one", 0.0), ("two", 0.1)]:
            rows = []
            for i in range(8):
                utterance, condition = f"u{i // 2}", ["a", "b"][i % 2]
                human_wer = [0.0, 0.2, 0.4, 0.6][i // 2] + (0.1 if condition == "b" else 0)
                human_cer = human_wer / 2
                rows.append({
                    "audio_file": f"audio/{condition}/{utterance}.wav",
                    "utterance_id": utterance, "condition": condition,
                    "speaker_id": "speaker", "gender": "F", "language": "eng",
                    "asr_system": label, "human_n_ratings": 3,
                    "human_mean_wer": human_wer, "human_mean_cer": human_cer,
                    "asr_wer": min(1.0, human_wer + offset),
                    "asr_cer": min(1.0, human_cer + offset),
                })
            write_table(self.comparisons / f"comparison_{label}.csv", rows)
        self.config = {
            "base_dir": ".", "comparison_dir": "comparisons", "output_dir": "output",
            "bootstrap_samples": 20, "random_seed": 11, "allow_incomplete": False,
        }
        self.config_path = self.base / "config.json"
        self.config_path.write_text(json.dumps(self.config))

    def test_end_to_end_uses_only_comparisons(self):
        result, out = run(self.config_path)
        self.assertEqual(result["n_expected_recordings"], 8)
        self.assertEqual(result["n_utterance_clusters"], 4)
        self.assertNotIn("human_aggregations", result)

        with (out / "02_statistics/utterance_level/02_correlations_by_condition.csv").open() as handle:
            utterance_results = list(csv.DictReader(handle))
        self.assertEqual(len(utterance_results), 8)
        self.assertEqual({int(row["n_pairs"]) for row in utterance_results}, {4})
        self.assertEqual({int(row["bootstrap_requested"]) for row in utterance_results}, {20})

        for metric in ["wer", "cer"]:
            with (out / f"02_statistics/model_level/01_condition_means_{metric}.csv").open() as handle:
                model_inputs = list(csv.DictReader(handle))
            self.assertEqual(len(model_inputs), 4)  # 2 ASRs x 2 conditions
            self.assertIn(f"human_mean_{metric}", model_inputs[0])
        with (out / "02_statistics/model_level/02_correlations.csv").open() as handle:
            model_agreement = list(csv.DictReader(handle))
        self.assertEqual(len(model_agreement), 4)
        self.assertEqual({int(row["n_conditions"]) for row in model_agreement}, {2})
        with (out / "02_statistics/utterance_level/03_correlations_within_condition.csv").open() as handle:
            within = list(csv.DictReader(handle))
        self.assertEqual(len(within), 4)  # 2 ASRs x 2 metrics; no anchors configured
        self.assertEqual({row["condition_set"] for row in within}, {"all_conditions"})
        self.assertEqual({int(row["n_pairs"]) for row in within}, {8})
        self.assertTrue((out / "02_statistics/utterance_level/wer_spearman_by_condition.csv").exists())
        self.assertTrue((out / "02_statistics/utterance_level/wer_pearson_by_condition.csv").exists())
        self.assertTrue((out / "02_statistics/disagreement/wer_disagreement_summary.csv").exists())
        self.assertFalse(list((out / "02_statistics").glob("model_level_*.csv")))

        # The input tables reproduce every utterance-level coefficient.
        with (out / "02_statistics/utterance_level/01_pairs_wer.csv").open() as handle:
            pair_rows = list(csv.DictReader(handle))
        self.assertEqual(len(pair_rows), 16)  # 2 ASRs x 8 recordings
        col = lambda rows, name: [float(r[name]) for r in rows]
        for row in utterance_results:
            if row["metric"] != "wer":
                continue
            group = [r for r in pair_rows if r["asr_system"] == row["asr_system"] and r["condition"] == row["condition"]]
            self.assertAlmostEqual(pearson(col(group, "human_mean_wer"), col(group, "asr_wer")), float(row["pearson_r"]))
            self.assertAlmostEqual(pearson(col(group, "human_rank"), col(group, "asr_rank")), float(row["spearman_rho"]))
        for row in within:
            if row["metric"] != "wer":
                continue
            group = [r for r in pair_rows if r["asr_system"] == row["asr_system"]]
            self.assertAlmostEqual(pearson(col(group, "human_centred"), col(group, "asr_centred")), float(row["pearson_r"]))
            self.assertAlmostEqual(pearson(col(group, "human_rank_centred"), col(group, "asr_rank_centred")), float(row["spearman_rho"]))

        with (out / "02_statistics/disagreement/disagreements.csv").open() as handle:
            disagreements = list(csv.DictReader(handle))
        self.assertEqual(len(disagreements), 32)
        self.assertNotIn("human_median_wer", disagreements[0])
        self.assertTrue(all(row["review_category"] == "" for row in disagreements))
        self.assertFalse((out / "02_statistics/rtf_summary.csv").exists())
        self.assertFalse((out / "scoring_audit.csv").exists())

        figures = sorted((out / "03_figures").glob("*.svg"))
        self.assertEqual(len(figures), 4)
        self.assertTrue((out / "03_figures/human_asr_wer_scatter_all_conditions.svg").exists())
        self.assertTrue((out / "03_figures/human_asr_wer_scatter_generated_conditions.svg").exists())
        self.assertTrue((out / "03_figures/wer_disagreement.svg").exists())
        self.assertTrue((out / "03_figures/model_level_wer_comparison.svg").exists())
        self.assertFalse(list((out / "03_figures").glob("model_level_*_wer.svg")))
        for path in figures:
            ET.parse(path)
        for name in ["human_asr_wer_scatter_all_conditions_plot_data.csv",
                     "human_asr_wer_scatter_generated_conditions_plot_data.csv",
                     "wer_disagreement_plot_data.csv", "model_level_wer_comparison_plot_data.csv"]:
            self.assertTrue((out / "03_figures" / name).exists())
        summary_text = (out / "03_figures/human_asr_wer_scatter_all_conditions.svg").read_text()
        self.assertIn("one", summary_text)
        self.assertIn("two", summary_text)
        self.assertIn("least-squares fit within each condition", summary_text)
        self.assertFalse((out / "02_statistics/correlations.csv").exists())
        self.assertFalse((out / "REPORT.md").exists())
        self.assertFalse((out / "config_used.json").exists())
        manifest = json.loads((out / "run_manifest.json").read_text())
        self.assertEqual(manifest["config"]["bootstrap_samples"], 20)
        self.assertEqual(len(manifest["figures"]), 4)
        with self.assertRaisesRegex(ValueError, "new/empty"):
            run(self.config_path)

    def test_anchor_conditions_add_generated_only_set(self):
        self.config["anchor_conditions"] = ["b"]
        self.config_path.write_text(json.dumps(self.config))
        _, out = run(self.config_path)
        with (out / "02_statistics/utterance_level/03_correlations_within_condition.csv").open() as handle:
            within = list(csv.DictReader(handle))
        generated = [row for row in within if row["condition_set"] == "generated_only"]
        self.assertEqual(len(generated), 4)
        self.assertEqual({int(row["n_pairs"]) for row in generated}, {4})
        self.assertEqual({row["conditions"] for row in generated}, {"a"})

    def test_missing_scores_require_explicit_common_subset(self):
        tables = load_comparisons(self.comparisons)
        tables["one"][0]["asr_wer"] = None
        with self.assertRaisesRegex(ValueError, "Missing ASR scores"):
            analyze(tables, self.base / "strict", samples=0)
        self.assertTrue((self.base / "strict/coverage.csv").exists())

        result = analyze(tables, self.base / "allowed", samples=0, allow_incomplete=True)
        self.assertEqual(result["n_common_wer"], 7)
        self.assertEqual(result["n_common_cer"], 8)
        with (self.base / "allowed/utterance_level/02_correlations_by_condition.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(
            sorted({int(row["n_pairs"]) for row in rows if row["metric"] == "wer"}),
            [3, 4],
        )

    def test_inconsistent_human_scores_across_asr_are_rejected(self):
        path = self.comparisons / "comparison_two.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["human_mean_wer"] = .12345
        write_table(path, rows)
        with self.assertRaisesRegex(ValueError, "inconsistent human"):
            load_comparisons(self.comparisons)

    def test_different_recording_universe_is_rejected(self):
        path = self.comparisons / "comparison_two.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        rows.pop()
        write_table(path, rows)
        with self.assertRaisesRegex(ValueError, "recording universe differs"):
            load_comparisons(self.comparisons)


if __name__ == "__main__":
    unittest.main()
