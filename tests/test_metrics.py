import unittest

from transcription_analysis.metrics import edit_counts, edit_counts_with_wildcards
from transcription_analysis.normalize import EXCLUDED_NAME_TOKEN, normalize_text


class MetricTests(unittest.TestCase):
    def test_edit_decomposition(self):
        result = edit_counts("the quick fox".split(), "the slow fox jumps".split())
        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["insertions"], 1)
        self.assertEqual(result["deletions"], 0)
        self.assertAlmostEqual(result["error_rate"], 2 / 3)

    def test_conservative_normalization_preserves_xxx(self):
        settings = {
            "unicode_nfkc": True,
            "standardize_apostrophes": True,
            "lowercase": True,
            "remove_punctuation": True,
            "collapse_whitespace": True,
            "word_map": {},
        }
        self.assertEqual(normalize_text("  I CAN’T hear—xxx! ", settings), "i can't hear xxx")

    def test_apostrophes_can_be_removed_without_splitting_words(self):
        settings = {
            "unicode_nfkc": True,
            "standardize_apostrophes": True,
            "lowercase": True,
            "remove_apostrophes": True,
            "remove_punctuation": True,
            "collapse_whitespace": True,
            "word_map": {},
        }
        text = "Ain't aren't haven't I'll we're you'll neighbor's"
        self.assertEqual(
            normalize_text(text, settings),
            "aint arent havent ill were youll neighbors",
        )

    def test_name_wildcards_ignore_name_variants_but_score_other_words(self):
        reference = ["she", "was", "created", "by", EXCLUDED_NAME_TOKEN, "and", EXCLUDED_NAME_TOKEN]
        hypothesis = "he was created by team shooter and mike sack".split()
        result = edit_counts_with_wildcards(reference, hypothesis, EXCLUDED_NAME_TOKEN)
        self.assertEqual(result["substitutions"], 1)
        self.assertEqual(result["deletions"], 0)
        self.assertEqual(result["insertions"], 0)
        self.assertEqual(result["reference_length"], 5)
        self.assertEqual(result["evaluated_reference"], ["she", "was", "created", "by", "and"])
        self.assertEqual(result["evaluated_hypothesis"], ["he", "was", "created", "by", "and"])

    def test_name_wildcard_prefers_absorbing_variant_on_equal_error_paths(self):
        reference = [EXCLUDED_NAME_TOKEN, "is", "currently"]
        hypothesis = ["stacys", "currently"]
        result = edit_counts_with_wildcards(reference, hypothesis, EXCLUDED_NAME_TOKEN)
        self.assertEqual(result["evaluated_reference"], ["is", "currently"])
        self.assertEqual(result["evaluated_hypothesis"], ["currently"])
        self.assertEqual(result["deletions"], 1)
        self.assertEqual(result["insertions"], 0)

    def test_blank_transcription_is_all_deletions(self):
        result = edit_counts("three reference words".split(), [])
        self.assertEqual(result["deletions"], 3)
        self.assertEqual(result["error_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
