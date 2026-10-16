from __future__ import annotations

import unittest

from transcription_analysis.summarize import summarize_inter_listener_consistency


class InterListenerConsistencyTests(unittest.TestCase):
    def test_complete_groups_use_sample_sd_and_range(self):
        rows = [
            {"utterance_id": "u1", "condition": "a", "participant_id": "p1", "session_id": "s1", "wer": 0.0, "cer": 0.0},
            {"utterance_id": "u1", "condition": "a", "participant_id": "p2", "session_id": "s2", "wer": 0.5, "cer": 0.3},
            {"utterance_id": "u1", "condition": "a", "participant_id": "p3", "session_id": "s3", "wer": 1.0, "cer": 0.6},
        ]

        by_item, by_condition = summarize_inter_listener_consistency(rows, expected_ratings=3)

        self.assertEqual(len(by_item), 1)
        self.assertTrue(by_item[0]["complete_listener_set"])
        self.assertAlmostEqual(by_item[0]["listener_sd_wer"], 0.5)
        self.assertAlmostEqual(by_item[0]["listener_range_wer"], 1.0)
        self.assertAlmostEqual(by_item[0]["listener_sd_cer"], 0.3)
        self.assertEqual(by_condition[0]["complete_listener_groups"], 1)
        self.assertAlmostEqual(by_condition[0]["median_listener_sd_wer"], 0.5)

    def test_incomplete_groups_remain_visible_but_not_aggregated(self):
        rows = [
            {"utterance_id": "u1", "condition": "a", "participant_id": "p1", "session_id": "s1", "wer": 0.0, "cer": 0.0},
            {"utterance_id": "u1", "condition": "a", "participant_id": "p2", "session_id": "s2", "wer": 1.0, "cer": 0.5},
        ]

        by_item, by_condition = summarize_inter_listener_consistency(rows, expected_ratings=3)

        self.assertFalse(by_item[0]["complete_listener_set"])
        self.assertAlmostEqual(by_item[0]["listener_range_wer"], 1.0)
        self.assertEqual(by_condition[0]["complete_listener_groups"], 0)
        self.assertEqual(by_condition[0]["incomplete_listener_groups"], 1)
        self.assertEqual(by_condition[0]["median_listener_sd_wer"], "")


if __name__ == "__main__":
    unittest.main()
