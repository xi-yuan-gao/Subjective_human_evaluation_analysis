import unittest

from transcription_analysis.timing import session_timing_qc


class TimingTests(unittest.TestCase):
    def test_nonmonotonic_elapsed_time_is_flagged(self):
        key = ("p1", "s1")
        sessions = {
            key: [
                {"list_id": "1", "item_order": "0", "response_time_sec": "5", "study_elapsed_sec": "10", "submitted_at": "2026-10-12T12:00:10+00:00"},
                {"list_id": "1", "item_order": "1", "response_time_sec": "4", "study_elapsed_sec": "8", "submitted_at": "2026-10-12T12:00:08+00:00"},
            ]
        }
        rows = session_timing_qc(sessions, {key}, {})
        self.assertEqual(rows[0]["timing_qc_status"], "REVIEW")
        self.assertIn("nonmonotonic_study_elapsed_sec", rows[0]["timing_qc_reasons"])
        self.assertIn("nonmonotonic_submitted_at", rows[0]["timing_qc_reasons"])


if __name__ == "__main__":
    unittest.main()
