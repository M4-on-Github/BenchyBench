#!/usr/bin/env python3
"""Tests for Eval_CASTOR P3 semantic judge helpers.

    python tests/test_judge_castor.py

Covers the pure parts of judge_castor.py: prompt construction and verdict
unpacking. The Ollama call itself is not exercised — everything around it is.

Two behaviours here decide what the judge sees and how its answer is counted,
and both fail quietly when wrong.

Pure text and dict handling: no Ollama, no cluster.
"""

import sys
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent / "Eval_CASTOR"
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.judge_castor import (              # noqa: E402
    _gv, build_prompt_full, build_prompt_extracted, unpack_verdict,
    JUDGE_FIELDS, _TEXT_WINDOW,
)

GT = {"state": "aground", "vessel_type": "cargo ship",
      "size_estimate": "large", "cargo": "containers",
      "q1": "yes", "q2": "no", "q3": "yes", "q4": "no", "q5": "yes"}


class TestGemmaValue(unittest.TestCase):
    """_gv normalises a possibly-missing extracted field for the prompt."""

    def test_none_becomes_unknown(self):
        self.assertEqual(_gv(None), "UNKNOWN")

    def test_literal_unknown_is_normalised_regardless_of_case(self):
        for v in ("UNKNOWN", "unknown", "Unknown", "  unknown  "):
            self.assertEqual(_gv(v), "UNKNOWN", repr(v))

    def test_ordinary_values_are_stripped(self):
        self.assertEqual(_gv("  aground  "), "aground")

    def test_non_strings_are_coerced(self):
        self.assertEqual(_gv(42), "42")
        self.assertEqual(_gv(True), "True")

    def test_empty_string_is_not_unknown(self):
        # An empty extraction is distinct from an explicit UNKNOWN — it means
        # the field was absent rather than the model declining to answer.
        self.assertEqual(_gv(""), "")


class TestFullPrompt(unittest.TestCase):

    def test_returns_system_and_user_parts(self):
        sys_p, user_p = build_prompt_full("some answer", GT)
        self.assertTrue(sys_p.strip())
        self.assertTrue(user_p.strip())

    def test_ground_truth_is_interpolated(self):
        _, user_p = build_prompt_full("answer", GT)
        self.assertIn("aground", user_p)
        self.assertIn("cargo ship", user_p)

    def test_short_text_is_included_whole(self):
        _, user_p = build_prompt_full("the vessel is aground", GT)
        self.assertIn("the vessel is aground", user_p)

    def test_long_text_keeps_the_TAIL_not_the_head(self):
        # Deliberate: chain-of-thought output puts the JSON answer LAST, so
        # truncating from the front would discard the very thing being judged.
        text = ("HEAD" + "x" * (_TEXT_WINDOW * 2) + "TAIL")
        _, user_p = build_prompt_full(text, GT)
        self.assertIn("TAIL", user_p)
        self.assertNotIn("HEAD", user_p)

    def test_truncation_boundary(self):
        exact = "y" * _TEXT_WINDOW
        _, user_p = build_prompt_full(exact, GT)
        self.assertIn(exact, user_p)

    def test_missing_ground_truth_fields_do_not_raise(self):
        _, user_p = build_prompt_full("answer", {})
        self.assertTrue(user_p.strip())


class TestExtractedPrompt(unittest.TestCase):

    def test_predictions_are_interpolated(self):
        rec = {"state": "sunken", "vessel_type": "tanker"}
        _, user_p = build_prompt_extracted(rec, GT)
        self.assertIn("sunken", user_p)
        self.assertIn("tanker", user_p)

    def test_missing_predictions_render_as_unknown(self):
        _, user_p = build_prompt_extracted({}, GT)
        self.assertIn("UNKNOWN", user_p)

    def test_both_ground_truth_and_prediction_appear(self):
        _, user_p = build_prompt_extracted({"state": "sunken"}, GT)
        self.assertIn("sunken", user_p)     # prediction
        self.assertIn("aground", user_p)    # ground truth


class TestUnpackVerdict(unittest.TestCase):

    def test_dict_form_yields_flag_and_reason(self):
        out = unpack_verdict({"state": {"correct": True, "reason": "matches"}})
        self.assertTrue(out["state"])
        self.assertEqual(out["state_reason"], "matches")

    def test_bare_bool_is_accepted_with_an_empty_reason(self):
        out = unpack_verdict({"state": True})
        self.assertTrue(out["state"])
        self.assertEqual(out["state_reason"], "")

    def test_every_field_is_always_present(self):
        out = unpack_verdict({})
        for field in JUDGE_FIELDS:
            self.assertIn(field, out)
            self.assertIn(field + "_reason", out)

    def test_a_missing_field_defaults_to_False(self):
        out = unpack_verdict({})
        self.assertFalse(out["state"])

    def test_UNPARSEABLE_JUDGE_OUTPUT_COUNTS_AS_WRONG(self):
        # The distinction this does NOT make: a malformed judge response is
        # recorded as correct=False, identical to the judge saying the answer
        # was wrong. A judge formatting failure therefore depresses the
        # reported accuracy rather than being excluded from it.
        #
        # The reason string is the only trace — it records the offending value,
        # so such records are findable after the fact.
        out = unpack_verdict({"state": "not a verdict"})
        self.assertFalse(out["state"])
        self.assertIn("unexpected", out["state_reason"])

    def test_the_offending_value_is_preserved_in_the_reason(self):
        out = unpack_verdict({"state": 12345})
        self.assertIn("12345", out["state_reason"])

    def test_dict_without_correct_key_defaults_False(self):
        out = unpack_verdict({"state": {"reason": "no verdict given"}})
        self.assertFalse(out["state"])

    def test_nine_fields_are_judged(self):
        # state, vessel_type, size_estimate, cargo, and q1-q5
        self.assertEqual(len(JUDGE_FIELDS), 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
