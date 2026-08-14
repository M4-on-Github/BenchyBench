#!/usr/bin/env python3
"""Tests for Eval_CASTOR/shared/metrics.py.

    python tests/test_shared_metrics.py

This module is the foundation of the evaluation stack — normalize_state and
extract_json_block are imported by eval_castor, eval_separated, judge_castor,
salvage_analysis and visual_classification's regex_eval. A change here moves
every reported number, so its edge cases are pinned here.

Pure functions only: no cluster, no GPU, no model weights.
"""

import sys
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent / "Eval_CASTOR"
sys.path.insert(0, str(EVAL_ROOT))

from shared.metrics import (          # noqa: E402
    VALID_STATES, STATE_MAP, normalize_state, extract_json_block,
    json_loads_safe, normalize_size, vessel_jaccard, cargo_match,
)

UNPARSEABLE = "UNPARSEABLE"


class TestNormalizeState(unittest.TestCase):

    def test_canonical_states_round_trip(self):
        for state in VALID_STATES:
            self.assertEqual(normalize_state(state), state)

    def test_underscores_and_spaces_are_equivalent(self):
        self.assertEqual(normalize_state("on_fire"), "on_fire")
        self.assertEqual(normalize_state("on fire"), "on_fire")

    def test_case_and_surrounding_whitespace_ignored(self):
        self.assertEqual(normalize_state("  AGROUND  "), "aground")

    def test_substring_match_in_a_sentence(self):
        self.assertEqual(normalize_state("the vessel is aground on rocks"), "aground")

    def test_returns_the_sentinel_not_none(self):
        # Callers test against the literal "UNPARSEABLE"; returning None would
        # silently pass truthiness checks written the other way round.
        self.assertEqual(normalize_state("a calm harbour"), UNPARSEABLE)
        self.assertEqual(normalize_state(None), UNPARSEABLE)
        self.assertEqual(normalize_state(""), UNPARSEABLE)

    def test_hedged_answers_are_refused_not_resolved(self):
        # The important guard: when the model offers alternatives, substring
        # matching is skipped so one state is not picked arbitrarily. A hedge
        # is a parse failure, not a confident answer.
        self.assertEqual(normalize_state("aground|capsized"), UNPARSEABLE)
        self.assertEqual(normalize_state("aground/capsized"), UNPARSEABLE)

    def test_exact_match_still_wins_despite_a_separator(self):
        # Exact lookup happens before the separator guard.
        self.assertEqual(normalize_state("aground"), "aground")

    def test_escaped_backslashes_are_stripped(self):
        # Model output frequently arrives with escaping artefacts.
        self.assertEqual(normalize_state("\\\\aground"), "aground")

    def test_non_string_input_is_coerced(self):
        self.assertEqual(normalize_state(123), UNPARSEABLE)

    def test_good_is_a_fifth_state_outside_VALID_STATES(self):
        """Documents a real asymmetry rather than asserting it away.

        STATE_MAP maps good / floating / undamaged -> "good", but VALID_STATES
        lists only the four CASTOR classes. That is intentional as far as
        scoring goes: the prompts do offer "or good", no ground-truth image is
        labelled "good", so such a prediction parses successfully and counts as
        wrong. Correct.

        The consequence worth knowing: judge_castor.py iterates VALID_STATES
        for its per-state reports, so "good" predictions are counted in the
        totals but never appear as a row. They are invisible in the breakdown,
        not miscounted.

        Widening VALID_STATES would change the shape of every confusion matrix
        and per-state report, so it is left alone deliberately.
        """
        self.assertEqual(normalize_state("good"), "good")
        self.assertEqual(normalize_state("undamaged"), "good")
        self.assertNotIn("good", VALID_STATES)

    def test_every_other_state_map_value_is_a_valid_state(self):
        for key, value in STATE_MAP.items():
            if value == "good":
                continue        # see the test above
            self.assertIn(value, VALID_STATES, "%s -> %s" % (key, value))


class TestExtractJsonBlock(unittest.TestCase):

    def test_extracts_a_plain_object(self):
        parsed, reason = extract_json_block('{"state": "aground"}')
        self.assertEqual(parsed["state"], "aground")
        self.assertEqual(reason, "")

    def test_extracts_json_embedded_in_prose(self):
        text = 'Here is my answer:\n{"state": "sunken"}\nHope that helps.'
        parsed, _ = extract_json_block(text)
        self.assertEqual(parsed["state"], "sunken")

    def test_prefers_the_block_containing_state(self):
        # A reasoning block without 'state' must not win over the answer block.
        text = '{"note": "thinking"} then {"state": "capsized"}'
        parsed, reason = extract_json_block(text)
        self.assertEqual(parsed["state"], "capsized")
        self.assertEqual(reason, "")

    def test_falls_back_to_a_stateless_dict_and_says_so(self):
        parsed, reason = extract_json_block('{"vessel": "cargo ship"}')
        self.assertIsNotNone(parsed)
        self.assertIn("no_state_key", reason)

    def test_no_braces_is_reported_distinctly(self):
        parsed, reason = extract_json_block("The vessel is aground.")
        self.assertIsNone(parsed)
        self.assertIn("no_braces", reason)

    def test_unparseable_braces_report_the_attempts(self):
        parsed, reason = extract_json_block("{not json at all}")
        self.assertIsNone(parsed)
        self.assertTrue(reason)

    def test_truncated_json_does_not_raise(self):
        # Judge and VLM output is regularly cut off by a token limit.
        parsed, reason = extract_json_block('{"state": "aground", "vessel": "car')
        self.assertIsNone(parsed)
        self.assertTrue(reason)

    def test_nested_objects_are_matched_to_the_right_brace(self):
        text = '{"state": "sunken", "detail": {"depth": "10m"}}'
        parsed, _ = extract_json_block(text)
        self.assertEqual(parsed["state"], "sunken")
        self.assertEqual(parsed["detail"]["depth"], "10m")

    def test_empty_input(self):
        parsed, reason = extract_json_block("")
        self.assertIsNone(parsed)
        self.assertIn("no_braces", reason)

    def test_a_reason_is_always_returned_on_failure(self):
        # The reason string is what makes a parse failure diagnosable.
        for text in ("", "no braces", "{broken", "{}"):
            parsed, reason = extract_json_block(text)
            if parsed is None:
                self.assertTrue(reason, "no reason given for %r" % text)


class TestJsonLoadsSafe(unittest.TestCase):

    def test_parses_ordinary_json(self):
        self.assertEqual(json_loads_safe('{"a": 1}'), {"a": 1})

    def test_retries_with_strict_false_for_control_characters(self):
        # Raw newlines inside strings are invalid under strict JSON but common
        # in model output; the retry is what rescues those records.
        self.assertEqual(json_loads_safe('{"a": "line1\nline2"}')["a"],
                         "line1\nline2")


class TestFieldMatchers(unittest.TestCase):

    def test_normalize_size_returns_a_string(self):
        for text in ("large", "small", "medium", "", "nonsense"):
            self.assertIsInstance(normalize_size(text), str)

    def test_vessel_jaccard_bounds(self):
        self.assertEqual(vessel_jaccard("cargo ship", "cargo ship"), 1.0)
        self.assertEqual(vessel_jaccard("cargo ship", "fishing trawler"), 0.0)
        overlap = vessel_jaccard("large cargo ship", "cargo ship")
        self.assertGreater(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)

    def test_vessel_jaccard_handles_empty_input(self):
        self.assertIsInstance(vessel_jaccard("", ""), float)
        self.assertIsInstance(vessel_jaccard("ship", ""), float)

    def test_cargo_match_returns_a_string_verdict(self):
        for gt, pred in [("oil", "oil"), ("oil", "containers"),
                         ("", ""), ("oil", None)]:
            self.assertIsInstance(cargo_match(gt, pred), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
