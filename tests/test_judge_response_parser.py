#!/usr/bin/env python3
"""Tests for the P5 judge response parser.

    python tests/test_judge_response_parser.py

This is the code path that failed at ~50% earlier in this project, when
DeepSeek-R1 and Selene hit their token limit mid-JSON. Raising max_tokens fixed
the symptom; these tests pin what the parser does and does not rescue, so the
next occurrence is diagnosable rather than mysterious.

The parser has to survive several judge-specific quirks:

  * DeepSeek-R1 emits a <think>...</think> reasoning block before its answer
  * an unclosed <think> means the response was TRUNCATED mid-reasoning
  * models wrap JSON in markdown code fences
  * some escape underscores LaTeX-style
  * some add prose before or after the JSON object

Pure text and JSON handling: no vLLM, no cluster, no weights.
"""

import json
import sys
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent / "Eval_CASTOR"
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.judge_panel.run_judge import (        # noqa: E402
    parse_judge_response, _coerce_list, VALID_SCORES,
)


def judge_json(score=3, rationale="looks right", hallucinations=None):
    return json.dumps({
        "final_score": score,
        "visual_alignment_rationale": rationale,
        "hallucinations_detected": hallucinations or [],
    })


class TestCleanResponses(unittest.TestCase):

    def test_plain_json(self):
        out = parse_judge_response(judge_json())
        self.assertTrue(out["parse_ok"])
        self.assertEqual(out["score"], 3)
        self.assertEqual(out["rationale"], "looks right")

    def test_all_valid_scores_accepted(self):
        for score in sorted(VALID_SCORES):
            out = parse_judge_response(judge_json(score=score))
            self.assertTrue(out["parse_ok"], score)
            self.assertEqual(out["score"], score)

    def test_hallucinations_are_carried_through(self):
        out = parse_judge_response(judge_json(hallucinations=["ghost crane"]))
        self.assertEqual(out["hallucinations"], ["ghost crane"])


class TestJudgeQuirks(unittest.TestCase):

    def test_deepseek_think_block_is_stripped(self):
        raw = "<think>Let me reason about this...</think>\n" + judge_json()
        out = parse_judge_response(raw)
        self.assertTrue(out["parse_ok"])
        self.assertEqual(out["score"], 3)

    def test_markdown_code_fence_is_stripped(self):
        raw = "```json\n" + judge_json() + "\n```"
        self.assertTrue(parse_judge_response(raw)["parse_ok"])

    def test_bare_fence_without_language(self):
        raw = "```\n" + judge_json() + "\n```"
        self.assertTrue(parse_judge_response(raw)["parse_ok"])

    def test_latex_escaped_underscores_are_unescaped(self):
        raw = judge_json().replace("final_score", "final\\_score")
        self.assertTrue(parse_judge_response(raw)["parse_ok"])

    def test_preamble_prose_before_the_json(self):
        raw = "Here is my assessment:\n" + judge_json()
        self.assertTrue(parse_judge_response(raw)["parse_ok"])

    def test_trailing_prose_after_the_json(self):
        raw = judge_json() + "\n\nI hope this helps."
        self.assertTrue(parse_judge_response(raw)["parse_ok"])

    def test_think_block_plus_fence_plus_preamble(self):
        raw = ("<think>reasoning</think>\nHere you go:\n```json\n"
               + judge_json() + "\n```\nDone.")
        self.assertTrue(parse_judge_response(raw)["parse_ok"])

    def test_raw_newlines_inside_strings_are_tolerated(self):
        # Strict JSON forbids these; the parser retries with strict=False.
        raw = '{"final_score": 3, "visual_alignment_rationale": "line1\nline2"}'
        self.assertTrue(parse_judge_response(raw)["parse_ok"])


class TestTruncation(unittest.TestCase):
    """The failure mode that actually occurred in this project."""

    def test_unclosed_think_block_means_truncated_mid_reasoning(self):
        # The response ran out of tokens before it ever emitted JSON. Stripping
        # from "<think>" to end leaves nothing parseable.
        out = parse_judge_response("<think>I am still reasoning and then the")
        self.assertFalse(out["parse_ok"])
        self.assertIsNone(out["score"])

    def test_json_cut_off_mid_object(self):
        # The observed failure: a long rationale hit max_tokens before the
        # closing brace.
        raw = '{"visual_alignment_rationale": "The VLM describes a vessel which is not'
        out = parse_judge_response(raw)
        self.assertFalse(out["parse_ok"])
        self.assertIsNone(out["score"])

    def test_failure_preserves_the_raw_response_for_diagnosis(self):
        # Without this the only evidence of WHY a record failed is gone.
        out = parse_judge_response("total garbage, no json here")
        self.assertFalse(out["parse_ok"])
        self.assertIn("raw_response", out)
        self.assertIn("garbage", out["raw_response"])

    def test_raw_response_is_capped(self):
        out = parse_judge_response("x" * 5000)
        self.assertLessEqual(len(out["raw_response"]), 500)


class TestInvalidScores(unittest.TestCase):

    def test_out_of_range_score_is_rejected_but_rationale_kept(self):
        # A judge answering 5 on a 1-3 scale did produce a rationale, so the
        # text is retained even though the score is discarded.
        out = parse_judge_response(judge_json(score=5))
        self.assertIsNone(out["score"])
        self.assertEqual(out["rationale"], "looks right")

    def test_zero_is_not_a_valid_score(self):
        self.assertIsNone(parse_judge_response(judge_json(score=0))["score"])

    def test_string_score_is_rejected(self):
        raw = '{"final_score": "3", "visual_alignment_rationale": "r"}'
        self.assertIsNone(parse_judge_response(raw)["score"])

    def test_missing_score_field(self):
        raw = '{"visual_alignment_rationale": "no score given"}'
        self.assertIsNone(parse_judge_response(raw)["score"])

    def test_valid_scores_are_one_to_three(self):
        self.assertEqual(VALID_SCORES, {1, 2, 3})


class TestDegenerate(unittest.TestCase):

    def test_empty_response(self):
        out = parse_judge_response("")
        self.assertFalse(out["parse_ok"])

    def test_json_array_instead_of_object(self):
        out = parse_judge_response('[1, 2, 3]')
        self.assertFalse(out["parse_ok"])

    def test_json_scalar_instead_of_object(self):
        self.assertFalse(parse_judge_response('42')["parse_ok"])


class TestCoerceList(unittest.TestCase):

    def test_none_becomes_empty(self):
        self.assertEqual(_coerce_list(None), [])

    def test_list_is_stringified_elementwise(self):
        self.assertEqual(_coerce_list(["a", 1]), ["a", "1"])

    def test_scalar_is_wrapped(self):
        # A judge naming one hallucination as a bare string must not be
        # iterated character-by-character.
        self.assertEqual(_coerce_list("ghost crane"), ["ghost crane"])

    def test_empty_list_stays_empty(self):
        self.assertEqual(_coerce_list([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
