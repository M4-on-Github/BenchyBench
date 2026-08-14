#!/usr/bin/env python3
"""Tests for the P8 salvage-plan step parser.

    python tests/test_parse_steps.py

Plan coherence is scored on the SEQUENCE of steps, so this parser decides what
the judge is even shown. A plan whose steps fail to parse scores as if it had
none — a formatting difference becomes a coherence finding, which is the wrong
conclusion for the right-looking reason.

Pure text processing: no LLM, no cluster.
"""

import sys
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent / "Eval_CASTOR"
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_coherence.parse_steps import parse_steps      # noqa: E402


class TestNumberedFormat(unittest.TestCase):
    """Primary format: a plain numbered list."""

    def test_simple_list(self):
        steps = parse_steps("1. Secure the area.\n2. Deploy divers.")
        self.assertEqual(steps, [(1, "Secure the area."), (2, "Deploy divers.")])

    def test_step_numbers_are_preserved_not_reindexed(self):
        # The judge reasons about ordering, so a plan that skips or repeats a
        # number must keep that as-is rather than being silently renumbered.
        steps = parse_steps("1. First.\n3. Third.\n2. Second.")
        self.assertEqual([n for n, _ in steps], [1, 3, 2])

    def test_document_order_is_preserved(self):
        steps = parse_steps("2. Second.\n1. First.")
        self.assertEqual([t for _, t in steps], ["Second.", "First."])

    def test_multiline_step_bodies_are_joined(self):
        steps = parse_steps("1. Secure the area\n   and wait for tide.\n2. Next.")
        self.assertEqual(steps[0][1], "Secure the area and wait for tide.")

    def test_internal_whitespace_is_collapsed(self):
        self.assertEqual(parse_steps("1. Too    many     spaces.")[0][1],
                         "Too many spaces.")

    def test_bold_markers_are_stripped(self):
        self.assertEqual(parse_steps("1. **Secure** the area.")[0][1],
                         "Secure the area.")

    def test_double_digit_steps(self):
        steps = parse_steps("\n".join("%d. Step %d." % (i, i) for i in range(1, 13)))
        self.assertEqual(len(steps), 12)
        self.assertEqual(steps[-1][0], 12)

    def test_leading_prose_before_the_list_is_ignored(self):
        steps = parse_steps("Here is my plan:\n\n1. Secure.\n2. Deploy.")
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0][1], "Secure.")

    def test_BUG_an_empty_step_swallows_the_following_step(self):
        """Documents a real defect rather than asserting the desired behaviour.

        "1. \\n2. Real step." parses as ONE step whose body is "2. Real step.",
        instead of dropping the empty step 1 and returning step 2.

        Cause: the pattern is

            (?:^|\\n)\\s*(\\d+)\\.\\s+(.*?)(?=\\n\\s*\\d+\\.|\\Z)

        and `\\s+` after "1." consumes the newline. The lookahead then cannot
        find "\\n2." as a boundary, so the lazy body runs to end-of-string and
        absorbs the remaining steps.

        Impact: a plan containing an empty numbered line collapses into a
        single step, and P8 scores its coherence as if the model produced one
        step. Nothing raises.

        Not fixed here: this feeds a starred pipeline, and changing the parse
        would change previously reported coherence numbers. Recorded so the
        trade-off is a decision rather than a surprise.
        """
        steps = parse_steps("1. \n2. Real step.")
        self.assertEqual(steps, [(1, "2. Real step.")])

    def test_a_genuinely_empty_trailing_step_is_dropped(self):
        # When nothing follows, the empty body is correctly discarded.
        self.assertEqual(parse_steps("1. Real step.\n2. "),
                         [(1, "Real step.")])


class TestBoldHeaderFallback(unittest.TestCase):
    """Fallback format: **Step N: Title** with a bullet body beneath."""

    def test_bold_step_headers(self):
        text = ("**Step 1: Assess the wreck**\n"
                "- Survey the hull\n"
                "**Step 2: Refloat**\n"
                "- Pump the compartments\n")
        steps = parse_steps(text)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0][0], 1)
        self.assertIn("Assess the wreck", steps[0][1])

    def test_body_is_folded_into_the_step_text(self):
        text = "**Step 1: Assess**\n- Survey the hull\n- Check the tide\n"
        self.assertIn("Survey the hull", parse_steps(text)[0][1])

    def test_fallback_is_only_used_when_the_primary_finds_nothing(self):
        # A plain numbered list must not be re-parsed by the bold matcher.
        text = "1. Plain step.\n2. Another."
        self.assertEqual(parse_steps(text), [(1, "Plain step."), (2, "Another.")])

    def test_case_insensitive_step_keyword(self):
        self.assertTrue(parse_steps("**STEP 1: Assess**\n- do it\n"))


class TestDegenerateInput(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(parse_steps(""), [])

    def test_whitespace_only(self):
        self.assertEqual(parse_steps("   \n\n  "), [])

    def test_none_is_handled(self):
        self.assertEqual(parse_steps(None), [])

    def test_prose_with_no_steps(self):
        # This is the case that silently costs a plan its score: unparseable
        # output is indistinguishable from a plan with no steps.
        self.assertEqual(
            parse_steps("The vessel should be refloated at high tide."), [])

    def test_a_refusal_yields_no_steps(self):
        self.assertEqual(parse_steps("I cannot determine a salvage plan."), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
