#!/usr/bin/env python3
"""Tests for the P8 coherence statistics.

    python tests/test_coherence_stats.py

Fleiss' kappa measures whether the FIVE coherence judges agree beyond chance.
P8 uses a larger panel than P5's three-model judge panel.
It is reported as a confidence signal on the whole pipeline: low kappa means
the judges are not measuring the same thing, and every per-run coherence number
should be read more sceptically.

A bug here would not raise. It would report a plausible agreement figure, and a
wrong kappa is worse than none — it invites confidence that is not earned.

Pure arithmetic: no vLLM, no cluster.
"""

import sys
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent / "Eval_CASTOR"
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.plan_coherence.aggregate_coherence import (   # noqa: E402
    _fleiss_kappa, _safe_mean, JUDGE_MODELS, MAJORITY_THRESHOLD,
)


def step(*votes):
    """One per-step row: a vote per judge, '1' valid / '0' invalid.

    P8 uses FIVE coherence judges, unlike P5's three-model panel. A row must
    carry a vote for every one of them or the step counts as unrated and is
    skipped entirely.
    """
    assert len(votes) == len(JUDGE_MODELS), (
        "expected %d votes, got %d" % (len(JUDGE_MODELS), len(votes)))
    return {f"{m}_valid": v for m, v in zip(JUDGE_MODELS, votes)}


class TestFleissKappa(unittest.TestCase):

    def test_unanimous_on_both_categories_is_perfect_agreement(self):
        # Judges always agree, and both categories occur, so chance agreement
        # is below 1 and kappa is defined.
        rows = [step("1", "1", "1", "1", "1"), step("0", "0", "0", "0", "0"),
                step("1", "1", "1", "1", "1"), step("0", "0", "0", "0", "0")]
        self.assertEqual(_fleiss_kappa(rows), 1.0)

    def test_single_category_throughout_is_undefined_not_perfect(self):
        # Every judge votes valid on every step. Observed agreement is total,
        # but so is expected agreement — there is no variance to agree about,
        # so kappa is undefined rather than 1.0. Returning 1.0 here would
        # claim perfect reliability from a degenerate sample.
        rows = [step("1", "1", "1", "1", "1") for _ in range(5)]
        self.assertIsNone(_fleiss_kappa(rows))

    def test_maximal_disagreement_gives_negative_kappa(self):
        # Judges split on every step: worse than chance.
        rows = [step("1", "0", "1", "0", "1"), step("0", "1", "0", "1", "0"),
                step("1", "0", "1", "0", "1"), step("0", "1", "0", "1", "0")]
        k = _fleiss_kappa(rows)
        self.assertIsNotNone(k)
        self.assertLess(k, 0)

    def test_partial_agreement_lands_between(self):
        rows = [step("1", "1", "1", "1", "1"), step("1", "1", "1", "1", "0"),
                step("0", "0", "0", "0", "0"), step("0", "0", "0", "0", "1")]
        k = _fleiss_kappa(rows)
        self.assertGreater(k, 0)
        self.assertLess(k, 1.0)

    def test_fewer_than_two_rated_steps_is_undefined(self):
        self.assertIsNone(_fleiss_kappa([]))
        self.assertIsNone(_fleiss_kappa([step("1", "1", "1", "1", "1")]))

    def test_steps_with_a_missing_vote_are_skipped(self):
        # A judge that errored on a step leaves that step unrated. Counting it
        # as agreement or disagreement would both be wrong.
        rows = [step("1", "1", "1", "1", ""), step("1", "1", "1", "1", "1"),
                step("0", "0", "0", "0", "0")]
        self.assertEqual(_fleiss_kappa(rows), 1.0)   # only the two full rows

    def test_error_votes_are_skipped(self):
        rows = [step("1", "error", "1", "1", "1"), step("1", "1", "1", "1", "1"),
                step("0", "0", "0", "0", "0")]
        self.assertEqual(_fleiss_kappa(rows), 1.0)

    def test_all_steps_unrated_is_undefined(self):
        self.assertIsNone(_fleiss_kappa([step("", "", "", "", ""),
                                         step("error", "", "", "", "")]))

    def test_result_is_rounded_for_reporting(self):
        rows = [step("1", "1", "1", "1", "1"), step("1", "1", "1", "1", "0"),
                step("0", "0", "0", "0", "0"), step("0", "0", "0", "0", "1")]
        k = _fleiss_kappa(rows)
        self.assertEqual(k, round(k, 4))

    def test_five_judges_are_configured(self):
        # P8 coherence uses a LARGER panel than P5's three-model judge panel.
        self.assertEqual(len(JUDGE_MODELS), 5)

    def test_majority_threshold_is_a_bare_majority_of_five(self):
        # 3 of 5 — a step is called invalid on a simple majority, not on
        # unanimity. Two judges can dissent and the step is still marked bad.
        self.assertEqual(MAJORITY_THRESHOLD, 3)
        self.assertGreater(MAJORITY_THRESHOLD, len(JUDGE_MODELS) / 2)


class TestSafeMean(unittest.TestCase):

    def test_averages_values(self):
        self.assertEqual(_safe_mean([1.0, 2.0, 3.0]), 2.0)

    def test_empty_is_none_not_zero(self):
        # 0.0 would read as "the judges scored this zero", the opposite of
        # "there was nothing to score".
        self.assertIsNone(_safe_mean([]))

    def test_single_value(self):
        self.assertEqual(_safe_mean([2.5]), 2.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
