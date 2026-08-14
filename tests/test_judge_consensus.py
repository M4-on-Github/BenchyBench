#!/usr/bin/env python3
"""Tests for the LLM judge panel consensus (Eval_CASTOR P5).

    python tests/test_judge_consensus.py

This is where three judge models' opinions become one verdict. Everything
downstream — summary.csv accuracies, per-image tiers, the regex-judge kappa —
is computed from these records, so a change here moves every judged number.

Characterization tests, written before restructuring. Pure data processing:
no cluster, no vLLM, no model weights.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent / "Eval_CASTOR"
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.judge_panel.aggregate import (      # noqa: E402
    _record_id, load_judge_jsonl, compute_consensus,
    JUDGE_MODELS, FIELD_KEYS, STD_FLAG_THRESHOLD,
)


class TestRecordId(unittest.TestCase):

    def test_builds_the_composite_key(self):
        rec = {"image": "aground/1.jpg", "model_tag": "llava",
               "method": "degf", "prompt_stem": "p1"}
        self.assertEqual(_record_id(rec), "aground/1.jpg||llava||degf||p1")

    def test_same_image_different_method_are_distinct_keys(self):
        # The whole point of the composite: one image appears once per
        # combination, and each needs its own verdict. Keying on image alone
        # would keep only the last and silently discard the rest.
        base = {"image": "a/1.jpg", "model_tag": "llava", "prompt_stem": "p1"}
        self.assertNotEqual(_record_id(dict(base, method="baseline")),
                            _record_id(dict(base, method="degf")))

    def test_missing_fields_yield_empty_segments_NOT_a_bare_image(self):
        # The docstring claims it "falls back to image alone for old records
        # that lack the extra fields". It does NOT — all three separators are
        # always emitted, so an old record keys as "a/1.jpg||||||" and will not
        # match a lookup by bare image path. The behaviour is fine; the
        # docstring describes a fallback that was never implemented.
        self.assertEqual(_record_id({"image": "a/1.jpg"}), "a/1.jpg||||||")
        self.assertNotEqual(_record_id({"image": "a/1.jpg"}), "a/1.jpg")


class TestLoadJudgeJsonl(unittest.TestCase):

    def _write(self, lines):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                          delete=False, encoding="utf-8")
        tmp.write("\n".join(lines))
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)

    def test_keys_records_by_composite_id(self):
        p = self._write([json.dumps({"image": "a/1.jpg", "model_tag": "llava",
                                     "method": "degf", "prompt_stem": "p1"})])
        loaded = load_judge_jsonl(p)
        self.assertIn("a/1.jpg||llava||degf||p1", loaded)

    def test_malformed_lines_are_skipped(self):
        # A judge job killed mid-write must not block aggregation of the rest.
        p = self._write([json.dumps({"image": "a/1.jpg"}),
                         '{"image": "b/2.jpg", "trunca',
                         json.dumps({"image": "c/3.jpg"})])
        self.assertEqual(len(load_judge_jsonl(p)), 2)

    def test_records_without_an_image_are_dropped(self):
        p = self._write([json.dumps({"score": 3}),
                         json.dumps({"image": "a/1.jpg"})])
        self.assertEqual(len(load_judge_jsonl(p)), 1)

    def test_blank_lines_ignored(self):
        p = self._write([json.dumps({"image": "a/1.jpg"}), "", "   "])
        self.assertEqual(len(load_judge_jsonl(p)), 1)


def consensus(scores, field_votes=None, **kw):
    return compute_consensus(
        image="a/1.jpg", gt_state="aground", pred_text="text",
        verbosity_flagged=False, scores=scores,
        rationales={m: "" for m in scores},
        hallucinations={m: [] for m in scores},
        field_votes=field_votes, **kw)


class TestConsensusScoring(unittest.TestCase):

    def test_mean_and_std_across_judges(self):
        rec = consensus({"a": 3, "b": 3, "c": 3})
        self.assertEqual(rec["mean_score"], 3)
        self.assertEqual(rec["score_std"], 0.0)

    def test_single_judge_has_zero_std_not_an_error(self):
        # stdev() of one sample raises; the code must special-case it.
        rec = consensus({"a": 2})
        self.assertEqual(rec["score_std"], 0.0)

    def test_all_judges_failing_to_parse_is_distinct_from_a_low_score(self):
        # This is the distinction that matters: no_score must never be read as
        # "the panel judged this inaccurate".
        rec = consensus({"a": None, "b": None, "c": None})
        self.assertIsNone(rec["mean_score"])
        self.assertEqual(rec["consensus_status"], "parse_error")
        self.assertEqual(rec["judge_verdict"], "no_score")

    def test_partial_parse_failure_uses_the_judges_that_did_parse(self):
        rec = consensus({"a": 3, "b": None, "c": 3})
        self.assertEqual(rec["mean_score"], 3)
        self.assertEqual(rec["judge_verdict"], "accurate")

    def test_verdict_threshold_is_2_5(self):
        self.assertEqual(consensus({"a": 3, "b": 2})["judge_verdict"], "accurate")   # 2.5
        self.assertEqual(consensus({"a": 2, "b": 2})["judge_verdict"], "inaccurate")  # 2.0

    def test_disagreement_above_the_threshold_is_flagged(self):
        spread = consensus({"a": 1, "b": 3, "c": 3})
        self.assertGreater(spread["score_std"], STD_FLAG_THRESHOLD)
        self.assertEqual(spread["consensus_status"], "flagged_for_review")

    def test_agreement_is_not_flagged(self):
        self.assertEqual(consensus({"a": 3, "b": 3})["consensus_status"], "consensus")


class TestFieldConsensus(unittest.TestCase):

    def _votes(self, **per_field):
        return {fk: per_field.get(fk, {}) for fk in FIELD_KEYS}

    def test_unanimous_true(self):
        rec = consensus({"a": 3}, self._votes(state_correct={"a": True, "b": True}))
        self.assertTrue(rec["field_consensus"]["state_correct"])

    def test_majority_wins(self):
        rec = consensus({"a": 3},
                        self._votes(state_correct={"a": True, "b": True, "c": False}))
        self.assertTrue(rec["field_consensus"]["state_correct"])

    def test_minority_loses(self):
        rec = consensus({"a": 3},
                        self._votes(state_correct={"a": True, "b": False, "c": False}))
        self.assertFalse(rec["field_consensus"]["state_correct"])

    def test_an_even_split_resolves_to_False(self):
        # Threshold is len/2 + 0.5, so 1-of-2 does NOT carry. A tie is treated
        # as "not established" rather than credited as correct.
        rec = consensus({"a": 3}, self._votes(state_correct={"a": True, "b": False}))
        self.assertFalse(rec["field_consensus"]["state_correct"])

    def test_no_votes_for_a_field_gives_None_not_False(self):
        # None means "no judge could assess it"; False means "judges said no".
        # Conflating them would count unassessable fields as failures.
        rec = consensus({"a": 3}, self._votes())
        self.assertIsNone(rec["field_consensus"]["state_correct"])

    def test_all_four_fields_are_present(self):
        rec = consensus({"a": 3}, self._votes(state_correct={"a": True}))
        self.assertEqual(set(rec["field_consensus"]), set(FIELD_KEYS))

    def test_field_keys_omitted_entirely_when_no_votes_supplied(self):
        self.assertNotIn("field_consensus", consensus({"a": 3}))


class TestHallucinationUnion(unittest.TestCase):

    def test_unions_across_judges_and_deduplicates(self):
        rec = compute_consensus(
            image="a/1.jpg", gt_state="aground", pred_text="t",
            verbosity_flagged=False,
            scores={"a": 3, "b": 3},
            rationales={"a": "", "b": ""},
            hallucinations={"a": ["ghost crane", "extra mast"],
                            "b": ["ghost crane"]})
        self.assertEqual(sorted(rec["hallucination_union"]),
                         ["extra mast", "ghost crane"])

    def test_none_from_a_judge_does_not_crash(self):
        rec = compute_consensus(
            image="a/1.jpg", gt_state="aground", pred_text="t",
            verbosity_flagged=False, scores={"a": 3},
            rationales={"a": ""}, hallucinations={"a": None})
        self.assertEqual(rec["hallucination_union"], [])


class TestRecordShape(unittest.TestCase):

    def test_record_id_matches_the_loader_key_format(self):
        # aggregate writes record_id; visual_classification's merge looks up by
        # the same composite. If the two ever diverged the merge would silently
        # match nothing and every judge column would come back empty.
        rec = consensus({"a": 3}, model_tag="llava", method="degf",
                        prompt_stem="p1")
        self.assertEqual(rec["record_id"],
                         _record_id({"image": "a/1.jpg", "model_tag": "llava",
                                     "method": "degf", "prompt_stem": "p1"}))

    def test_carries_the_join_fields(self):
        rec = consensus({"a": 3}, model_tag="llava", method="degf",
                        prompt_stem="p1")
        for k in ("image", "model_tag", "method", "prompt_stem", "gt_state"):
            self.assertIn(k, rec)

    def test_three_judge_models_are_configured(self):
        self.assertEqual(len(JUDGE_MODELS), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
