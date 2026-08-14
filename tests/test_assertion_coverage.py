#!/usr/bin/env python3
"""Tests for Eval_CASTOR P7 assertion selection and contamination scanning.

    python tests/test_assertion_coverage.py

P7 asks two questions about a salvage plan:

  coverage       does it mention the domain concepts that APPLY to this
                 casualty type?
  contamination  does it mention concepts specific to a DIFFERENT casualty
                 type? That is a hallucination signal — a plan for an aground
                 vessel discussing dive teams and depth is describing a wreck
                 it was not shown.

Both rest on which assertions are selected for an image, so a selection error
changes the finding without changing anything visible.

Pure data handling: the vLLM coverage judge is not exercised, only the
selection and keyword logic around it.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent / "Eval_CASTOR"
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.assertion_coverage.check_assertions import (    # noqa: E402
    load_registry, relevant_assertions, contamination_assertions,
    keyword_hit, scan_contamination, UNIVERSAL_TYPES,
)


def assertion(aid, casualty_type, keywords):
    return {"id": aid, "casualty_type": casualty_type,
            "keywords": [k.lower() for k in keywords]}


REGISTRY = [
    assertion("AG1", "aground", ["tide", "refloat"]),
    assertion("SU1", "sunken", ["dive team", "depth"]),
    assertion("FIRE1", "on_fire", ["boundary cooling"]),
    assertion("CAP1", "capsized", ["righting moment"]),
    assertion("RES1", "resources", ["tug"]),
    assertion("CC1", "cross-cutting", ["safety briefing"]),
]


class TestRelevantAssertions(unittest.TestCase):

    def test_includes_the_matching_casualty_type(self):
        ids = [a["id"] for a in relevant_assertions(REGISTRY, "aground")]
        self.assertIn("AG1", ids)

    def test_excludes_other_casualty_types(self):
        ids = [a["id"] for a in relevant_assertions(REGISTRY, "aground")]
        self.assertNotIn("SU1", ids)
        self.assertNotIn("FIRE1", ids)

    def test_universal_types_always_apply(self):
        # Resources and cross-cutting concerns are relevant to every casualty,
        # so they are checked regardless of state.
        for state in ("aground", "sunken", "on_fire", "capsized"):
            ids = [a["id"] for a in relevant_assertions(REGISTRY, state)]
            self.assertIn("RES1", ids, state)
            self.assertIn("CC1", ids, state)

    def test_universal_types_are_declared(self):
        self.assertEqual(UNIVERSAL_TYPES, {"resources", "cross-cutting"})

    def test_unknown_state_still_yields_the_universal_ones(self):
        ids = [a["id"] for a in relevant_assertions(REGISTRY, "not_a_state")]
        self.assertEqual(sorted(ids), ["CC1", "RES1"])


class TestContaminationAssertions(unittest.TestCase):

    def test_selects_the_other_three_casualty_types(self):
        ids = sorted(a["id"] for a in contamination_assertions(REGISTRY, "aground"))
        self.assertEqual(ids, ["CAP1", "FIRE1", "SU1"])

    def test_excludes_the_correct_type(self):
        ids = [a["id"] for a in contamination_assertions(REGISTRY, "aground")]
        self.assertNotIn("AG1", ids)

    def test_universal_types_are_NOT_contamination(self):
        # Mentioning tugs or a safety briefing is appropriate for any casualty,
        # so counting them as contamination would flag correct plans.
        ids = [a["id"] for a in contamination_assertions(REGISTRY, "aground")]
        self.assertNotIn("RES1", ids)
        self.assertNotIn("CC1", ids)

    def test_relevant_and_contamination_never_overlap(self):
        for state in ("aground", "sunken", "on_fire", "capsized"):
            rel = {a["id"] for a in relevant_assertions(REGISTRY, state)}
            con = {a["id"] for a in contamination_assertions(REGISTRY, state)}
            self.assertEqual(rel & con, set(), state)


class TestKeywordHit(unittest.TestCase):

    def test_matches_a_whole_word(self):
        self.assertTrue(keyword_hit("Monitor the tide before refloating", ["tide"]))

    def test_is_case_insensitive(self):
        self.assertTrue(keyword_hit("Monitor the TIDE", ["tide"]))

    def test_matches_a_multi_word_phrase(self):
        self.assertTrue(keyword_hit("Deploy the dive team at dawn", ["dive team"]))

    def test_does_NOT_match_inside_a_longer_word(self):
        # Word boundaries matter: "tidewater" is not a tide reference, and
        # substring matching would manufacture contamination hits.
        self.assertFalse(keyword_hit("The tidewater region", ["tide"]))

    def test_any_keyword_matching_is_enough(self):
        self.assertTrue(keyword_hit("Plan to refloat", ["tide", "refloat"]))

    def test_no_match(self):
        self.assertFalse(keyword_hit("Deploy a tug", ["tide", "refloat"]))

    def test_empty_keywords_never_match(self):
        self.assertFalse(keyword_hit("anything at all", []))

    def test_regex_characters_in_a_keyword_are_escaped_not_interpreted(self):
        # Keywords come from a CSV and are data, not patterns. Escaping means
        # "a.b" does not match "axb".
        self.assertFalse(keyword_hit("axb pump", ["a.b"]))
        self.assertTrue(keyword_hit("use the a.b valve", ["a.b"]))

    def test_LATENT_a_keyword_ending_in_punctuation_can_never_match(self):
        # The pattern is \b + escaped keyword + \b. A keyword whose last
        # character is non-word (e.g. "c++") has no word boundary after it, so
        # it silently never matches.
        #
        # Verified against IMPROVED_assertion_registry.csv: all 65 keywords
        # currently start and end with word characters, so nothing is affected
        # today. Recorded because adding a keyword like "c++" or "24/7" would
        # disable it with no error — it would simply never register a hit.
        self.assertFalse(keyword_hit("use a c++ pump", ["c++"]))


class TestScanContamination(unittest.TestCase):

    def test_reports_offending_assertion_ids(self):
        wrong = contamination_assertions(REGISTRY, "aground")
        hits, count = scan_contamination(
            "Deploy the dive team to assess depth.", wrong)
        self.assertIn("SU1", hits)
        self.assertEqual(count, len(hits))

    def test_clean_plan_has_no_hits(self):
        wrong = contamination_assertions(REGISTRY, "aground")
        hits, count = scan_contamination(
            "Wait for the tide, then refloat with a tug.", wrong)
        self.assertEqual(hits, [])
        self.assertEqual(count, 0)

    def test_counts_multiple_distinct_contaminations(self):
        wrong = contamination_assertions(REGISTRY, "aground")
        hits, count = scan_contamination(
            "Dive team at depth; apply boundary cooling.", wrong)
        self.assertEqual(count, 2)
        self.assertEqual(sorted(hits), ["FIRE1", "SU1"])

    def test_empty_plan(self):
        self.assertEqual(
            scan_contamination("", contamination_assertions(REGISTRY, "aground")),
            ([], 0))


class TestLoadRegistry(unittest.TestCase):

    def test_splits_keywords_on_slash_and_lowercases(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "reg.csv"
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["id", "casualty_type",
                                                  "checkable_keyword"])
                w.writeheader()
                w.writerow({"id": "AG1", "casualty_type": "aground",
                            "checkable_keyword": "Tide / ReFloat / "})
            reg = load_registry(p)
            self.assertEqual(reg[0]["keywords"], ["tide", "refloat"])

    def test_blank_keyword_segments_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "reg.csv"
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["id", "casualty_type",
                                                  "checkable_keyword"])
                w.writeheader()
                w.writerow({"id": "X", "casualty_type": "aground",
                            "checkable_keyword": "a //  / b"})
            self.assertEqual(load_registry(p)[0]["keywords"], ["a", "b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
