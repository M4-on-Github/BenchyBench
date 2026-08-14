#!/usr/bin/env python3
"""Tests for P6 salvage-plan templating statistics and P2 extraction resume.

    python tests/test_salvage_stats.py

P6 asks whether a model's salvage plans are TEMPLATED — whether it reaches for
the same elements regardless of what casualty it thinks it is looking at. That
is answered with Fisher tests per element per state, so the multiple-comparison
correction is doing real work: with dozens of elements across four states,
uncorrected p-values would manufacture significant findings from noise.

P2's resume logic decides which records are re-sent to Ollama. Getting it wrong
either re-does completed work or, worse, silently skips records that failed.

Pure statistics and file handling: no Ollama, no cluster.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent / "Eval_CASTOR"
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.eval_salvage_plan import (          # noqa: E402
    _safe_pct, apply_fdr_correction, SIGNIFICANCE_THRESHOLD,
)
from pipelines.extract_gemma import (              # noqa: E402
    discover_runs, load_existing_output,
)


class FakeTest:
    """Stand-in for the element-test record apply_fdr_correction mutates."""
    def __init__(self, element, state, state_source, p_value):
        self.element = element
        self.state = state
        self.state_source = state_source
        self.p_value = p_value
        self.p_corrected = None


class TestSafePct(unittest.TestCase):

    def test_ordinary_proportion(self):
        self.assertAlmostEqual(_safe_pct(3, 4), 0.75)

    def test_zero_denominator_is_zero_not_a_crash(self):
        # A state with no images must not abort the whole report.
        self.assertEqual(_safe_pct(0, 0), 0.0)

    def test_zero_numerator(self):
        self.assertEqual(_safe_pct(0, 10), 0.0)


class TestFdrCorrection(unittest.TestCase):
    """Benjamini-Hochberg, applied per (state_source, state) group."""

    def test_correction_is_applied(self):
        tests = [FakeTest("tug", "aground", "predicted", 0.01),
                 FakeTest("dive", "aground", "predicted", 0.04)]
        apply_fdr_correction(tests)
        for t in tests:
            self.assertIsNotNone(t.p_corrected)

    def test_corrected_p_is_never_smaller_than_raw(self):
        # BH only ever inflates a p-value; a correction that shrank one would
        # manufacture significance.
        tests = [FakeTest("a", "aground", "predicted", 0.01),
                 FakeTest("b", "aground", "predicted", 0.02),
                 FakeTest("c", "aground", "predicted", 0.03)]
        apply_fdr_correction(tests)
        for t in tests:
            self.assertGreaterEqual(t.p_corrected, t.p_value)

    def test_groups_are_corrected_INDEPENDENTLY(self):
        # The design decision this pipeline documents: each (source, state)
        # pair gets its own FDR budget. Pooling them would penalise every
        # finding for tests answering a different question.
        #
        # One group of three, one of one. The lone test in its own group is
        # corrected as a family of one, so its p-value is unchanged.
        tests = [FakeTest("a", "aground", "predicted", 0.04),
                 FakeTest("b", "aground", "predicted", 0.04),
                 FakeTest("c", "aground", "predicted", 0.04),
                 FakeTest("d", "sunken", "predicted", 0.04)]
        apply_fdr_correction(tests)
        lone = [t for t in tests if t.state == "sunken"][0]
        self.assertAlmostEqual(lone.p_corrected, 0.04)

    def test_predicted_and_gt_tracks_are_separate_families(self):
        # predicted_state and gt_state answer different questions and are
        # reported as independent findings, so they are not pooled.
        tests = [FakeTest("a", "aground", "predicted", 0.04),
                 FakeTest("a", "aground", "gt", 0.04)]
        apply_fdr_correction(tests)
        for t in tests:
            self.assertAlmostEqual(t.p_corrected, 0.04)

    def test_empty_input_is_returned_unchanged(self):
        self.assertEqual(apply_fdr_correction([]), [])

    def test_significance_threshold_is_the_conventional_five_percent(self):
        self.assertEqual(SIGNIFICANCE_THRESHOLD, 0.05)


class TestExtractionResume(unittest.TestCase):
    """P2 re-sends only what has not already succeeded."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write_out(self, records):
        p = self.dir / "out.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        return p

    def test_successful_records_are_skipped_on_resume(self):
        p = self.write_out([{"image": "a/1.jpg", "gemma_parse_ok": True}])
        self.assertEqual(load_existing_output(p), {"a/1.jpg"})

    def test_FAILED_records_are_retried(self):
        # The important half: a record that failed extraction must NOT count
        # as done, or a transient Ollama error would permanently drop it.
        p = self.write_out([{"image": "a/1.jpg", "gemma_parse_ok": False},
                            {"image": "b/2.jpg", "gemma_parse_ok": True}])
        self.assertEqual(load_existing_output(p), {"b/2.jpg"})

    def test_records_missing_the_ok_flag_are_retried(self):
        p = self.write_out([{"image": "a/1.jpg"}])
        self.assertEqual(load_existing_output(p), set())

    def test_absent_output_file_means_nothing_is_done(self):
        self.assertEqual(load_existing_output(self.dir / "nope.jsonl"), set())

    def test_malformed_lines_do_not_abort_the_resume_scan(self):
        p = self.dir / "out.jsonl"
        p.write_text(
            json.dumps({"image": "a/1.jpg", "gemma_parse_ok": True}) + "\n"
            + '{"image": "b/2.jpg", "trunca' + "\n"
            + json.dumps({"image": "c/3.jpg", "gemma_parse_ok": True}) + "\n",
            encoding="utf-8")
        self.assertEqual(load_existing_output(p), {"a/1.jpg", "c/3.jpg"})


class TestExtractionDiscovery(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def touch(self, name):
        (self.dir / name).write_text("{}", encoding="utf-8")

    def test_finds_all_jsonl_sorted(self):
        for n in ("c.jsonl", "a.jsonl", "b.jsonl"):
            self.touch(n)
        self.assertEqual([p.name for p in discover_runs(self.dir)],
                         ["a.jsonl", "b.jsonl", "c.jsonl"])

    def test_filter_restricts_to_named_files(self):
        self.touch("a.jsonl"); self.touch("b.jsonl")
        self.assertEqual([p.name for p in discover_runs(self.dir, ["b.jsonl"])],
                         ["b.jsonl"])

    def test_empty_filter_list_is_treated_as_no_filter(self):
        # `if filter_names and ...` — an empty list is falsy, so everything
        # passes rather than nothing.
        self.touch("a.jsonl")
        self.assertEqual(len(discover_runs(self.dir, [])), 1)

    def test_non_jsonl_ignored(self):
        self.touch("a.jsonl")
        (self.dir / "notes.txt").write_text("x", encoding="utf-8")
        self.assertEqual(len(discover_runs(self.dir)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
