#!/usr/bin/env python3
"""Tests for Eval_CASTOR P1 run discovery.

    python tests/test_run_discovery.py

discover_runs() decides two things about every inference file it finds, and
both are HEURISTICS rather than recorded facts:

  * whether the run used diffusion — inferred from the FILENAME
  * whether the prompt was chain-of-thought — inferred from the CONTENT of the
    first record

Getting either wrong does not raise. It silently mislabels a run, and the
mislabel propagates into every comparison table built from it. These tests pin
the current behaviour, including where the heuristics are fragile.

Pure file I/O: no cluster, no models.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent / "Eval_CASTOR"
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.eval_castor import discover_runs      # noqa: E402


class DiscoveryCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name, text="The vessel is aground."):
        (self.dir / name).write_text(
            json.dumps({"image": "aground/1.jpg", "text": text}) + "\n",
            encoding="utf-8")

    def discover(self):
        return {r[0]: r for r in discover_runs(self.dir)}


class TestDiffusionDetection(DiscoveryCase):

    def test_degf_in_the_filename_marks_diffusion(self):
        self.write("answers_degf_promptv4.jsonl")
        self.assertTrue(self.discover()["answers_degf_promptv4.jsonl"][1])

    def test_baseline_filename_does_not(self):
        self.write("answers_baseline_promptv4.jsonl")
        self.assertFalse(self.discover()["answers_baseline_promptv4.jsonl"][1])

    def test_detection_is_case_insensitive(self):
        self.write("answers_DeGF_run.jsonl")
        self.assertTrue(self.discover()["answers_DeGF_run.jsonl"][1])

    def test_only_runs_are_not_marked_diffusion(self):
        # ONLY needs no diffusion — that is its whole efficiency claim.
        self.write("answers_only_promptv4.jsonl")
        self.assertFalse(self.discover()["answers_only_promptv4.jsonl"][1])

    def test_FRAGILE_substring_match_can_mislabel(self):
        # Documents a real limitation: the check is a substring test on the
        # filename, so any name containing "degf" is marked as a diffusion run
        # regardless of what it actually contains. A baseline file named for
        # comparison against DeGF is mislabelled, silently.
        self.write("answers_baseline_vs_degf.jsonl")
        self.assertTrue(self.discover()["answers_baseline_vs_degf.jsonl"][1])


class TestPromptStyleDetection(DiscoveryCase):

    def test_plain_output_is_direct(self):
        self.write("answers_a.jsonl", "The vessel is aground on rocks.")
        self.assertEqual(self.discover()["answers_a.jsonl"][2], "direct")

    def test_step_markers_indicate_cot(self):
        self.write("answers_b.jsonl", "Step 1 — Evidence Catalog. The hull ...")
        self.assertEqual(self.discover()["answers_b.jsonl"][2], "cot")

    def test_step_2_also_counts(self):
        self.write("answers_c.jsonl", "Step 2: Visual grounding questions ...")
        self.assertEqual(self.discover()["answers_c.jsonl"][2], "cot")

    def test_detection_is_case_insensitive(self):
        self.write("answers_d.jsonl", "STEP 1 evidence")
        self.assertEqual(self.discover()["answers_d.jsonl"][2], "cot")

    def test_only_the_FIRST_record_is_inspected(self):
        # Style is decided from record one and applied to the whole file. A run
        # whose first answer happens to omit the step markers is labelled
        # direct even if every later record has them.
        p = self.dir / "answers_e.jsonl"
        p.write_text(
            json.dumps({"image": "a/1.jpg", "text": "Aground."}) + "\n" +
            json.dumps({"image": "a/2.jpg", "text": "Step 1 — evidence"}) + "\n",
            encoding="utf-8")
        self.assertEqual(self.discover()["answers_e.jsonl"][2], "direct")

    def test_later_step_mention_still_counts_within_the_first_record(self):
        self.write("answers_f.jsonl", "Here is my answer. Step 1 was ...")
        self.assertEqual(self.discover()["answers_f.jsonl"][2], "cot")


class TestDiscoveryMechanics(DiscoveryCase):

    def test_returns_one_entry_per_jsonl(self):
        self.write("a.jsonl"); self.write("b.jsonl")
        self.assertEqual(len(discover_runs(self.dir)), 2)

    def test_non_jsonl_files_are_ignored(self):
        self.write("a.jsonl")
        (self.dir / "notes.txt").write_text("ignore me", encoding="utf-8")
        (self.dir / "summary.csv").write_text("a,b", encoding="utf-8")
        self.assertEqual(len(discover_runs(self.dir)), 1)

    def test_results_are_sorted_for_determinism(self):
        for name in ("c.jsonl", "a.jsonl", "b.jsonl"):
            self.write(name)
        self.assertEqual([r[0] for r in discover_runs(self.dir)],
                         ["a.jsonl", "b.jsonl", "c.jsonl"])

    def test_empty_directory(self):
        self.assertEqual(discover_runs(self.dir), [])

    def test_unreadable_first_line_defaults_to_direct(self):
        # A truncated or malformed file must not abort discovery of the rest.
        (self.dir / "broken.jsonl").write_text('{"image": "a/1.jpg", "tex',
                                               encoding="utf-8")
        runs = self.discover()
        self.assertIn("broken.jsonl", runs)
        self.assertEqual(runs["broken.jsonl"][2], "direct")

    def test_empty_file_does_not_abort_discovery(self):
        (self.dir / "empty.jsonl").write_text("", encoding="utf-8")
        self.write("good.jsonl")
        self.assertEqual(len(discover_runs(self.dir)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
