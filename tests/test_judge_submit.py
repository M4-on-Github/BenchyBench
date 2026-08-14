#!/usr/bin/env python3
"""Tests for visual_classification/judge_submit.py.

    python tests/test_judge_submit.py

Includes a Python 3.6 compatibility guard. Unlike its sibling scripts this
module runs on the node's bare python3 (3.6), not inside castor.sif (3.10).
Two separate outages came from forgetting that — a `text=True` kwarg and, in
the shell wrapper, a swallowed error that hid it. The guard makes the
constraint enforceable rather than remembered.
"""

import ast
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

VC_DIR = Path(__file__).resolve().parent.parent / "visual_classification"
sys.path.insert(0, str(VC_DIR))

import judge_submit as js


class TestPython36Compatibility(unittest.TestCase):
    """Static checks for syntax and APIs unavailable on Python 3.6."""

    @classmethod
    def setUpClass(cls):
        cls.source = io.open(str(VC_DIR / "judge_submit.py"), encoding="utf-8").read()
        cls.tree = ast.parse(cls.source)

    def test_does_not_import_dataclasses(self):
        """dataclasses landed in 3.7."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.assertNotEqual(a.name, "dataclasses")
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "dataclasses")

    def test_no_walrus_operator(self):
        """:= landed in 3.8."""
        named = [n for n in ast.walk(self.tree)
                 if isinstance(n, getattr(ast, "NamedExpr", ()))]
        self.assertEqual(named, [], "walrus operator is not valid on Python 3.6")

    def test_subprocess_run_does_not_use_text_kwarg(self):
        """text= landed in 3.7; universal_newlines is the 3.6 spelling.

        This is the exact bug that broke the judge phase once already.
        """
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "text":
                        self.fail("subprocess text= kwarg is not valid on Python 3.6")
        self.assertIn("universal_newlines", self.source)

    def test_no_fstring_equals_specifier(self):
        """f'{x=}' landed in 3.8."""
        self.assertNotIn("=}", self.source)


class TestBuildRecords(unittest.TestCase):

    def _row(self, image, model="llava", method="baseline", prompt="p1", text="out"):
        return {"image": image, "raw_text": text, "model_tag": model,
                "method": method, "prompt_stem": prompt, "question_id": "1"}

    def test_extracts_the_fields_the_judge_needs(self):
        rec = js.JudgeSubmission.build_records([self._row("aground/1.jpg")])[0]
        self.assertEqual(rec["image"], "aground/1.jpg")
        self.assertEqual(rec["text"], "out")          # raw_text -> text
        self.assertEqual(rec["model_tag"], "llava")

    def test_same_image_across_combinations_is_kept(self):
        # The key subtlety: keying on image alone would collapse every method
        # into a single verdict and discard all but the last.
        rows = [self._row("a/1.jpg", method="baseline"),
                self._row("a/1.jpg", method="degf"),
                self._row("a/1.jpg", method="only")]
        self.assertEqual(len(js.JudgeSubmission.build_records(rows)), 3)

    def test_exact_duplicates_are_dropped(self):
        rows = [self._row("a/1.jpg"), self._row("a/1.jpg")]
        self.assertEqual(len(js.JudgeSubmission.build_records(rows)), 1)

    def test_distinguishes_on_every_key_field(self):
        for field, value in [("model", "qwen"), ("method", "degf"), ("prompt", "p2")]:
            kw = {field: value}
            rows = [self._row("a/1.jpg"), self._row("a/1.jpg", **kw)]
            self.assertEqual(len(js.JudgeSubmission.build_records(rows)), 2, field)

    def test_missing_columns_become_empty_strings(self):
        rec = js.JudgeSubmission.build_records([{"image": "a/1.jpg"}])[0]
        self.assertEqual(rec["text"], "")
        self.assertEqual(rec["method"], "")

    def test_empty_input(self):
        self.assertEqual(js.JudgeSubmission.build_records([]), [])


class TestParseAggJobId(unittest.TestCase):

    def test_returns_the_last_job_id(self):
        # The submit script prints one job per judge model, then the
        # aggregation job that depends on them. The last one is what chains.
        out = "submitted job=1001\nsubmitted job=1002\nagg job=1003\n"
        self.assertEqual(js.JudgeSubmission.parse_agg_job_id(out), "1003")

    def test_returns_none_when_absent(self):
        self.assertIsNone(js.JudgeSubmission.parse_agg_job_id("nothing here"))
        self.assertIsNone(js.JudgeSubmission.parse_agg_job_id(""))

    def test_ignores_similar_looking_text(self):
        self.assertIsNone(js.JudgeSubmission.parse_agg_job_id("jobs=12 job_id 55"))


class TestPaths(unittest.TestCase):

    def setUp(self):
        self.sub = js.JudgeSubmission("/out/run01", "/bb", "vc_run01")

    def test_paths_derive_from_the_three_inputs(self):
        self.assertEqual(self.sub.eval_castor_root, Path("/bb/Eval_CASTOR"))
        self.assertEqual(self.sub.per_record_path,
                         Path("/out/run01/eval/regex/per_record.csv"))
        self.assertEqual(self.sub.judge_input_path,
                         Path("/bb/Eval_CASTOR/p5_to_judge/vc_run01.jsonl"))
        self.assertEqual(self.sub.judge_output_dir,
                         Path("/bb/Eval_CASTOR/results/p5_judge/vc_run01"))

    def test_run_name_flows_into_input_and_output(self):
        other = js.JudgeSubmission("/out/run01", "/bb", "other_run")
        self.assertNotEqual(self.sub.judge_input_path, other.judge_input_path)
        self.assertNotEqual(self.sub.judge_output_dir, other.judge_output_dir)


class TestCommand(unittest.TestCase):

    def setUp(self):
        self.sub = js.JudgeSubmission("/out", "/bb", "run01")

    def test_minimal_command(self):
        cmd = self.sub.build_command()
        self.assertEqual(cmd[0], "bash")
        self.assertIn("--run", cmd)
        self.assertIn("run01", cmd)
        self.assertNotIn("--limit", cmd)

    def test_limit_and_dry_run_are_forwarded(self):
        cmd = self.sub.build_command(limit=5, dry_run=True)
        self.assertIn("--limit", cmd)
        self.assertIn("5", cmd)
        self.assertIn("--dry-run", cmd)


class TestWriteInput(unittest.TestCase):

    def test_writes_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = js.JudgeSubmission(tmp, tmp, "run01")
            sub.write_input([{"image": "a/1.jpg", "text": "x"},
                             {"image": "b/2.jpg", "text": "y"}])
            lines = [l for l in sub.judge_input_path.read_text(
                encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 2)

    def test_creates_the_staging_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = js.JudgeSubmission(tmp, tmp, "run01")
            self.assertFalse(sub.judge_input_path.parent.exists())
            sub.write_input([{"image": "a/1.jpg", "text": "x"}])
            self.assertTrue(sub.judge_input_path.exists())


class TestLoadPerRecordCsv(unittest.TestCase):

    def test_reads_rows_as_dicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "per_record.csv"
            with p.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["image", "raw_text"])
                w.writeheader()
                w.writerow({"image": "a/1.jpg", "raw_text": "aground"})
            rows = js.load_per_record_csv(p)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["image"], "a/1.jpg")


if __name__ == "__main__":
    unittest.main(verbosity=2)
