#!/usr/bin/env python3
"""Tests for visual_classification/regex_eval.py.

    python tests/test_regex_eval.py

Characterization tests written before the OOP refactor, then kept as the
regression suite. Focused on combination identity resolution, which carries a
three-level fallback chain that is easy to break and silent when it does: a
wrong combination key does not raise, it splits or merges groups and quietly
changes every downstream metric.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visual_classification"))

import regex_eval as re_eval


class TestExtractComboFields(unittest.TestCase):
    """Resolution of (model_tag, method, prompt_stem) for a record."""

    def test_sidecar_meta_is_preferred(self):
        rec = {"_source_stem": "answers_1", "model_tag": "ignored",
               "method": "ignored", "run_name": "x_y_ignored"}
        meta = {"answers_1": {"model": "llava", "method": "degf",
                              "prompt_stem": "promptv4"}}
        self.assertEqual(re_eval.extract_combo_fields(rec, meta),
                         ("llava", "degf", "promptv4"))

    def test_falls_back_to_record_fields_without_meta(self):
        rec = {"_source_stem": "missing", "model_tag": "qwen",
               "method": "only", "run_name": "qwen_only_promptv5"}
        self.assertEqual(re_eval.extract_combo_fields(rec, {}),
                         ("qwen", "only", "promptv5"))

    def test_model_tag_precedes_model_id(self):
        # model_id varies by checkpoint and would split one combination.
        rec = {"model_tag": "llava", "model_id": "llava-v1.5-7b-hf"}
        self.assertEqual(re_eval.extract_combo_fields(rec, {})[0], "llava")

    def test_model_id_used_when_tag_absent(self):
        self.assertEqual(
            re_eval.extract_combo_fields({"model_id": "qwen3vl"}, {})[0], "qwen3vl")

    def test_prompt_stem_parsed_from_run_name_as_last_resort(self):
        # run_name convention: {model}_{method}_{stem}
        rec = {"run_name": "llava_baseline_promptv3"}
        self.assertEqual(re_eval.extract_combo_fields(rec, {})[2], "promptv3")

    def test_short_run_name_is_used_whole(self):
        # Fewer than 3 parts means the convention does not hold; using the
        # last part would silently produce a wrong stem.
        self.assertEqual(re_eval.extract_combo_fields({"run_name": "exp1"}, {})[2],
                         "exp1")

    def test_everything_missing_yields_unknown(self):
        self.assertEqual(re_eval.extract_combo_fields({}, {}),
                         ("unknown", "unknown", "unknown"))

    def test_empty_meta_entry_does_not_mask_record_fields(self):
        # A meta file present but lacking keys must not blank the record's own.
        rec = {"_source_stem": "s", "model_tag": "llava", "method": "degf",
               "run_name": "a_b_p1"}
        self.assertEqual(re_eval.extract_combo_fields(rec, {"s": {}}),
                         ("llava", "degf", "p1"))


class TestLoadGt(unittest.TestCase):

    def test_reads_image_to_state_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_dir = root / "Eval_CASTOR" / "human_ground_truth_label"
            gt_dir.mkdir(parents=True)
            with (gt_dir / "human_gt.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["image", "state"])
                w.writeheader()
                w.writerow({"image": "aground/1.jpg", "state": "aground"})
                w.writerow({"image": "sunken/2.jpg", "state": "sunken"})
            gt = re_eval.load_gt(root)
            self.assertEqual(gt.get("aground/1.jpg"), "aground")
            self.assertEqual(gt.get("sunken/2.jpg"), "sunken")


class TestWriteCsv(unittest.TestCase):

    def test_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out" / "x.csv"
            re_eval.write_csv(p, [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
            rows = list(csv.DictReader(p.open(encoding="utf-8")))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["a"], "1")

    def test_empty_rows_still_creates_the_file(self):
        # A present-but-empty file distinguishes "ran, found nothing" from
        # "never ran" when reading a run directory later.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out" / "empty.csv"
            re_eval.write_csv(p, [])
            self.assertTrue(p.exists())


class TestRepetitionHelper(unittest.TestCase):
    """The detector must agree with health_check's, which gates the run."""

    def test_agrees_with_health_check(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                               / "visual_classification"))
        import health_check as hc
        looping = "the vessel is listing to port " * 4
        normal = ("The vessel appears aground near a rocky shore with visible "
                  "hull damage and no crew present on the upper deck areas.")
        for text in (looping, normal):
            # populate_health_flags consumes rows already shaped by
            # build_per_records, which sets raw_text_len.
            rows = [{"raw_text": text, "raw_text_len": len(text),
                     "parse_success": "True"}]
            re_eval.populate_health_flags(rows)
            self.assertEqual(
                str(rows[0]["repetition_detected"]).lower(),
                str(hc.detect_repetition(text)).lower(),
                "regex_eval and health_check disagree on: %r" % text[:40])


class TestComboIdentity(unittest.TestCase):
    """The class API behind extract_combo_fields."""

    def test_is_usable_as_a_grouping_key(self):
        # It is the key every downstream metric groups by, so equality and
        # hashing must be by value, not identity.
        a = re_eval.ComboIdentity("llava", "degf", "p1")
        b = re_eval.ComboIdentity("llava", "degf", "p1")
        c = re_eval.ComboIdentity("llava", "only", "p1")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len({a, b, c}), 2)

    def test_grouping_a_dict_by_identity(self):
        counts = {}
        for method in ("degf", "degf", "only"):
            key = re_eval.ComboIdentity("llava", method, "p1")
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(counts[re_eval.ComboIdentity("llava", "degf", "p1")], 2)

    def test_matches_the_facade(self):
        rec = {"model_tag": "qwen", "method": "only", "run_name": "a_b_p9"}
        self.assertEqual(re_eval.ComboIdentity.from_record(rec, {}).as_tuple(),
                         re_eval.extract_combo_fields(rec, {}))

    def test_repr_is_readable(self):
        self.assertIn("llava",
                      repr(re_eval.ComboIdentity("llava", "degf", "p1")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
