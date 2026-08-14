#!/usr/bin/env python3
"""Tests for visual_classification/aggregate_report.py.

    python tests/test_aggregate_report.py

Characterization tests written before the OOP extraction, then kept as the
regression suite. Focused on outcome tiering and the statistics — the parts
that carry research meaning. The _*_html helpers are presentation and are
covered only through the smoke test at the end.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visual_classification"))

import aggregate_report as ar


def row(image, correct, model="llava", method="baseline", prompt="p1", gt="aground"):
    """One per_record.csv row."""
    return {"image": image, "regex_correct": correct, "gt_label": gt,
            "model_tag": model, "method": method, "prompt_stem": prompt}


class TestBoolCoercion(unittest.TestCase):
    """CSV round-trips lose types; "False" is a truthy string."""

    def test_truthy_spellings(self):
        for v in (True, "True", "true", "1", "yes", "YES"):
            self.assertTrue(ar._bool(v), repr(v))

    def test_falsy_spellings(self):
        for v in (False, "False", "false", "0", "no", "", "  "):
            self.assertFalse(ar._bool(v), repr(v))

    def test_unknown_values_are_false_not_an_error(self):
        # A blank or absent column must read as False rather than raising.
        self.assertFalse(ar._bool(None))
        self.assertFalse(ar._bool("maybe"))


class TestTierClassification(unittest.TestCase):

    def _tier_for(self, rows):
        return ar.compute_per_image_tiers(rows)[0]

    def test_tier1_every_combination_correct(self):
        rows = [row("a/1.jpg", True, method=m) for m in ("baseline", "degf", "only")]
        out = self._tier_for(rows)
        self.assertEqual(out["tier"], 1)
        self.assertEqual(out["consensus"], "all_correct")
        self.assertEqual(out["difficulty_score"], 0)

    def test_tier2_every_combination_wrong(self):
        rows = [row("a/1.jpg", False, method=m) for m in ("baseline", "degf", "only")]
        out = self._tier_for(rows)
        self.assertEqual(out["tier"], 2)
        self.assertEqual(out["consensus"], "all_wrong")
        self.assertEqual(out["difficulty_score"], 1)

    def test_tier3_contested(self):
        rows = [row("a/1.jpg", True, method="baseline"),
                row("a/1.jpg", False, method="degf")]
        out = self._tier_for(rows)
        self.assertEqual(out["tier"], 3)
        self.assertEqual(out["consensus"], "contested")

    def test_images_without_ground_truth_are_untiered(self):
        # Not tier 2 — an image with no GT is unscored, not universally failed.
        # Conflating them would inflate the apparent failure count.
        rows = [row("a/1.jpg", False, gt=""), row("a/1.jpg", False, gt="")]
        out = self._tier_for(rows)
        self.assertIsNone(out["tier"])
        self.assertEqual(out["consensus"], "no_gt")
        self.assertIsNone(out["difficulty_score"])

    def test_difficulty_is_the_fraction_wrong(self):
        rows = [row("a/1.jpg", True), row("a/1.jpg", True),
                row("a/1.jpg", False), row("a/1.jpg", False)]
        self.assertAlmostEqual(self._tier_for(rows)["difficulty_score"], 0.5)

    def test_images_are_grouped_independently(self):
        rows = [row("a/1.jpg", True), row("b/2.jpg", False)]
        out = {r["image"]: r for r in ar.compute_per_image_tiers(rows)}
        self.assertEqual(out["a/1.jpg"]["tier"], 1)
        self.assertEqual(out["b/2.jpg"]["tier"], 2)

    def test_string_booleans_from_csv_are_honoured(self):
        # The realistic path: values arrive as strings, not bools.
        rows = [row("a/1.jpg", "True"), row("a/1.jpg", "True")]
        self.assertEqual(self._tier_for(rows)["tier"], 1)


class TestTierSubTypes(unittest.TestCase):
    """Tier 3 sub-types are multi-label: an image can split on several axes."""

    def _sub_types(self, rows):
        return ar.compute_per_image_tiers(rows)[0].get("sub_types", "")

    def test_model_split_when_only_the_model_differs(self):
        rows = [row("a/1.jpg", True,  model="llava"),
                row("a/1.jpg", False, model="qwen")]
        self.assertIn("model_split", self._sub_types(rows))

    def test_method_split_when_only_the_method_differs(self):
        rows = [row("a/1.jpg", True,  method="baseline"),
                row("a/1.jpg", False, method="degf")]
        self.assertIn("method_split", self._sub_types(rows))

    def test_method_regression_when_a_method_loses_to_baseline(self):
        # The finding that matters: the mitigation made this image worse.
        rows = [row("a/1.jpg", True,  method="baseline"),
                row("a/1.jpg", False, method="degf")]
        self.assertIn("method_regression", self._sub_types(rows))

    def test_prompt_split_needs_differing_LABELS_not_just_correctness(self):
        # Asymmetry worth knowing: model_split and method_split key on
        # correctness differing, but prompt_split keys on the predicted LABEL
        # differing. It detects label instability across prompts, which is a
        # different phenomenon from one prompt happening to be right.
        a = row("a/1.jpg", True,  prompt="p1"); a["parsed_label"] = "aground"
        b = row("a/1.jpg", False, prompt="p2"); b["parsed_label"] = "sunken"
        self.assertIn("prompt_split", self._sub_types([a, b]))

    def test_differing_correctness_alone_is_not_a_prompt_split(self):
        # Without parsed_label there is no evidence of label instability, so
        # this falls through to the catch-all rather than being mislabelled.
        rows = [row("a/1.jpg", True,  prompt="p1"),
                row("a/1.jpg", False, prompt="p2")]
        self.assertIn("combo_split", self._sub_types(rows))

    def test_tier1_and_tier2_carry_no_sub_types(self):
        self.assertFalse(self._sub_types([row("a/1.jpg", True)]))
        self.assertFalse(self._sub_types([row("a/1.jpg", False)]))


class TestOutcomeTierClass(unittest.TestCase):
    """The classification rule in isolation."""

    def test_classify_covers_each_case(self):
        self.assertEqual(ar.OutcomeTier.classify(3, 3),
                         (ar.OutcomeTier.ALL_CORRECT, "all_correct"))
        self.assertEqual(ar.OutcomeTier.classify(0, 3),
                         (ar.OutcomeTier.ALL_WRONG, "all_wrong"))
        self.assertEqual(ar.OutcomeTier.classify(1, 3),
                         (ar.OutcomeTier.CONTESTED, "contested"))

    def test_no_ground_truth_is_not_a_tier(self):
        tier, consensus = ar.OutcomeTier.classify(0, 0)
        self.assertIsNone(tier)
        self.assertEqual(consensus, "no_gt")

    def test_difficulty_is_none_not_zero_without_gt(self):
        # 0.0 would read as "every combination was right" — the opposite of
        # "unknown".
        self.assertIsNone(ar.OutcomeTier.difficulty(0, 0))
        self.assertEqual(ar.OutcomeTier.difficulty(3, 3), 0)
        self.assertEqual(ar.OutcomeTier.difficulty(0, 3), 1)

    def test_tier_numbers_are_stable(self):
        # These are written to CSVs and read by downstream tooling.
        self.assertEqual(ar.OutcomeTier.ALL_CORRECT, 1)
        self.assertEqual(ar.OutcomeTier.ALL_WRONG, 2)
        self.assertEqual(ar.OutcomeTier.CONTESTED, 3)

    def test_subtype_names_are_stable(self):
        self.assertEqual(ar.SubType.MODEL_SPLIT, "model_split")
        self.assertEqual(ar.SubType.METHOD_REGRESSION, "method_regression")
        self.assertEqual(ar.SubType.COMBO_SPLIT, "combo_split")


class TestEmptyInput(unittest.TestCase):

    def test_no_rows_yields_no_results(self):
        self.assertEqual(ar.compute_per_image_tiers([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
