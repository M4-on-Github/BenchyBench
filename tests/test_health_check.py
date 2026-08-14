#!/usr/bin/env python3
"""Tests for visual_classification/health_check.py.

Written as characterization tests BEFORE the OOP refactor, so they pin the
existing behaviour and prove the restructure changed nothing. They are kept
afterwards as the regression suite.

    python tests/test_health_check.py

Runs locally: no cluster, no GPU, no model weights.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visual_classification"))

import health_check as hc


class TestStateDetection(unittest.TestCase):
    """normalize_state and the per-record detectors."""

    def test_recognises_each_state(self):
        for text, expected in [
            ("The vessel is aground on a sandbar.", "aground"),
            ("The ship has capsized completely.",   "capsized"),
            ("The vessel is on fire, smoke visible.", "on_fire"),
            ("A sunken wreck lies below.",          "sunken"),
        ]:
            self.assertEqual(hc.normalize_state(text), expected, text)

    def test_synonyms_map_to_canonical_state(self):
        self.assertEqual(hc.normalize_state("the hull is overturned"), "capsized")
        self.assertEqual(hc.normalize_state("the ship is ablaze"),     "on_fire")
        self.assertEqual(hc.normalize_state("it is submerged"),        "sunken")
        self.assertEqual(hc.normalize_state("ran aground"),            "aground")

    def test_returns_none_when_no_state_present(self):
        self.assertIsNone(hc.normalize_state("A photograph of calm water."))
        self.assertIsNone(hc.normalize_state(""))

    def test_matches_whole_words_only(self):
        # 'fire' must not match inside 'firefighter' / 'misfire'
        self.assertIsNone(hc.normalize_state("a misfired engine"))

    def test_first_match_wins_when_several_states_named(self):
        # Documented behaviour: hedged text still parses to a single label.
        # This is why hedging needs detecting separately.
        result = hc.normalize_state("possibly aground or capsized")
        self.assertIn(result, ("aground", "capsized"))
        self.assertIsNotNone(result)


class TestRepetition(unittest.TestCase):

    def test_detects_a_repeated_phrase(self):
        text = "the vessel is listing to port " * 4
        self.assertTrue(hc.detect_repetition(text))

    def test_normal_prose_is_not_flagged(self):
        text = ("The vessel appears aground near a rocky shore with visible "
                "hull damage and no crew present on the upper deck areas.")
        self.assertFalse(hc.detect_repetition(text))

    def test_text_too_short_to_repeat_is_not_flagged(self):
        # Fewer than ngram * threshold words cannot repeat that often.
        self.assertFalse(hc.detect_repetition("aground on rocks"))

    def test_threshold_is_respected(self):
        text = "alpha beta gamma delta epsilon " * 2      # 2 occurrences
        self.assertFalse(hc.detect_repetition(text, ngram=5, threshold=3))
        text = "alpha beta gamma delta epsilon " * 3      # 3 occurrences
        self.assertTrue(hc.detect_repetition(text, ngram=5, threshold=3))


class TestHedgeAndRefusal(unittest.TestCase):

    def test_hedge_requires_more_than_one_state(self):
        self.assertTrue(hc.detect_hedge("either aground or capsized"))
        self.assertFalse(hc.detect_hedge("clearly aground"))
        self.assertFalse(hc.detect_hedge("no state mentioned here"))

    def test_synonyms_of_one_state_are_not_a_hedge(self):
        # 'sunk' and 'submerged' are both 'sunken' — one state, not two.
        self.assertFalse(hc.detect_hedge("the sunk vessel is submerged"))

    def test_refusal_keywords(self):
        for text in ["I cannot determine the state",
                     "unable to determine from this image",
                     "It is unclear",
                     "there is no image provided"]:
            self.assertTrue(hc.detect_refusal(text), text)

    def test_confident_answer_is_not_a_refusal(self):
        self.assertFalse(hc.detect_refusal("The vessel is aground."))


class TestLoadRecords(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "inference").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name, lines):
        (self.root / "inference" / name).write_text("\n".join(lines), encoding="utf-8")

    def test_loads_records_and_tags_source(self):
        self._write("answers_llava_baseline_1.jsonl", [
            json.dumps({"image": "aground/1.jpg", "text": "aground"}),
            json.dumps({"image": "sunken/2.jpg",  "text": "sunken"}),
        ])
        records = hc.load_records(self.root)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["_source_stem"], "answers_llava_baseline_1")

    def test_malformed_lines_are_skipped_not_fatal(self):
        # A truncated write from an interrupted job must not block the health
        # check that exists to report it.
        self._write("answers_x_1.jsonl", [
            json.dumps({"image": "a/1.jpg", "text": "aground"}),
            '{"image": "b/2.jpg", "text": "trunca',
            json.dumps({"image": "c/3.jpg", "text": "sunken"}),
        ])
        self.assertEqual(len(hc.load_records(self.root)), 2)

    def test_blank_lines_are_ignored(self):
        self._write("answers_x_1.jsonl",
                    [json.dumps({"image": "a/1.jpg", "text": "aground"}), "", "  "])
        self.assertEqual(len(hc.load_records(self.root)), 1)

    def test_missing_directory_yields_no_records(self):
        empty = Path(self._tmp.name) / "nothing"
        (empty / "inference").mkdir(parents=True)
        self.assertEqual(hc.load_records(empty), [])


class TestAnnotateRecords(unittest.TestCase):

    def test_every_record_gains_all_health_flags(self):
        records = [{"text": "The vessel is aground."}]
        hc.annotate_records(records)
        flags = records[0]["_health"]
        for key in ("repetition_detected", "hedge_detected", "refusal_detected",
                    "length_anomaly", "parsed_label"):
            self.assertIn(key, flags)
        self.assertEqual(flags["parsed_label"], "aground")

    def test_length_anomaly_is_relative_to_the_whole_run(self):
        # One record far longer than the rest is the outlier; uniform-length
        # records produce no outliers at all.
        records = [{"text": "aground. " * 5} for _ in range(10)]
        records.append({"text": "aground. " * 500})
        hc.annotate_records(records)
        self.assertTrue(records[-1]["_health"]["length_anomaly"])
        self.assertFalse(records[0]["_health"]["length_anomaly"])

    def test_records_without_text_do_not_crash(self):
        # Failed inference writes {question_id, image, error} and no text.
        records = [{"image": "a/1.jpg", "error": "oom-skip"}]
        hc.annotate_records(records)
        self.assertIsNone(records[0]["_health"]["parsed_label"])


class TestPerComboStats(unittest.TestCase):

    def _records(self, n, label_text, method="baseline", prompt="p1"):
        recs = [{"model_tag": "llava", "_method": method, "_prompt_stem": prompt,
                 "image": "aground/%d.jpg" % i, "text": label_text}
                for i in range(n)]
        return hc.annotate_records(recs)

    def test_groups_by_model_method_prompt(self):
        recs = self._records(3, "aground") + self._records(3, "sunken", method="degf")
        stats = hc.per_combo_stats(recs)
        self.assertEqual(len(stats), 2)
        self.assertIn(("llava", "baseline", "p1"), stats)
        self.assertIn(("llava", "degf", "p1"), stats)

    def test_parse_fail_rate_counts_unparseable_records(self):
        recs = self._records(5, "aground") + self._records(5, "a calm harbour photo")
        stats = hc.per_combo_stats(recs)[("llava", "baseline", "p1")]
        self.assertEqual(stats["n_records"], 10)
        self.assertEqual(stats["n_parse_fail"], 5)
        self.assertAlmostEqual(stats["parse_fail_rate"], 0.5)

    def test_high_parse_failure_gates_the_run(self):
        recs = self._records(20, "a calm harbour photo")
        stats = hc.per_combo_stats(recs)[("llava", "baseline", "p1")]
        self.assertTrue(stats["GATE_FAIL"])

    def test_label_bias_warns_but_never_gates(self):
        # Bias may be a genuine property of an unbalanced image set, so it must
        # not fail the run on its own.
        recs = self._records(20, "aground")
        stats = hc.per_combo_stats(recs)[("llava", "baseline", "p1")]
        self.assertTrue(stats["label_bias"])
        self.assertFalse(stats["GATE_FAIL"])

    def test_small_samples_suppress_both_checks(self):
        # Neither a distribution nor a rate is meaningful on smoke-size runs.
        recs = self._records(3, "a calm harbour photo")
        stats = hc.per_combo_stats(recs)[("llava", "baseline", "p1")]
        self.assertFalse(stats["GATE_FAIL"])
        self.assertFalse(stats["label_bias"])

    def test_reports_the_dominant_label(self):
        recs = self._records(12, "aground")
        stats = hc.per_combo_stats(recs)[("llava", "baseline", "p1")]
        self.assertEqual(stats["bias_dominant"], "aground")

    def test_self_inconsistency_counts_images_labelled_two_ways(self):
        recs = hc.annotate_records([
            {"model_tag": "llava", "_method": "baseline", "_prompt_stem": "p1",
             "image": "aground/1.jpg", "text": "aground"},
            {"model_tag": "llava", "_method": "baseline", "_prompt_stem": "p1",
             "image": "aground/1.jpg", "text": "sunken"},
        ])
        stats = hc.per_combo_stats(recs)[("llava", "baseline", "p1")]
        self.assertEqual(stats["n_self_inconsistent_images"], 1)


class TestStateClassifier(unittest.TestCase):
    """The class API, which the free functions delegate to."""

    def test_accepts_an_alternative_synonym_map(self):
        # The point of encapsulating the map: a prompt eliciting different
        # vocabulary can be scored without mutating global state.
        custom = hc.StateClassifier({"beached": ["beached", "stranded"],
                                     "burning": ["burning"]})
        self.assertEqual(custom.normalize("the ship is stranded"), "beached")
        self.assertIsNone(custom.normalize("the vessel is aground"))

    def test_default_map_is_not_mutated_by_a_custom_instance(self):
        hc.StateClassifier({"only_one": ["x"]})
        self.assertEqual(hc.StateClassifier().normalize("aground here"), "aground")

    def test_states_mentioned_collapses_synonyms(self):
        c = hc.StateClassifier()
        self.assertEqual(c.states_mentioned("sunk and submerged"), {"sunken"})
        self.assertEqual(c.states_mentioned("aground and capsized"),
                         {"aground", "capsized"})

    def test_empty_text_is_handled(self):
        c = hc.StateClassifier()
        self.assertEqual(c.states_mentioned(""), set())
        self.assertIsNone(c.normalize(""))
        self.assertFalse(c.is_hedged(""))


class TestHealthFlags(unittest.TestCase):

    def test_parse_failed_tracks_missing_label(self):
        self.assertTrue(hc.HealthFlags().parse_failed)
        self.assertFalse(hc.HealthFlags(parsed_label="aground").parse_failed)

    def test_hedged_records_still_carry_a_label(self):
        # The subtlety the flags exist to expose: hedging is not parse failure.
        flags = hc.RecordAnnotator().fit([]).flags_for("either aground or capsized")
        self.assertTrue(flags.hedge_detected)
        self.assertFalse(flags.parse_failed)

    def test_to_dict_matches_the_downstream_shape(self):
        d = hc.HealthFlags(parsed_label="sunken").to_dict()
        self.assertEqual(set(d), {"repetition_detected", "hedge_detected",
                                  "refusal_detected", "length_anomaly",
                                  "parsed_label"})


class TestComboStats(unittest.TestCase):

    def _stats(self, **kw):
        base = dict(model="llava", method="baseline", prompt_stem="p1")
        base.update(kw)
        return hc.ComboStats(**base)

    def test_rates_are_derived_not_stored(self):
        s = self._stats(n_records=10, n_parse_fail=3)
        self.assertAlmostEqual(s.parse_fail_rate, 0.3)
        self.assertEqual(s.n_parsed, 7)

    def test_empty_combination_does_not_divide_by_zero(self):
        s = self._stats(n_records=0)
        self.assertEqual(s.parse_fail_rate, 0.0)
        self.assertIsNone(s.bias_dominant)
        self.assertFalse(s.gate_fail)

    def test_gate_fires_only_above_the_parse_limit(self):
        self.assertFalse(self._stats(n_records=20, n_parse_fail=2).gate_fail)   # 10%
        self.assertTrue(self._stats(n_records=20, n_parse_fail=8).gate_fail)    # 40%

    def test_bias_never_gates(self):
        s = self._stats(n_records=20, n_parse_fail=0,
                        label_distribution={"aground": 20})
        self.assertTrue(s.label_bias)
        self.assertFalse(s.gate_fail)

    def test_small_samples_suppress_both_judgements(self):
        s = self._stats(n_records=5, n_parse_fail=5,
                        label_distribution={"aground": 5})
        self.assertFalse(s.gate_fail)
        self.assertFalse(s.label_bias)

    def test_serialised_key_stays_upper_case_for_compatibility(self):
        # health_report.json already uses GATE_FAIL; downstream tooling reads it.
        self.assertIn("GATE_FAIL", self._stats(n_records=1).to_dict())


class TestComboAggregator(unittest.TestCase):

    def test_model_tag_is_preferred_over_model_id(self):
        # model_id varies by checkpoint and would split one combination in two.
        key = hc.ComboAggregator.combo_key(
            {"model_tag": "llava", "model_id": "llava-v1.5-7b-hf"})
        self.assertEqual(key[0], "llava")

    def test_falls_back_to_model_id(self):
        key = hc.ComboAggregator.combo_key({"model_id": "qwen3vl"})
        self.assertEqual(key[0], "qwen3vl")

    def test_returns_typed_objects(self):
        recs = hc.annotate_records(
            [{"model_tag": "llava", "_method": "degf", "_prompt_stem": "p1",
              "image": "a/1.jpg", "text": "aground"}])
        result = hc.ComboAggregator().aggregate(recs)
        stats = result[("llava", "degf", "p1")]
        self.assertIsInstance(stats, hc.ComboStats)
        self.assertEqual(stats.n_records, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
