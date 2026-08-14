#!/usr/bin/env python3
"""Tests for Eval_CASTOR P4 separated-field run discovery.

    python tests/test_separated_discovery.py

P4 evaluates the multi-turn prompt format, where each field is asked as its own
question. One run is therefore a DIRECTORY of per-field files rather than a
single answers file, which is why its discovery differs from P1's.

The behaviour most worth pinning is the tolerated misspelling. Early runs were
written as "separeted_into_parts_", and those directories still exist. If the
tolerance were dropped, historical runs would become invisible — no error, just
a shorter results table.

Pure filesystem work: no cluster, no models.
"""

import sys
import tempfile
import unittest
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent.parent / "Eval_CASTOR"
sys.path.insert(0, str(EVAL_ROOT))

from pipelines.eval_separated import (            # noqa: E402
    SeparatedRunDiscovery, discover_runs,
)
from shared.loaders import used_diffusion          # noqa: E402


class DiscoveryCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def make(self, name):
        (self.dir / name).mkdir()

    def names(self):
        return [n for _, n, _ in discover_runs(self.dir)]


class TestPrefixTolerance(DiscoveryCase):

    def test_correct_spelling_is_found(self):
        self.make("separated_into_parts_baseline")
        self.assertEqual(self.names(), ["separated_into_parts_baseline"])

    def test_historical_misspelling_is_still_found(self):
        # "separeted_" — the typo. Dropping this would make old runs vanish
        # from the results table with no error.
        self.make("separeted_into_parts_baseline")
        self.assertEqual(self.names(), ["separeted_into_parts_baseline"])

    def test_both_spellings_coexist(self):
        self.make("separated_into_parts_a")
        self.make("separeted_into_parts_b")
        self.assertEqual(len(self.names()), 2)

    def test_both_prefixes_are_declared(self):
        self.assertIn("separated_into_parts_", SeparatedRunDiscovery.PREFIXES)
        self.assertIn("separeted_into_parts_", SeparatedRunDiscovery.PREFIXES)

    def test_unrelated_directories_are_ignored(self):
        self.make("separated_into_parts_good")
        self.make("some_other_run")
        self.make("p1_regex")
        self.assertEqual(self.names(), ["separated_into_parts_good"])

    def test_files_are_not_mistaken_for_runs(self):
        (self.dir / "separated_into_parts_notadir.jsonl").write_text(
            "{}", encoding="utf-8")
        self.assertEqual(self.names(), [])


class TestDiffusionFlag(DiscoveryCase):

    def test_degf_directory_is_flagged(self):
        self.make("separated_into_parts_degf_p4")
        self.assertTrue(discover_runs(self.dir)[0][2])

    def test_baseline_directory_is_not(self):
        self.make("separated_into_parts_baseline_p4")
        self.assertFalse(discover_runs(self.dir)[0][2])

    def test_uses_the_same_rule_as_the_other_pipelines(self):
        # The point of sharing it: if P1 and P4 classified a run differently,
        # their comparison tables would disagree with nothing raising.
        self.make("separated_into_parts_degf_x")
        flag = discover_runs(self.dir)[0][2]
        self.assertEqual(flag, used_diffusion("separated_into_parts_degf_x"))


class TestMechanics(DiscoveryCase):

    def test_missing_directory_returns_empty_not_an_error(self):
        self.assertEqual(discover_runs(self.dir / "nope"), [])

    def test_empty_directory(self):
        self.assertEqual(discover_runs(self.dir), [])

    def test_results_are_sorted_for_determinism(self):
        for suffix in ("c", "a", "b"):
            self.make("separated_into_parts_" + suffix)
        self.assertEqual(self.names(),
                         ["separated_into_parts_a",
                          "separated_into_parts_b",
                          "separated_into_parts_c"])

    def test_returns_path_name_and_flag(self):
        self.make("separated_into_parts_x")
        path, name, flag = discover_runs(self.dir)[0]
        self.assertTrue(path.is_dir())
        self.assertEqual(name, "separated_into_parts_x")
        self.assertIsInstance(flag, bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
