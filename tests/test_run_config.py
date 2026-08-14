#!/usr/bin/env python3
"""Tests for CASTOR/run_config.py.

    python tests/test_run_config.py

This logic previously lived inside run_inference.py and was untestable: that
module imports the vendored LLaVA stack, which pins transformers==4.31.0 and
only imports inside the container. Extracting it made the precedence rule
verifiable.

The rule under test:  CLI argument  >  config.json  >  nothing

The failure this guards against is quiet. If an argparse default beat the
config, or a falsy override were skipped, the run would complete normally and
produce a full set of plausible results — for the wrong experiment.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

BB_ROOT = Path(__file__).resolve().parent.parent

CANDIDATES = {
    "DeGF": BB_ROOT / "DeGF" / "CASTOR" / "run_config.py",
    "ONLY": BB_ROOT / "ONLY" / "CASTOR" / "run_config.py",
    "QWEN-Maritime": BB_ROOT / "QWEN-Maritime" / "CASTOR" / "run_config.py",
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(
        "run_config_%s" % name.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODULES = [(n, load(n, p)) for n, p in CANDIDATES.items() if p.exists()]


class Args:
    """Stand-in for an argparse namespace."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


BASE = {
    "paths": {"model_path": "/cfg/model", "image_folder": "/cfg/images"},
    "hyperparameters": {"temperature": 1.0, "use_diffusion": True,
                        "max_new_tokens": 1024},
}


def fresh():
    return json.loads(json.dumps(BASE))


class TestRunConfig(unittest.TestCase):
    """Applied to every repo's copy."""

    def test_sections_are_exposed(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig(fresh())
                self.assertEqual(cfg.paths["model_path"], "/cfg/model")
                self.assertEqual(cfg.hyperparameters["temperature"], 1.0)

    def test_dict_access_still_works_for_existing_callers(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig(fresh())
                self.assertEqual(cfg["paths"]["model_path"], "/cfg/model")
                self.assertIn("hyperparameters", cfg)

    def test_missing_sections_are_created(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig({})
                self.assertEqual(cfg.paths, {})
                self.assertEqual(cfg.hyperparameters, {})

    def test_load_reads_json(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / "config.json"
                    p.write_text(json.dumps(BASE), encoding="utf-8")
                    cfg = mod.RunConfig.load(str(p))
                    self.assertEqual(cfg.paths["model_path"], "/cfg/model")

    # ── Precedence ───────────────────────────────────────────────────────────

    def test_supplied_argument_overrides_config(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig(fresh()).apply_overrides(
                    Args(model_path="/cli/model"), ("model_path",), ())
                self.assertEqual(cfg.paths["model_path"], "/cli/model")

    def test_none_does_not_override(self):
        # argparse defaults are None so "not supplied" is distinguishable.
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig(fresh()).apply_overrides(
                    Args(model_path=None), ("model_path",), ())
                self.assertEqual(cfg.paths["model_path"], "/cfg/model")

    def test_falsy_overrides_are_applied(self):
        # The bug this prevents: --temperature 0 or --no-diffusion being
        # silently discarded because the value is falsy.
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig(fresh()).apply_overrides(
                    Args(temperature=0.0, use_diffusion=False, max_new_tokens=0),
                    (), ("temperature", "use_diffusion", "max_new_tokens"))
                self.assertEqual(cfg.hyperparameters["temperature"], 0.0)
                self.assertIs(cfg.hyperparameters["use_diffusion"], False)
                self.assertEqual(cfg.hyperparameters["max_new_tokens"], 0)

    def test_absent_attribute_is_skipped_not_an_error(self):
        # Repos share this merge but expose different flags.
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig(fresh()).apply_overrides(
                    Args(), ("model_path",), ("temperature",))
                self.assertEqual(cfg.paths["model_path"], "/cfg/model")

    def test_only_listed_keys_are_considered(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig(fresh()).apply_overrides(
                    Args(model_path="/cli", image_folder="/cli/img"),
                    ("model_path",), ())
                self.assertEqual(cfg.paths["image_folder"], "/cfg/images")

    def test_new_keys_are_added(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig(fresh()).apply_overrides(
                    Args(answers_file="/cli/out.jsonl"), ("answers_file",), ())
                self.assertEqual(cfg.paths["answers_file"], "/cli/out.jsonl")

    def test_apply_overrides_is_chainable(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig(fresh())
                self.assertIs(cfg.apply_overrides(Args(), (), ()), cfg)

    # ── Expansion ────────────────────────────────────────────────────────────

    def test_resolve_expands_environment_variables(self):
        import os
        os.environ["CASTOR_TEST_USER"] = "someone"
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig({"paths": {"p": "/data/$CASTOR_TEST_USER/x"},
                                     "hyperparameters": {}})
                self.assertEqual(cfg.resolve("paths", "p"), "/data/someone/x")

    def test_resolve_passes_non_strings_through(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                cfg = mod.RunConfig({"paths": {}, "hyperparameters": {"n": 42}})
                self.assertEqual(cfg.resolve("hyperparameters", "n"), 42)


class TestNoDrift(unittest.TestCase):

    def test_copies_are_byte_identical(self):
        import hashlib
        digests = {n: hashlib.md5(p.read_bytes()).hexdigest()
                   for n, p in CANDIDATES.items() if p.exists()}
        if len(digests) < 2:
            self.skipTest("fewer than two copies present")
        self.assertEqual(len(set(digests.values())), 1,
                         "run_config.py copies have drifted: %s" % digests)


class TestKeyListsMatchParsers(unittest.TestCase):
    """The key lists in run_inference.py must name real argparse destinations.

    A typo here is silent: the flag parses fine, the merge skips it because
    getattr returns None, and the run proceeds using the config value while
    appearing to honour the override.
    """

    REPOS = {
        "DeGF": BB_ROOT / "DeGF" / "CASTOR" / "run_inference.py",
        "ONLY": BB_ROOT / "ONLY" / "CASTOR" / "run_inference.py",
        "QWEN-Maritime": BB_ROOT / "QWEN-Maritime" / "CASTOR" / "run_inference.py",
    }

    def test_every_declared_key_has_a_matching_argument(self):
        import re
        for repo, path in self.REPOS.items():
            if not path.exists():
                continue
            with self.subTest(repo=repo):
                src = path.read_text(encoding="utf-8")
                declared = set()
                for name in ("_CFG_PATH_KEYS", "_CFG_HP_KEYS"):
                    m = re.search(r"%s = \((.*?)\)" % name, src, re.S)
                    if m:
                        declared |= set(re.findall(r'"(\w+)"', m.group(1)))

                # argparse turns --foo-bar into dest foo_bar unless dest= is given.
                flags = set(re.findall(r'add_argument\(\s*"--([\w-]+)"', src))
                dests = {f.replace("-", "_") for f in flags}
                dests |= set(re.findall(r'dest\s*=\s*"(\w+)"', src))

                missing = declared - dests
                self.assertFalse(
                    missing,
                    "%s declares keys with no matching argument: %s" % (repo, sorted(missing)))


if __name__ == "__main__":
    if not MODULES:
        print("no run_config.py found in any repo")
        sys.exit(1)
    print("testing: %s" % ", ".join(n for n, _ in MODULES))
    unittest.main(verbosity=2)
