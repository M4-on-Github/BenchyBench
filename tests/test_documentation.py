#!/usr/bin/env python3
"""Documentation coverage across BenchyBench and its submodules.

    python tests/test_documentation.py

Enforces that first-party Python carries module docstrings, so documentation
is a checked property rather than something that decays quietly as files are
added. Parses source with ast — nothing is imported, so modules requiring a
GPU, model weights, or a pinned transformers version are still covered.

Vendored upstream code (experiments/, transformers/, lavis/, minigpt4/) is out
of scope: it is third-party and not ours to document.
"""

import ast
import io
import unittest
from pathlib import Path

BB_ROOT = Path(__file__).resolve().parent.parent

VENDORED = ("experiments", "transformers", "lavis", "minigpt4", "eval",
            "__pycache__", "build")

REPOS = ["DeGF", "ONLY", "QWEN-Maritime", "Eval_CASTOR"]

#: Files implementing published methods. They must carry an explicit warning,
#: because their arithmetic determines published numbers and a well-meaning
#: refactor there is the most expensive mistake available in this codebase.
PAPER_METHOD_FILES = [
    BB_ROOT / "DeGF" / "degf_utils" / "degf_sample.py",
    BB_ROOT / "ONLY" / "only_utils" / "only_sample.py",
]


def is_vendored(path):
    return any(part in VENDORED for part in path.parts)


def first_party_python():
    """Every first-party .py file across the parent repo and submodules."""
    found = []
    for rel in ["visual_classification"] + REPOS:
        root = BB_ROOT / rel
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if not is_vendored(p.relative_to(BB_ROOT)):
                found.append(p)
    return sorted(found)


def module_docstring(path):
    try:
        return ast.get_docstring(ast.parse(io.open(str(path), encoding="utf-8").read()))
    except SyntaxError:
        return None


class TestModuleDocstrings(unittest.TestCase):

    def test_every_first_party_module_is_documented(self):
        files = first_party_python()
        self.assertGreater(len(files), 20, "file discovery looks wrong")

        undocumented = []
        for p in files:
            if p.name == "__init__.py":
                continue          # package markers are legitimately empty
            if not module_docstring(p):
                undocumented.append(str(p.relative_to(BB_ROOT)))

        self.assertEqual(
            undocumented, [],
            "modules without a docstring:\n  " + "\n  ".join(undocumented))

    def test_all_first_party_files_parse(self):
        for p in first_party_python():
            with self.subTest(file=str(p.relative_to(BB_ROOT))):
                try:
                    ast.parse(io.open(str(p), encoding="utf-8").read())
                except SyntaxError as e:
                    self.fail("%s: %s" % (p, e))


class TestPaperMethodWarnings(unittest.TestCase):
    """Files implementing published methods must say so."""

    def test_they_carry_a_caution(self):
        for path in PAPER_METHOD_FILES:
            if not path.exists():
                continue
            with self.subTest(file=path.name):
                doc = module_docstring(path) or ""
                self.assertTrue(
                    "CAUTION" in doc,
                    "%s must warn that it determines published numbers" % path.name)

    def test_they_explain_the_monkey_patch(self):
        # Both files replace transformers' generation methods globally. Anyone
        # reading them needs to know that before they wonder why a baseline run
        # also goes through this code.
        for path in PAPER_METHOD_FILES:
            if not path.exists():
                continue
            with self.subTest(file=path.name):
                doc = (module_docstring(path) or "").lower()
                self.assertIn("patch", doc)
                self.assertIn("4.31.0", doc,
                              "%s should record why transformers is pinned" % path.name)


class TestKeyModulesDocumentTheirContract(unittest.TestCase):
    """Modules whose behaviour is easy to get subtly wrong."""

    CASES = [
        # (path, substring that must appear in the module docstring, why)
        ("visual_classification/judge_submit.py", "3.6",
         "runs on bare python3, not in the container"),
        ("DeGF/CASTOR/run_config.py", "config.json",
         "states the CLI-over-config precedence rule"),
        ("DeGF/CASTOR/prepare_dataset.py", "identical",
         "warns the file is duplicated across repos"),
        ("DeGF/degf_utils/vcd_add_noise.py", "paper-method",
         "flags that it feeds a published method"),
        ("DeGF/degf_utils/image_similarity.py", "IMPORT",
         "warns CLIP loads at import time"),
    ]

    def test_contracts_are_stated(self):
        for rel, needle, why in self.CASES:
            path = BB_ROOT / rel
            if not path.exists():
                continue
            with self.subTest(file=rel):
                doc = module_docstring(path) or ""
                self.assertIn(needle, doc, "%s: docstring should %s" % (rel, why))


class TestCoverageSummary(unittest.TestCase):
    """Not an assertion so much as a visible count."""

    def test_report_coverage(self):
        files = [p for p in first_party_python() if p.name != "__init__.py"]
        documented = [p for p in files if module_docstring(p)]
        pct = 100.0 * len(documented) / len(files) if files else 0
        print("\n  module docstring coverage: %d/%d (%.0f%%)"
              % (len(documented), len(files), pct))
        self.assertEqual(len(documented), len(files))


if __name__ == "__main__":
    unittest.main(verbosity=2)
