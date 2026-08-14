#!/usr/bin/env python3
"""Tests for CASTOR/prepare_dataset.py.

    python tests/test_prepare_dataset.py

The same script is duplicated in DeGF, ONLY and QWEN-Maritime. This suite runs
against every copy that exists, so behaviour cannot silently diverge between
repos — they had already drifted cosmetically before these tests were added.

The property that matters most is DETERMINISTIC ORDERING. question_id is
positional, so if directory traversal order ever changed, every question_id
would shift and results could no longer be joined against previously generated
answers. Nothing would raise; the join would just be wrong.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

BB_ROOT = Path(__file__).resolve().parent.parent

REPOS = ["DeGF", "ONLY", "QWEN-Maritime"]


def load_copy(repo):
    """Import a repo's prepare_dataset.py under a unique module name."""
    path = BB_ROOT / repo / "CASTOR" / "prepare_dataset.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("prepare_dataset_%s" % repo.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODULES = [(r, m) for r, m in ((r, load_copy(r)) for r in REPOS) if m is not None]


def make_images(root, layout):
    """layout: {subdir_or_None: [filenames]}"""
    for sub, names in layout.items():
        d = Path(root) / sub if sub else Path(root)
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).write_bytes(b"")


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


class PrepareDatasetContract(unittest.TestCase):
    """Applied to every copy of the script."""

    def _run(self, mod, layout, prompt="PROMPT"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        img_dir = Path(tmp.name) / "images"
        img_dir.mkdir()
        make_images(img_dir, layout)
        out = Path(tmp.name) / "nested" / "questions.jsonl"
        mod.prepare(str(img_dir), str(out), prompt)
        return read_jsonl(out)

    def test_flat_layout(self):
        for repo, mod in MODULES:
            with self.subTest(repo=repo):
                recs = self._run(mod, {None: ["b.jpg", "a.jpg"]})
                self.assertEqual(len(recs), 2)
                self.assertEqual([r["image"] for r in recs], ["a.jpg", "b.jpg"])

    def test_categorized_layout_uses_relative_paths(self):
        for repo, mod in MODULES:
            with self.subTest(repo=repo):
                recs = self._run(mod, {"aground": ["1.jpg"], "sunken": ["2.jpg"]})
                self.assertEqual([r["image"] for r in recs],
                                 ["aground/1.jpg", "sunken/2.jpg"])

    def test_ordering_is_deterministic(self):
        # question_id is positional; unstable ordering would silently break
        # joins against previously generated answers.
        for repo, mod in MODULES:
            with self.subTest(repo=repo):
                layout = {"sunken": ["9.jpg", "10.jpg"], "aground": ["2.jpg", "1.jpg"]}
                first = [r["image"] for r in self._run(mod, layout)]
                second = [r["image"] for r in self._run(mod, layout)]
                self.assertEqual(first, second)
                self.assertEqual(first, sorted(first))

    def test_question_ids_are_sequential_from_zero(self):
        for repo, mod in MODULES:
            with self.subTest(repo=repo):
                recs = self._run(mod, {"a": ["1.jpg", "2.jpg", "3.jpg"]})
                self.assertEqual([r["question_id"] for r in recs], [0, 1, 2])

    def test_uppercase_extensions_are_included(self):
        # The dataset genuinely contains e.g. aground/00039.JPG — dropping it
        # would silently shrink the evaluation set.
        for repo, mod in MODULES:
            with self.subTest(repo=repo):
                recs = self._run(mod, {"aground": ["a.JPG", "b.JPEG", "c.PNG"]})
                self.assertEqual(len(recs), 3)

    def test_non_image_files_are_skipped(self):
        for repo, mod in MODULES:
            with self.subTest(repo=repo):
                recs = self._run(mod, {"a": ["1.jpg", "notes.txt", ".DS_Store"]})
                self.assertEqual([r["image"] for r in recs], ["a/1.jpg"])

    def test_prompt_is_applied_to_every_record(self):
        for repo, mod in MODULES:
            with self.subTest(repo=repo):
                recs = self._run(mod, {"a": ["1.jpg", "2.jpg"]}, prompt="XYZ")
                self.assertTrue(all(r["text"] == "XYZ" for r in recs))

    def test_record_has_exactly_the_expected_fields(self):
        for repo, mod in MODULES:
            with self.subTest(repo=repo):
                recs = self._run(mod, {"a": ["1.jpg"]})
                self.assertEqual(set(recs[0]), {"question_id", "image", "text"})

    def test_output_parent_directory_is_created(self):
        # The output path points into /data/$USER/... which may not exist yet.
        for repo, mod in MODULES:
            with self.subTest(repo=repo):
                recs = self._run(mod, {"a": ["1.jpg"]})   # writes to nested/
                self.assertEqual(len(recs), 1)

    def test_missing_image_dir_raises(self):
        for repo, mod in MODULES:
            with self.subTest(repo=repo):
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaises(FileNotFoundError):
                        mod.prepare(str(Path(tmp) / "nope"),
                                    str(Path(tmp) / "out.jsonl"), "p")

    def test_empty_directory_yields_empty_output(self):
        for repo, mod in MODULES:
            with self.subTest(repo=repo):
                self.assertEqual(self._run(mod, {}), [])


class TestCopiesAgree(unittest.TestCase):
    """All copies must produce byte-identical output for identical input."""

    def test_all_copies_produce_the_same_records(self):
        if len(MODULES) < 2:
            self.skipTest("fewer than two copies present")
        layout = {"aground": ["1.jpg", "2.JPG"], "sunken": ["3.png"], None: ["top.jpeg"]}
        outputs = []
        for repo, mod in MODULES:
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            img_dir = Path(tmp.name) / "images"
            img_dir.mkdir()
            make_images(img_dir, layout)
            out = Path(tmp.name) / "q.jsonl"
            mod.prepare(str(img_dir), str(out), "P")
            outputs.append((repo, read_jsonl(out)))

        reference_repo, reference = outputs[0]
        for repo, recs in outputs[1:]:
            self.assertEqual(recs, reference,
                             "%s disagrees with %s" % (repo, reference_repo))


class TestNoDrift(unittest.TestCase):
    """The copies must stay byte-identical.

    Duplication is only safe if divergence is caught automatically rather than
    relied on being remembered. These three had already drifted once.
    """

    def test_copies_are_byte_identical(self):
        import hashlib
        digests = {}
        for repo in REPOS:
            p = BB_ROOT / repo / "CASTOR" / "prepare_dataset.py"
            if p.exists():
                digests[repo] = hashlib.md5(p.read_bytes()).hexdigest()
        if len(digests) < 2:
            self.skipTest("fewer than two copies present")
        self.assertEqual(len(set(digests.values())), 1,
                         "prepare_dataset.py copies have drifted: %s" % digests)


class TestPromptSource(unittest.TestCase):
    """Prompt resolution, present on the refactored copies."""

    def test_file_inline_and_default(self):
        for repo, mod in MODULES:
            if not hasattr(mod, "PromptSource"):
                continue
            with self.subTest(repo=repo):
                self.assertEqual(mod.PromptSource(prompt="X").resolve(), "X")
                self.assertEqual(mod.PromptSource().resolve(),
                                 mod.PromptSource.DEFAULT)
                with tempfile.TemporaryDirectory() as tmp:
                    f = Path(tmp) / "p.txt"
                    # Trailing newline must not become part of the prompt.
                    f.write_text("FROM FILE\n", encoding="utf-8")
                    self.assertEqual(
                        mod.PromptSource(prompt_file=str(f)).resolve(), "FROM FILE")


if __name__ == "__main__":
    if not MODULES:
        print("no prepare_dataset.py copies found")
        sys.exit(1)
    print("testing copies: %s" % ", ".join(r for r, _ in MODULES))
    unittest.main(verbosity=2)
