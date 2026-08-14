#!/usr/bin/env python3
"""Tests for the DeGF-ablation reference selector.

    python tests/test_reference_selector.py

The ablation replaces DeGF's generated reference image with deliberately wrong
or fixed ones, to test what the contrast is actually doing. Picking the WRONG
wrong-image would invalidate the conclusion silently — the run completes, the
numbers look plausible, and nothing indicates the reference was not what the
mode claims.

Path resolution is pure, so every mode is verified here without any image file
being present.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

BB_ROOT = Path(__file__).resolve().parent.parent
TARGET = BB_ROOT / "QWEN-Maritime" / "CASTOR" / "degf_ablate" / "run_degf_ablate.py"


def load_selector():
    """Import ReferenceSelector without executing the module's model imports."""
    if not TARGET.exists():
        return None
    import ast, io
    src = io.open(str(TARGET), encoding="utf-8").read()
    tree = ast.parse(src)
    wanted = [n for n in tree.body
              if isinstance(n, ast.ClassDef) and n.name == "ReferenceSelector"]
    if not wanted:
        return None
    # Execute just the class definition against a namespace with its two deps.
    import os
    # Stand-in for PIL's Image module. Needs an `.Image` attribute because the
    # method annotations reference `Image.Image`, which is evaluated when the
    # class body executes.
    class _FakeImageModule:
        Image = type("Image", (), {})

        @staticmethod
        def open(path):
            raise AssertionError("resolve() must not open files")

    ns = {"os": os, "Image": _FakeImageModule}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "<selector>", "exec"), ns)
    return ns["ReferenceSelector"]


Selector = load_selector()


@unittest.skipUnless(Selector is not None, "ReferenceSelector not found")
class TestImageClass(unittest.TestCase):

    def test_extracts_the_class_directory(self):
        self.assertEqual(Selector.image_class("aground/00017.jpg"), "aground")
        self.assertEqual(Selector.image_class("on_fire/00011.jpg"), "on_fire")

    def test_splits_on_forward_slash_regardless_of_platform(self):
        # These are dataset keys from questions.jsonl, always forward-slashed,
        # not filesystem paths.
        self.assertEqual(Selector.image_class("sunken/1.jpg"), "sunken")


@unittest.skipUnless(Selector is not None, "ReferenceSelector not found")
class TestSwapMap(unittest.TestCase):

    def test_swap_is_symmetric(self):
        # Reciprocal pairing: swapping twice returns the original class.
        for cls, swapped in Selector.SWAP_MAP.items():
            self.assertEqual(Selector.SWAP_MAP[swapped], cls,
                             "%s -> %s is not reciprocal" % (cls, swapped))

    def test_no_class_maps_to_itself(self):
        # A class mapping to itself would make swap_sd identical to
        # matched_real, silently collapsing two ablation arms into one.
        for cls, swapped in Selector.SWAP_MAP.items():
            self.assertNotEqual(cls, swapped)

    def test_covers_all_four_casualty_classes(self):
        self.assertEqual(set(Selector.SWAP_MAP),
                         {"aground", "capsized", "on_fire", "sunken"})


@unittest.skipUnless(Selector is not None, "ReferenceSelector not found")
class TestResolve(unittest.TestCase):

    def setUp(self):
        self.sel = Selector("/assets")

    def test_swap_sd_picks_the_opposite_class(self):
        path, label = self.sel.resolve("swap_sd", "aground/1.jpg")
        self.assertIn("sunken", label)
        self.assertNotIn("aground", label)

    def test_matched_real_picks_the_correct_class(self):
        path, label = self.sel.resolve("matched_real", "aground/1.jpg")
        self.assertIn("aground", label)
        self.assertIn("real", label)

    def test_swap_real_picks_the_opposite_class_photograph(self):
        path, label = self.sel.resolve("swap_real", "on_fire/1.jpg")
        self.assertIn("capsized", label)
        self.assertIn("real", label)

    def test_banana_is_the_same_for_every_query(self):
        a = self.sel.resolve("banana_sd", "aground/1.jpg")[1]
        b = self.sel.resolve("banana_sd", "sunken/2.jpg")[1]
        self.assertEqual(a, b)
        self.assertIn("banana", a)

    def test_fixed_real_is_the_same_for_every_query(self):
        a = self.sel.resolve("fixed_real", "aground/1.jpg")[1]
        b = self.sel.resolve("fixed_real", "on_fire/2.jpg")[1]
        self.assertEqual(a, b)

    def test_matched_and_swap_never_agree(self):
        # If they ever picked the same file the ablation would compare an arm
        # against itself and report no difference — a null result caused by a
        # bug rather than by the method.
        for cls in Selector.SWAP_MAP:
            img = "%s/1.jpg" % cls
            self.assertNotEqual(self.sel.resolve("matched_real", img)[1],
                                self.sel.resolve("swap_real", img)[1])

    def test_every_declared_mode_resolves(self):
        for mode in Selector.MODES:
            with self.subTest(mode=mode):
                path, label = self.sel.resolve(mode, "aground/1.jpg")
                self.assertTrue(path)
                self.assertTrue(label)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            self.sel.resolve("not_a_mode", "aground/1.jpg")

    def test_resolve_touches_no_files(self):
        # Pure: works against a directory that does not exist.
        Selector("/nonexistent").resolve("swap_sd", "aground/1.jpg")


if __name__ == "__main__":
    if Selector is None:
        print("ReferenceSelector not found — skipping")
        sys.exit(0)
    unittest.main(verbosity=2)
