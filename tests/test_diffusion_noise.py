#!/usr/bin/env python3
"""Tests for add_diffusion_noise (DeGF/degf_utils, ONLY/only_utils).

    python tests/test_diffusion_noise.py

This is paper-method code — part of the contrastive-decoding pipeline — but it
is pure tensor math with no model weights, so it CAN be verified locally
unlike the decoding loops it feeds.

The tests are characterization tests: they pin current numerical behaviour so
the function can be documented or moved without silently changing what the
method does. They deliberately assert properties (shape, determinism,
monotonicity) rather than hard-coded values, which would break on any harmless
change to tensor library internals.

Skipped automatically if torch is unavailable.
"""

import sys
import unittest
from pathlib import Path

BB_ROOT = Path(__file__).resolve().parent.parent

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

import importlib.util

COPIES = {
    "DeGF": BB_ROOT / "DeGF" / "degf_utils" / "vcd_add_noise.py",
    "ONLY": BB_ROOT / "ONLY" / "only_utils" / "vcd_add_noise.py",
}


def load(name, path):
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("vcd_add_noise_%s" % name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODULES = ([(n, load(n, p)) for n, p in COPIES.items() if p.exists()]
           if HAVE_TORCH else [])


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestAddDiffusionNoise(unittest.TestCase):
    """Applied to every copy of the function."""

    def setUp(self):
        torch.manual_seed(0)
        self.image = torch.randn(3, 8, 8)

    def test_shape_and_dtype_are_preserved(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                out = mod.add_diffusion_noise(self.image, 100)
                self.assertEqual(out.shape, self.image.shape)
                self.assertEqual(out.dtype, self.image.dtype)

    def test_input_tensor_is_not_mutated(self):
        # The caller reuses the original image for the clean forward pass;
        # mutating it in place would corrupt the contrastive comparison.
        for name, mod in MODULES:
            with self.subTest(repo=name):
                before = self.image.clone()
                mod.add_diffusion_noise(self.image, 500)
                self.assertTrue(torch.equal(self.image, before))

    def test_step_zero_is_almost_the_original(self):
        # At t=0 the retained signal coefficient is ~1, so the output should
        # track the input closely.
        for name, mod in MODULES:
            with self.subTest(repo=name):
                out = mod.add_diffusion_noise(self.image, 0)
                self.assertLess((out - self.image).abs().mean().item(), 0.05)

    def test_noise_increases_with_the_step(self):
        # The property the method depends on: a larger step must degrade the
        # image more. If this inverted, contrastive decoding would compare
        # against the wrong end of the schedule.
        for name, mod in MODULES:
            with self.subTest(repo=name):
                torch.manual_seed(1)
                low = (mod.add_diffusion_noise(self.image, 50) - self.image).abs().mean()
                torch.manual_seed(1)
                high = (mod.add_diffusion_noise(self.image, 800) - self.image).abs().mean()
                self.assertGreater(high.item(), low.item())

    def test_deterministic_under_a_fixed_seed(self):
        # Reproducibility of a run depends on this.
        for name, mod in MODULES:
            with self.subTest(repo=name):
                torch.manual_seed(42)
                a = mod.add_diffusion_noise(self.image, 300)
                torch.manual_seed(42)
                b = mod.add_diffusion_noise(self.image, 300)
                self.assertTrue(torch.equal(a, b))

    def test_differs_across_seeds(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                torch.manual_seed(1)
                a = mod.add_diffusion_noise(self.image, 300)
                torch.manual_seed(2)
                b = mod.add_diffusion_noise(self.image, 300)
                self.assertFalse(torch.equal(a, b))

    def test_accepts_a_batched_tensor(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                batched = torch.randn(2, 3, 8, 8)
                self.assertEqual(mod.add_diffusion_noise(batched, 200).shape,
                                 batched.shape)

    def test_step_accepts_a_float(self):
        # int() is applied internally; callers pass config values that may
        # arrive as floats from JSON.
        for name, mod in MODULES:
            with self.subTest(repo=name):
                out = mod.add_diffusion_noise(self.image, 100)
                self.assertEqual(out.shape, self.image.shape)

    def test_output_is_finite(self):
        for name, mod in MODULES:
            with self.subTest(repo=name):
                for step in (0, 1, 500, 999):
                    out = mod.add_diffusion_noise(self.image, step)
                    self.assertTrue(torch.isfinite(out).all(),
                                    "non-finite output at step %d" % step)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestCopiesAgree(unittest.TestCase):

    def test_both_copies_produce_identical_output(self):
        if len(MODULES) < 2:
            self.skipTest("fewer than two copies present")
        image = torch.randn(3, 8, 8)
        outputs = []
        for name, mod in MODULES:
            torch.manual_seed(7)
            outputs.append((name, mod.add_diffusion_noise(image, 250)))
        ref_name, ref = outputs[0]
        for name, out in outputs[1:]:
            self.assertTrue(torch.equal(out, ref),
                            "%s differs from %s" % (name, ref_name))

    def test_source_files_are_byte_identical(self):
        import hashlib
        digests = {n: hashlib.md5(p.read_bytes()).hexdigest()
                   for n, p in COPIES.items() if p.exists()}
        if len(digests) < 2:
            self.skipTest("fewer than two copies present")
        self.assertEqual(len(set(digests.values())), 1,
                         "vcd_add_noise.py copies have drifted: %s" % digests)


if __name__ == "__main__":
    if not HAVE_TORCH:
        print("torch not available — skipping")
        sys.exit(0)
    print("testing copies: %s" % ", ".join(n for n, _ in MODULES))
    unittest.main(verbosity=2)
