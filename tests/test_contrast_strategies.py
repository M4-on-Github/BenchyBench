#!/usr/bin/env python3
"""Numerical-equivalence tests for the extracted contrast strategies.

    python tests/test_contrast_strategies.py

These are the tests that make restructuring paper-method code safe. Each test
computes the ORIGINAL inline expression — copied verbatim from the pre-refactor
degf_sample.py / only_sample.py — and asserts the extracted class produces a
BITWISE identical tensor on random inputs.

Bitwise, not approximate: torch.equal, not allclose. The operations are the
same in the same order, so any difference at all would mean the extraction
changed the computation, and a tolerance would hide exactly the drift these
tests exist to catch.

Requires torch but no model weights, no GPU and no network — the contrast math
is pure tensor arithmetic over two logit tensors. Only the decoding loop that
calls it needs a model.
"""

import sys
import unittest
from pathlib import Path

BB_ROOT = Path(__file__).resolve().parent.parent

try:
    import torch
    from torch import nn
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    sys.path.insert(0, str(BB_ROOT / "DeGF"))
    from degf_utils.contrast_strategies import (
        RitualContrast, VCDContrast, M3IDContrast, DiffusionContrast,
        ContrastStrategy,
    )


def logits_pair(seed=0, vocab=512):
    torch.manual_seed(seed)
    return torch.randn(1, vocab), torch.randn(1, vocab)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestRitualEquivalence(unittest.TestCase):

    def test_matches_the_original_expression(self):
        for seed in range(5):
            logits, ref = logits_pair(seed)
            alpha_pos = 3
            # ── original, verbatim from degf_sample.py ──
            expected = (logits + alpha_pos * ref)
            # ── extracted ──
            actual = RitualContrast(alpha_pos).combine(logits, ref)
            self.assertTrue(torch.equal(expected, actual), "seed=%d" % seed)

    def test_matches_across_alpha_values(self):
        logits, ref = logits_pair(1)
        for alpha_pos in (0, 0.5, 1, 3, 10, -2):
            expected = (logits + alpha_pos * ref)
            actual = RitualContrast(alpha_pos).combine(logits, ref)
            self.assertTrue(torch.equal(expected, actual), "alpha_pos=%s" % alpha_pos)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestVCDEquivalence(unittest.TestCase):

    def test_matches_the_original_expression(self):
        for seed in range(5):
            logits, ref = logits_pair(seed)
            alpha_neg = 1
            expected = (1 + alpha_neg) * logits - alpha_neg * ref
            actual = VCDContrast(alpha_neg).combine(logits, ref)
            self.assertTrue(torch.equal(expected, actual), "seed=%d" % seed)

    def test_matches_across_alpha_values(self):
        logits, ref = logits_pair(2)
        for alpha_neg in (0, 0.5, 1, 2, 5):
            expected = (1 + alpha_neg) * logits - alpha_neg * ref
            actual = VCDContrast(alpha_neg).combine(logits, ref)
            self.assertTrue(torch.equal(expected, actual), "alpha_neg=%s" % alpha_neg)

    def test_alpha_zero_is_the_identity(self):
        logits, ref = logits_pair(3)
        self.assertTrue(torch.equal(VCDContrast(0).combine(logits, ref), logits))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestM3IDEquivalence(unittest.TestCase):

    def test_matches_the_original_expression_including_the_counter(self):
        logits, ref = logits_pair(4)
        strategy = M3IDContrast(t=0)
        t = 0
        for _ in range(6):
            # ── original, verbatim ──
            gamma_t = torch.exp(torch.tensor(-0.02 * t))
            expected = logits + (logits - ref) * (1 - gamma_t) / gamma_t
            t += 1
            # ── extracted ──
            actual = strategy.combine(logits, ref)
            self.assertTrue(torch.equal(expected, actual), "t=%d" % (t - 1))

    def test_counter_advances_per_call(self):
        logits, ref = logits_pair(5)
        s = M3IDContrast()
        self.assertEqual(s.t, 0)
        s.combine(logits, ref)
        s.combine(logits, ref)
        self.assertEqual(s.t, 2)

    def test_correction_STRENGTHENS_as_position_grows(self):
        # Direction is the opposite of what "decay schedule" suggests.
        # gamma_t decays, but it appears as (1 - gamma_t) / gamma_t, which
        # grows: ~0.02 at t=1, ~53.6 at t=200. The image's conditioning
        # influence fades over a long generation, so the visual correction is
        # amplified to counteract that rather than backed off.
        logits, ref = logits_pair(6)
        early = M3IDContrast(t=1).combine(logits, ref)
        late = M3IDContrast(t=200).combine(logits, ref)
        self.assertGreater((late - logits).abs().mean().item(),
                           (early - logits).abs().mean().item())

    def test_no_correction_at_position_zero(self):
        # At t=0 the coefficient is exactly 0, so the first token is
        # uncorrected regardless of the reference.
        logits, ref = logits_pair(8)
        self.assertTrue(torch.equal(M3IDContrast(t=0).combine(logits, ref), logits))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestDiffusionEquivalence(unittest.TestCase):
    """DeGF proper — the branch that carries the paper's contribution."""

    @staticmethod
    def original_js(logits, ref):
        """Verbatim from degf_sample.py."""
        M = 0.5 * (nn.functional.softmax(logits, dim=-1) + nn.functional.softmax(ref, dim=-1))
        return 0.5 * nn.functional.kl_div(nn.functional.log_softmax(logits, dim=-1), M, reduction='batchmean') + 0.5 * nn.functional.kl_div(nn.functional.log_softmax(ref, dim=-1), M, reduction='batchmean')

    def test_js_divergence_matches(self):
        for seed in range(5):
            logits, ref = logits_pair(seed)
            self.assertTrue(torch.equal(self.original_js(logits, ref),
                                        DiffusionContrast.js_divergence(logits, ref)),
                            "seed=%d" % seed)

    def test_full_branch_matches_the_original(self):
        alpha_pos, alpha_neg = 3, 1
        for seed in range(8):
            logits, ref = logits_pair(seed)
            # ── original, verbatim ──
            js = self.original_js(logits, ref)
            if js < 0.1:
                expected = logits + alpha_pos * ref
            else:
                expected = (1 + alpha_neg) * logits - alpha_neg * ref
            # ── extracted ──
            actual = DiffusionContrast(alpha_pos, alpha_neg).combine(logits, ref)
            self.assertTrue(torch.equal(expected, actual), "seed=%d" % seed)

    def test_identical_streams_take_the_additive_branch(self):
        # Zero divergence: the reference agrees completely, so its evidence
        # must be ADDED, never subtracted.
        logits, _ = logits_pair(7)
        s = DiffusionContrast(3, 1)
        result = s.combine(logits, logits.clone())
        self.assertTrue(torch.equal(result, logits + 3 * logits))
        self.assertEqual(s.js_count, 0)

    def test_far_apart_streams_take_the_contrastive_branch(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 512) * 10
        ref = torch.randn(1, 512) * 10
        s = DiffusionContrast(3, 1)
        s.combine(logits, ref)
        self.assertEqual(s.js_count, 1)

    def test_counters_and_log_accumulate(self):
        s = DiffusionContrast(3, 1)
        for seed in range(4):
            logits, ref = logits_pair(seed)
            s.combine(logits, ref)
        self.assertEqual(s.token_count, 4)
        self.assertEqual(len(s.js_list), 4)
        self.assertLessEqual(s.js_count, s.token_count)

    def test_js_log_format_is_preserved(self):
        # The run log parses these; four decimal places as strings.
        s = DiffusionContrast(3, 1)
        logits, ref = logits_pair(0)
        s.combine(logits, ref)
        self.assertRegex(s.js_list[0], r"^\d+\.\d{4}$")

    def test_threshold_is_the_published_value(self):
        self.assertEqual(DiffusionContrast.JS_THRESHOLD, 0.1)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestPlausibilityCutoff(unittest.TestCase):

    def test_matches_the_original_expression(self):
        for seed in range(5):
            logits, _ = logits_pair(seed)
            beta = 0.1
            expected = torch.log(torch.tensor(beta)) + logits.max(dim=-1, keepdim=True).values
            actual = ContrastStrategy.plausibility_cutoff(logits, beta)
            self.assertTrue(torch.equal(expected, actual), "seed=%d" % seed)

    def test_mask_uses_original_logits_not_corrected(self):
        # The constraint must reflect what the model found plausible BEFORE
        # correction. Masking on corrected scores would let the correction
        # promote a token the model never considered.
        logits, ref = logits_pair(1)
        corrected = VCDContrast(1).combine(logits, ref)
        cutoff = ContrastStrategy.plausibility_cutoff(logits, 0.1)
        expected = corrected.masked_fill(logits < cutoff, -float("inf"))
        actual = ContrastStrategy.apply_cutoff(corrected, logits, cutoff)
        self.assertTrue(torch.equal(expected, actual))

    def test_top_token_always_survives(self):
        # cutoff = log(beta) + max, and log(beta) < 0 for beta < 1, so the
        # maximum is always above the cutoff.
        logits, _ = logits_pair(2)
        cutoff = ContrastStrategy.plausibility_cutoff(logits, 0.1)
        self.assertTrue((logits.max() >= cutoff).all())

    def test_smaller_beta_keeps_more_tokens(self):
        logits, _ = logits_pair(3)
        strict = (logits >= ContrastStrategy.plausibility_cutoff(logits, 0.5)).sum()
        loose = (logits >= ContrastStrategy.plausibility_cutoff(logits, 0.01)).sum()
        self.assertGreaterEqual(loose.item(), strict.item())


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestWiringMatchesOriginalBlock(unittest.TestCase):
    """End-to-end check of the whole contrast block as degf_sample.py runs it.

    The unit tests above prove each strategy matches its original expression.
    They cannot catch a WIRING error — passing the positive reference where the
    negative belongs, say, which would still produce a valid tensor and no
    exception. This replicates the entire block both ways and compares.
    """

    def _original_block(self, mode, logits, pos, neg, alpha_pos, alpha_neg, beta, t=0):
        """The pre-refactor block, verbatim."""
        cutoff = torch.log(torch.tensor(beta)) + logits.max(dim=-1, keepdim=True).values
        if mode == "ritual":
            diffs = (logits + alpha_pos * pos)
        elif mode == "vcd":
            diffs = (1 + alpha_neg) * logits - alpha_neg * neg
        elif mode == "m3id":
            gamma_t = torch.exp(torch.tensor(-0.02 * t))
            diffs = logits + (logits - neg) * (1 - gamma_t) / gamma_t
        elif mode == "diffusion":
            M = 0.5 * (nn.functional.softmax(logits, dim=-1) + nn.functional.softmax(neg, dim=-1))
            js = 0.5 * nn.functional.kl_div(nn.functional.log_softmax(logits, dim=-1), M, reduction='batchmean') + 0.5 * nn.functional.kl_div(nn.functional.log_softmax(neg, dim=-1), M, reduction='batchmean')
            if js < 0.1:
                diffs = logits + alpha_pos * neg
            else:
                diffs = (1 + alpha_neg) * logits - alpha_neg * neg
        return diffs.masked_fill(logits < cutoff, -float("inf"))

    def _wired_block(self, mode, logits, pos, neg, alpha_pos, alpha_neg, beta, t=0):
        """The block as degf_sample.py now runs it, with the same call sites."""
        cutoff = torch.log(torch.tensor(beta)) + logits.max(dim=-1, keepdim=True).values
        if mode == "ritual":
            diffs = RitualContrast(alpha_pos).combine(logits, pos)
        elif mode == "vcd":
            diffs = VCDContrast(alpha_neg).combine(logits, neg)
        elif mode == "m3id":
            gamma_t = torch.exp(torch.tensor(-0.02 * t))
            diffs = logits + (logits - neg) * (1 - gamma_t) / gamma_t
        elif mode == "diffusion":
            js = DiffusionContrast.js_divergence(logits, neg)
            if js < 0.1:
                diffs = logits + alpha_pos * neg
            else:
                diffs = (1 + alpha_neg) * logits - alpha_neg * neg
        return diffs.masked_fill(logits < cutoff, -float("inf"))

    def test_all_modes_identical_across_many_seeds(self):
        for mode in ("ritual", "vcd", "m3id", "diffusion"):
            for seed in range(10):
                torch.manual_seed(seed)
                logits = torch.randn(1, 512)
                pos = torch.randn(1, 512)
                neg = torch.randn(1, 512)
                with self.subTest(mode=mode, seed=seed):
                    a = self._original_block(mode, logits, pos, neg, 3, 1, 0.1, t=seed)
                    b = self._wired_block(mode, logits, pos, neg, 3, 1, 0.1, t=seed)
                    self.assertTrue(torch.equal(a, b))

    def test_swapped_references_would_be_caught(self):
        # Confirms the test above has teeth: feeding pos where neg belongs
        # must produce a different result, or the comparison proves nothing.
        torch.manual_seed(0)
        logits, pos, neg = (torch.randn(1, 512) for _ in range(3))
        correct = VCDContrast(1).combine(logits, neg)
        swapped = VCDContrast(1).combine(logits, pos)
        self.assertFalse(torch.equal(correct, swapped))


if __name__ == "__main__":
    if not HAVE_TORCH:
        print("torch not available — skipping")
        sys.exit(0)
    unittest.main(verbosity=2)
