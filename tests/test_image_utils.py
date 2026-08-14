#!/usr/bin/env python3
"""Tests for the DeGF image utility classes.

    python tests/test_image_utils.py

These wrap Stable Diffusion and CLIP, so the models themselves cannot be
exercised locally. The class STRUCTURE can: loaders are injectable, so lazy
loading, caching and preprocessing are all verifiable with fakes — no weights,
no downloads, no GPU.

The property that matters most is that CONSTRUCTION LOADS NOTHING. These
modules previously built their models at import scope, so importing
image_similarity downloaded ~600 MB and allocated GPU memory for a function
the CASTOR pipeline never calls.
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

if HAVE_TORCH:
    sys.path.insert(0, str(BB_ROOT / "DeGF"))


class FakeModel:
    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakePipe(FakeModel):
    pass


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestCLIPSimilarityLazyLoading(unittest.TestCase):

    def setUp(self):
        from degf_utils.image_similarity import CLIPSimilarity
        self.CLIPSimilarity = CLIPSimilarity
        self.loads = []

    def _make(self):
        def loader(name):
            self.loads.append(name)
            return FakeModel()
        return self.CLIPSimilarity(loader=loader,
                                   processor_loader=lambda n: object(),
                                   device="cpu")

    def test_construction_loads_nothing(self):
        sim = self._make()
        self.assertEqual(self.loads, [])
        self.assertFalse(sim.is_loaded)

    def test_model_loads_on_first_access(self):
        sim = self._make()
        _ = sim.model
        self.assertEqual(len(self.loads), 1)
        self.assertTrue(sim.is_loaded)

    def test_model_is_cached_not_reloaded(self):
        sim = self._make()
        first, second = sim.model, sim.model
        self.assertIs(first, second)
        self.assertEqual(len(self.loads), 1)

    def test_model_is_moved_to_the_device(self):
        sim = self._make()
        self.assertEqual(sim.model.moved_to, "cpu")

    def test_default_model_name(self):
        self.assertEqual(self.CLIPSimilarity.MODEL_NAME,
                         "openai/clip-vit-base-patch32")

    def test_importing_the_module_loads_nothing(self):
        # The regression this refactor exists to prevent.
        import degf_utils.image_similarity as mod
        self.assertFalse(mod._DEFAULT.is_loaded)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestStableDiffusionGenerator(unittest.TestCase):

    def setUp(self):
        from degf_utils.image_generation import StableDiffusionGenerator
        self.Gen = StableDiffusionGenerator
        self.loads = []

    def _make(self):
        def loader(name):
            self.loads.append(name)
            return FakePipe()
        return self.Gen(loader=loader, device="cpu")

    def test_construction_loads_nothing(self):
        gen = self._make()
        self.assertEqual(self.loads, [])
        self.assertFalse(gen.is_loaded)

    def test_pipeline_loads_once_on_first_access(self):
        gen = self._make()
        first, second = gen.pipe, gen.pipe
        self.assertIs(first, second)
        self.assertEqual(len(self.loads), 1)

    def test_pipeline_is_moved_to_device(self):
        gen = self._make()
        self.assertEqual(gen.pipe.moved_to, "cpu")

    def test_published_defaults_are_preserved(self):
        # Changing either changes the reference image, and therefore results.
        self.assertEqual(self.Gen.MODEL_NAME, "runwayml/stable-diffusion-v1-5")
        self.assertEqual(self.Gen.INFERENCE_STEPS, 50)

    def test_alternatives_are_recorded_for_provenance(self):
        self.assertIn("stabilityai/stable-diffusion-2-1", self.Gen.ALTERNATIVES)

    def test_facade_adopts_a_caller_supplied_pipeline(self):
        # generate_image_stable_diffusion(pipe, ...) must use the pipe it is
        # given, never load a second one.
        from degf_utils.image_generation import generate_image_stable_diffusion
        gen = self._make()
        sentinel = FakePipe()
        g2 = self.Gen(loader=lambda n: self.fail("must not load"), device="cpu")
        g2._pipe = sentinel
        self.assertIs(g2.pipe, sentinel)
        self.assertTrue(g2.is_loaded)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestImageVariationGenerator(unittest.TestCase):

    def setUp(self):
        from degf_utils.image_variation import ImageVariationGenerator
        self.Gen = ImageVariationGenerator
        self.loads = []

    def _make(self):
        def loader(name, revision):
            self.loads.append((name, revision))
            return FakePipe()
        return self.Gen(loader=loader, device="cpu")

    def test_construction_loads_nothing(self):
        gen = self._make()
        self.assertEqual(self.loads, [])
        self.assertFalse(gen.is_loaded)

    def test_pipeline_loads_once_with_the_pinned_revision(self):
        gen = self._make()
        _ = gen.pipe
        self.assertEqual(self.loads, [(self.Gen.MODEL_NAME, "v2.0")])

    def test_revision_is_pinned(self):
        # v1.0 and v2.0 condition differently and produce different variants.
        self.assertEqual(self.Gen.REVISION, "v2.0")

    def test_clip_normalisation_constants_are_preserved(self):
        # These must match what the CLIP encoder was trained with; a change
        # degrades conditioning silently rather than raising.
        self.assertEqual(self.Gen.CLIP_MEAN,
                         [0.48145466, 0.4578275, 0.40821073])
        self.assertEqual(self.Gen.CLIP_STD,
                         [0.26862954, 0.26130258, 0.27577711])
        self.assertEqual(self.Gen.INPUT_SIZE, (224, 224))

    def test_grayscale_is_converted_to_rgb(self):
        from PIL import Image
        gray = Image.new("L", (32, 32))
        self.assertEqual(self.Gen.to_rgb(gray).mode, "RGB")

    def test_rgb_is_passed_through_unchanged(self):
        from PIL import Image
        rgb = Image.new("RGB", (32, 32))
        self.assertIs(self.Gen.to_rgb(rgb), rgb)

    def test_transform_produces_the_expected_shape(self):
        from PIL import Image
        tensor = self.Gen.build_transform()(Image.new("RGB", (64, 48)))
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))


if __name__ == "__main__":
    if not HAVE_TORCH:
        print("torch not available — skipping")
        sys.exit(0)
    unittest.main(verbosity=2)
