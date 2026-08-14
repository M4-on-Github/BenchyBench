# Method Internals — DeGF and ONLY

Notes on how the two decoding methods work, gathered while reading the code.

**The implementation files are deliberately untouched.** `DeGF/degf_utils/`,
`ONLY/only_utils/` and both `utils/` directories are byte-identical to the
published implementations. Their arithmetic determines published numbers, so
they are treated as upstream: read, not edited. Everything learned about them
is recorded here instead.

---

## Installation: a global monkey-patch

Both methods install themselves by overwriting HuggingFace's generation
functions on the `transformers` module:

```python
evolve_degf_sampling()   # transformers.generation.utils.GenerationMixin.sample = sample
evolve_only_sampling()   #                                    .greedy_search = greedy_search
```

Two consequences worth knowing:

1. **The patch is global and permanent for the process.** After the call, every
   model decodes through that file — including a baseline run. Baseline
   behaviour survives only because no contrast branch activates unless its flag
   is set in `model_kwargs`. If you are ever puzzled that baseline output passes
   through contrastive code, this is why.

2. **This is why `transformers` is pinned to 4.31.0.** These functions are
   copies of that version's originals with a contrast block inserted. A
   different version has a different generation loop and internal contract, so
   the patch would either fail outright or silently diverge.

---

## DeGF — Jensen-Shannon gate

`DeGF/degf_utils/degf_sample.py`

The model runs **twice per token**: once on the real image, once on a Stable
Diffusion image regenerated from the model's own description. DeGF measures the
Jensen-Shannon divergence between the two next-token distributions and switches
the correction's *direction* on it:

| Divergence | Meaning | Action |
|---|---|---|
| `js < 0.1` | the views agree, reference is trustworthy | **ADD** its signal |
| `js >= 0.1` | the views disagree, reference is a distractor | **SUBTRACT** it |

That per-token sign switch is the paper's contribution — fixed contrastive
decoding always subtracts.

**The `0.1` threshold is hardcoded**, not a config value. Changing it changes
the method, not a setting.

### Adaptive plausibility constraint

```python
cutoff = log(degf_beta) + next_token_logits.max()
logits = diffs.masked_fill(next_token_logits < cutoff, -inf)
```

The mask compares against the **original** logits, not the corrected ones. That
is deliberate: the constraint captures what the model found plausible *before*
any correction, so the correction cannot promote a token the model never
seriously considered. Subtracting a large reference logit can otherwise leave
something nonsensical on top.

---

## ONLY — total variation gate, one forward pass

`ONLY/only_utils/only_sample.py`

### Why it is cheaper

DeGF needs two forward passes plus a Stable Diffusion generation. ONLY needs
**one**:

```python
outputs, logits_cd = self(**model_inputs, ...)
```

A single call returns two logit streams — the normal output plus one from
intervening on a single transformer layer. No second pass, no diffusion model
resident in memory. That is the efficiency claim, visible in the call signature.

### Two naming traps

1. **The gate is NOT Jensen-Shannon**, despite the threshold being called
   `js_gamma` throughout. It is total variation distance:

   ```python
   tvd = torch.sum(torch.abs(softmax(next_token_logits) - softmax(logits_cd)))
   ```

   The name is inherited from an earlier JS-based formulation whose computation
   is still present, commented out, in the file.

2. **It is the L1 distance — twice the conventional TVD**, which is defined as
   half the L1 norm. `js_gamma` is calibrated to this doubled scale, so
   "correcting" the formula to the textbook definition would halve every
   distance and change which branch fires on every token.

Unlike DeGF's threshold, this one *is* configurable via `config.json`.

---

## M3ID — the schedule strengthens, it does not decay

Present in both files. Easy to read backwards:

```python
gamma_t = torch.exp(torch.tensor(-0.02 * t))
diffs = logits + (logits - logits_neg) * (1 - gamma_t) / gamma_t
```

`gamma_t` decays, but it appears as `(1 - gamma_t) / gamma_t`, which **grows**:

| t | coefficient |
|---|---|
| 0 | 0 (no correction at all) |
| 1 | ~0.02 |
| 200 | ~53.6 |

A VLM's conditioning on the image fades as generated text lengthens — later
tokens are increasingly driven by what was already written. The visual
correction is *amplified* to counteract that drift, not backed off.

---

## Image utilities

`DeGF/degf_utils/image_*.py`

- **`image_similarity.py` loads CLIP at module import**, not on first use.
  Importing it downloads ~600 MB on a cold cache and allocates GPU memory. It
  has no callers in the CASTOR pipeline, so the cost is latent — but import it
  lazily if you ever add one, or a baseline run starts paying for CLIP.

- **`image_generation.py` routes through `get_pipeline_embeds`** rather than
  passing the description straight to the pipeline. VLM descriptions routinely
  exceed CLIP's 77-token limit, and the direct path truncates silently —
  dropping exactly the trailing detail the contrast exists to test.

- **`image_variation.py` preprocesses with CLIP's constants**, not Stable
  Diffusion's, because that pipeline conditions on a CLIP image embedding. The
  mean/std must match what the encoder was trained with; changing them degrades
  conditioning silently rather than raising. `antialias=False` and the pinned
  `revision="v2.0"` are likewise load-bearing.

- **`vcd_add_noise.py`** rebuilds its 1000-step diffusion schedule on every
  call. Wasted work at one image per call, but left alone — the tensors feed a
  published method. Covered by `tests/test_diffusion_noise.py`, which verifies
  shape, determinism under a fixed seed, non-mutation of the input, and that
  noise increases with the step, all without modifying the file.

---

## If these files ever must change

Capture a fixed-seed reference run first, so output can be diffed
byte-for-byte afterwards. Local tests verify tensor properties; they cannot
tell you whether a changed number is still the method described in the paper.
