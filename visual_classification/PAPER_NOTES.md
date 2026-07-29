# visual_classification — Paper Notes

What to look for in results, how to interpret findings, and what to report.
Written during ideation before results exist — update as actual findings emerge.

---

## Framing

The pipeline compares two hallucination-mitigation methods (DeGF, ONLY) against
baseline across two models (LLaVA-1.5-7B, Qwen3-VL-8B) on a 4-class maritime
disaster classification task. The core questions are:

1. Do DeGF and ONLY improve classification accuracy over baseline?
2. Do they improve it for the right reasons (visual grounding) or the wrong ones
   (lucky prompt sensitivity)?
3. Where do they fail, and is the failure mode consistent across models?
4. How much does prompt wording drive results vs model/method capability?

---

## Primary results to report

### Table 1 — Main accuracy table
Rows: model × method. Cols: prompt variants + average.
Cells: regex accuracy (primary) + judge accuracy (secondary).
Report both macro F1 and accuracy. Do NOT report only accuracy — the class
imbalance (aground=42, on_fire=16) makes accuracy misleading without F1.

**What to look for:**
- Does DeGF consistently outperform baseline across both models?
- Does ONLY? Is one method more consistent than the other?
- Is the best combo the same across prompt variants, or does it flip?
  (Flipping = method × prompt interaction; must be discussed)

### Table 2 — Delta vs baseline
Signed accuracy and F1 improvement of DeGF and ONLY over baseline,
per model and per prompt. Negative = regression.

**What to look for:**
- Are regressions systematic (same method always hurts the same model)
  or image-specific (occasional accidents)?
- Qwen DeGF may behave differently from LLaVA DeGF — the architectures
  handle contrastive decoding differently. Report both separately.

### Table 3 — Per-class F1
The 4-class breakdown matters. on_fire (n=16) is the smallest class and
likely the hardest. aground (n=42) is the largest and likely easiest.

**What to look for:**
- Which class benefits most from DeGF/ONLY?
- Which class regresses? A method that improves aground but destroys on_fire
  accuracy at n=16 is net-positive on accuracy but net-negative on the harder cases
- Confusion asymmetry: is sunken→capsized more common than capsized→sunken?
  Directional confusions suggest visual feature overlap that is not symmetric

---

## Key analyses for the paper

### Analysis 1 — DeGF first-pass vs final (most important)

Compare `degf_first_pass_correct` vs `regex_correct` per image.

Three cases per image:
- First pass wrong, final right → SD guidance helped (true uplift)
- First pass right, final wrong → SD guidance hurt (regression)
- First pass and final agree → SD guidance had no effect

**What to report:**
- % of DeGF "correct" predictions that were already correct in the first pass
  (i.e., baseline would have gotten it right anyway — DeGF didn't add value)
- % of DeGF uplifts that came from first-pass-wrong → final-right (genuine gains)
- % of regressions: first-pass-right → final-wrong (SD actively broke it)

**Why this matters:** If the majority of DeGF's accuracy gains come from images
where the baseline already predicted correctly, the method isn't adding value —
it's just preserving baseline performance with extra compute. If regressions are
high, the SD reference image is actively misleading the model.

### Analysis 2 — SD plausibility and accuracy (DeGF)

CLIP cosine similarity between original image and SD reference image as a
proxy for "did SD generate a relevant reference?"

**What to report:**
- Scatter plot or binned bar chart: CLIP similarity vs correct/incorrect
- Threshold analysis: at what CLIP similarity does DeGF start helping vs hurting?
- Gallery: low-CLIP images (SD went off-topic) with original + SD image side by side

**Why this matters:** A key potential failure mode of DeGF is that Stable Diffusion
generates a plausible-looking ship that is the WRONG class, then the contrastive
signal pulls the model toward that wrong class. CLIP similarity lets you test this
hypothesis quantitatively.

**Expected finding:** DeGF helps when CLIP similarity is high (SD generated a
visually relevant reference) and hurts when it is low (SD hallucinated something
unrelated). If this pattern holds, it motivates a CLIP-gated version of DeGF as
future work.

### Analysis 3 — Wrong-image-set Jaccard

For a given (model, prompt): Jaccard similarity between the set of images
baseline got wrong and the set DeGF/ONLY got wrong.

**What to report:**
- Table: Jaccard per (model × method) per prompt; averaged across prompts
- High Jaccard means the method is not changing which images fail — it's
  shuffling answers on the same hard cases
- Low Jaccard means the method is genuinely finding different hard cases

**Why this matters:** A method that has higher accuracy but similar Jaccard to
baseline is mostly getting lucky on easy images, not solving the hard ones.
A method with lower Jaccard is actually changing the failure distribution —
which is what you want from a hallucination mitigation method.

### Analysis 4 — Prompt ranking stability

Rank prompts by accuracy per (model × method). Is the ranking consistent?

**What to report:**
- Kendall's tau or Spearman correlation of prompt rankings across (model × method) combos
- If ranking is unstable: report which prompt × method × model interactions flip
- Prompt sensitivity heatmap as a figure

**Why this matters:** If prompt A is best for LLaVA-baseline but worst for
Qwen-DeGF, then the "best prompt" result is meaningless without specifying the
full combo. This is a reproducibility and generalization concern.

**Expected finding:** Qwen is likely more sensitive to prompt wording than LLaVA
(larger model, more instruction-following behavior). DeGF may reduce prompt
sensitivity (the SD signal anchors the answer to visual content more than text).

### Analysis 5 — Reliability: regex vs judge vs inter-judge agreement

**What to report:**
- Cohen's kappa (regex vs judge) per combo — if kappa > 0.8, regex is reliable
- Fleiss' kappa (inter-judge) per combo — if kappa < 0.6, judge evaluation is
  itself unreliable and results must be interpreted with caution
- Divergence cases: images where regex says correct but judge says incorrect (or vice versa) — gallery

**Why this matters:** If the evaluation is unreliable, all accuracy numbers are
suspect. This is methodology due diligence that reviewers will ask about.

---

## Failure mode findings to highlight

### Finding type A — Systematic class confusion
If the confusion matrix shows a strong off-diagonal pattern (e.g., sunken is
consistently called capsized by all methods), this is a dataset-level finding
about visual ambiguity in the CASTOR taxonomy. Report with McNemar test for
asymmetry.

### Finding type B — Method regression on small classes
If DeGF or ONLY hurts performance on on_fire (n=16) while helping on aground
(n=42), this is a critical finding: the method trades accuracy on the majority
class for accuracy loss on the minority class that is arguably harder and more
safety-critical (a ship on fire is a different emergency than one aground).

### Finding type C — Universal failures as dataset quality signal
Tier 2 images (all combos wrong) are either genuinely ambiguous images or
potential GT labeling errors. Report them explicitly. If multiple humans agree
the GT label is wrong on some Tier 2 images, this motivates a GT revision.

### Finding type D — Prompt-split as visual grounding failure
High prompt_split rate (images where the same model+method gives different answers
across prompts) means the model is not robustly grounded in visual content — text
wording is driving the classification. This is exactly the hallucination behavior
DeGF and ONLY are supposed to fix. If DeGF reduces prompt_split rate vs baseline,
that is a positive finding independent of raw accuracy.

### Finding type E — degf_sd_flip rate
If `degf_first_pass_label ≠ parsed_label` is frequent, SD guidance is actively
changing the model's answer. High flip rate + high accuracy = SD is correcting
errors. High flip rate + low accuracy = SD is introducing errors. This is one
of the cleanest ways to quantify what DeGF is actually doing.

---

## What NOT to over-report

- Do not lead with regex accuracy if judge kappa is low — judge accuracy is the
  primary metric when available
- Do not report accuracy without F1 given class imbalance
- Do not claim DeGF "works" based on overall accuracy gain if:
  a) First-pass accuracy analysis shows the gain was already there in baseline, OR
  b) Jaccard analysis shows failure set is unchanged, OR
  c) The gain is only on the majority class (aground) and not on the minority
- Do not ignore inter-judge disagreement — report Fleiss' kappa prominently

---

## Figures to include in paper

1. Main accuracy table (Table 1) — model × method × prompt with F1
2. Delta vs baseline table (Table 2)
3. Per-class F1 breakdown (Table 3 or Figure)
4. Confusion matrices — one per model, averaged across methods and prompts
5. DeGF first-pass vs final bar chart — uplift / regression / no-effect breakdown
6. CLIP similarity vs accuracy scatter (DeGF) — Analysis 2
7. Prompt sensitivity heatmap — Analysis 4
8. Tier distribution pie/bar — Tier 1/2/3 counts with sub-type breakdown
9. SD flip gallery — 3-4 example images: original + SD reference + model outputs

---

## Potential contributions depending on findings

| Finding | Potential contribution |
|---|---|
| DeGF improves accuracy consistently across both models | Validates DeGF generalization beyond LLaVA |
| CLIP similarity predicts DeGF success | Motivates CLIP-gated DeGF as future work |
| Prompt-split rate lower with DeGF than baseline | New metric for visual grounding robustness |
| Wrong-set Jaccard shows methods fail on same images | Motivates ensemble or method combination |
| Tier 2 images reveal GT labeling errors | Dataset contribution: revised GT labels |
| Inter-judge kappa is low | Methodological finding: LLM-as-judge unreliable for this task |
| ONLY outperforms DeGF on Qwen | Architecture-specific finding: single-layer suppression scales better |

---

## Updates

_Add entries as results come in. Note date, run_name, and what was found._
