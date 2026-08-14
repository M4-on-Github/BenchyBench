#!/usr/bin/env python3
"""
visual_classification — Phase 4: outcome analysis and HTML report

Two phases, one script:

  --phase outcome   Merge judge results (if available), compute per-image tiers,
                    CLIP plausibility, prompt stability, confusion asymmetry.
                    Writes all tier CSVs and eval/outcome_analysis/per_image.csv.

  --phase report    Reads all CSVs; renders report.html with 14 sections.

Usage:
  python aggregate_report.py --phase outcome \
      --output-root /data/$USER/.../run01 \
      --benchybench-root /path/to/BenchyBench \
      [--run-name RUN_NAME]

  python aggregate_report.py --phase report \
      --output-root /data/$USER/.../run01 \
      --benchybench-root /path/to/BenchyBench \
      --run-name RUN_NAME
"""

import argparse
import base64
import csv
import io
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, StatisticsError


VALID_STATES = ["aground", "capsized", "on_fire", "sunken"]


# ─────────────────────────────────────────────────────────────────────────────
# Shared I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_csv(path):
    """Read a CSV into a list of dicts. Missing file yields [], not an error.

    Report sections degrade rather than fail: a run submitted with --skip-judge
    has no judge CSVs, and every other section should still render.
    """
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    """Write `rows` (list of dicts) to `path`, creating parent directories.

    Columns are taken from the first row, so callers must supply uniform keys.
    Empty input writes an empty file rather than skipping: a present-but-empty
    file distinguishes "this analysis ran and found nothing" from "this
    analysis never ran", which matters when reading a run directory later.
    """
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _bool(v):
    """Coerce a CSV field to bool.

    Needed because CSV round-trips lose types: a bool written as True comes
    back as the string "True", which is truthy either way — but so is "False".
    Anything outside the accepted set is False, so a blank or absent column
    reads as False rather than raising.
    """
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


# ─────────────────────────────────────────────────────────────────────────────
# Phase outcome — helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_judge_consensus(eval_castor_root, run_name):
    """
    Load judge consensus JSONL and return a lookup keyed on
    (image, model_tag, method, prompt_stem) → {judge_verdict, judge_score, judge_state_correct}.
    """
    consensus_dir = eval_castor_root / "results" / "p5_judge" / run_name
    if not consensus_dir.exists():
        return {}

    records = {}
    for path in consensus_dir.glob("*_consensus.jsonl"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                image       = rec.get("image", "")
                model_tag   = rec.get("model_tag", "")
                method      = rec.get("method", "")
                prompt_stem = rec.get("prompt_stem", "")
                # Fallback: parse from record_id if passthrough fields are absent
                if not model_tag and "record_id" in rec:
                    parts = rec["record_id"].split("||")
                    if len(parts) == 4:
                        image, model_tag, method, prompt_stem = parts
                key = (image, model_tag, method, prompt_stem)
                fc = rec.get("field_consensus") or {}
                records[key] = {
                    "judge_verdict":              rec.get("judge_verdict"),
                    "judge_score":                rec.get("mean_score"),
                    "judge_state_correct":        fc.get("state_correct"),
                    "judge_vessel_type_correct":  fc.get("vessel_type_correct"),
                    "judge_size_correct":         fc.get("size_correct"),
                    "judge_cargo_correct":        fc.get("cargo_correct"),
                }
    return records


class OutcomeTier:
    """How the combinations agreed on one image.

    The tiers partition images by consensus, and each answers a different
    question:

      Tier 1  every combination correct   the task's floor; no signal here
      Tier 2  every combination wrong     often a ground-truth problem rather
                                          than a model one — when nothing gets
                                          an image right, suspect the label
      Tier 3  combinations disagree       where method comparison actually lives

    Tier 3 is the point of the analysis. A run-level accuracy delta says one
    method beat another; the tier-3 sub-types say on which images and along
    which axis.

    NO_GT is deliberately not a tier. An image without ground truth is
    unscored, not universally failed, and folding it into tier 2 would inflate
    the apparent failure count.
    """

    ALL_CORRECT = 1
    ALL_WRONG = 2
    CONTESTED = 3

    CONSENSUS = {
        ALL_CORRECT: "all_correct",
        ALL_WRONG:   "all_wrong",
        CONTESTED:   "contested",
    }
    NO_GT_CONSENSUS = "no_gt"

    @classmethod
    def classify(cls, n_correct, n_with_gt):
        """Return (tier, consensus) for one image. Tier is None without GT."""
        if n_with_gt == 0:
            return None, cls.NO_GT_CONSENSUS
        if n_correct == n_with_gt:
            tier = cls.ALL_CORRECT
        elif n_correct == 0:
            tier = cls.ALL_WRONG
        else:
            tier = cls.CONTESTED
        return tier, cls.CONSENSUS[tier]

    @staticmethod
    def difficulty(n_correct, n_with_gt):
        """Fraction of scored combinations that got this image wrong.

        None without ground truth — 0.0 would read as 'every combination was
        right', the opposite of 'unknown'.
        """
        if not n_with_gt:
            return None
        return 1 - (n_correct / n_with_gt)


class SubType:
    """Vocabulary for why tier-3 combinations disagreed.

    Multi-label: one image can split along several axes at once.

    Note the asymmetry. MODEL_SPLIT and METHOD_SPLIT key on *correctness*
    differing. PROMPT_SPLIT keys on the predicted *label* differing — it
    detects label instability across prompts, which is a distinct phenomenon
    from one prompt happening to be right. COMBO_SPLIT is the catch-all for
    disagreement matching none of the above, so it should not be read as a
    finding in itself.
    """

    MODEL_SPLIT = "model_split"
    METHOD_SPLIT = "method_split"
    METHOD_REGRESSION = "method_regression"
    PROMPT_SPLIT = "prompt_split"
    COMBO_SPLIT = "combo_split"


def compute_per_image_tiers(rows):
    """
    Group rows by image, compute per-image tier and sub-types.

    See OutcomeTier for what the tiers mean and SubType for the sub-type
    vocabulary, including why prompt_split keys on labels rather than
    correctness.
    """
    image_groups = defaultdict(list)
    for row in rows:
        image_groups[row["image"]].append(row)

    results = []
    for image, image_rows in image_groups.items():
        gt_label = next((r["gt_label"] for r in image_rows if r.get("gt_label")), None)
        n = len(image_rows)
        n_correct = sum(1 for r in image_rows if _bool(r.get("regex_correct", False)))
        n_with_gt = sum(1 for r in image_rows if r.get("gt_label"))

        difficulty_score = OutcomeTier.difficulty(n_correct, n_with_gt)
        tier, consensus = OutcomeTier.classify(n_correct, n_with_gt)

        # Sub-types (Tier 3 only)
        sub_types = []
        if tier == OutcomeTier.CONTESTED:
            # model_split: does correctness vary by model (same method+prompt)?
            mp_groups = defaultdict(list)
            for r in image_rows:
                mp_groups[(r["method"], r["prompt_stem"])].append(r)
            for (meth, ps), grp in mp_groups.items():
                models = {r["model_tag"] for r in grp}
                if len(models) > 1:
                    corr_by_model = {r["model_tag"]: _bool(r.get("regex_correct")) for r in grp}
                    if len(set(corr_by_model.values())) > 1:
                        sub_types.append(SubType.MODEL_SPLIT)
                        break

            # method_split: does correctness vary by method (same model+prompt)?
            for r_model in {r["model_tag"] for r in image_rows}:
                for r_prompt in {r["prompt_stem"] for r in image_rows}:
                    grp = [r for r in image_rows if r["model_tag"] == r_model and r["prompt_stem"] == r_prompt]
                    methods = {r["method"] for r in grp}
                    if len(methods) > 1:
                        corr = {r["method"]: _bool(r.get("regex_correct")) for r in grp}
                        if len(set(corr.values())) > 1:
                            sub_types.append(SubType.METHOD_SPLIT)
                            break
                else:
                    continue
                break

            # method_regression: baseline correct, but degf or only wrong
            for r_model in {r["model_tag"] for r in image_rows}:
                for r_prompt in {r["prompt_stem"] for r in image_rows}:
                    grp = {r["method"]: r for r in image_rows
                           if r["model_tag"] == r_model and r["prompt_stem"] == r_prompt}
                    if "baseline" in grp:
                        baseline_ok = _bool(grp["baseline"].get("regex_correct"))
                        any_method_fail = any(
                            not _bool(r.get("regex_correct"))
                            for m, r in grp.items() if m != "baseline"
                        )
                        if baseline_ok and any_method_fail:
                            sub_types.append(SubType.METHOD_REGRESSION)
                            break
                else:
                    continue
                break

            # prompt_split: same model+method, different predicted labels across prompts
            for r_model in {r["model_tag"] for r in image_rows}:
                for r_method in {r["method"] for r in image_rows}:
                    grp = [r for r in image_rows if r["model_tag"] == r_model and r["method"] == r_method]
                    labels = {r.get("parsed_label") for r in grp if r.get("parsed_label")}
                    if len(labels) > 1:
                        sub_types.append(SubType.PROMPT_SPLIT)
                        break
                else:
                    continue
                break

            if not sub_types:
                sub_types = [SubType.COMBO_SPLIT]

        sub_types = sorted(set(sub_types))

        # Failure type labeling (from health flags and pattern)
        all_predicted = [r.get("parsed_label") for r in image_rows if r.get("parsed_label")]
        most_common_pred = max(set(all_predicted), key=all_predicted.count) if all_predicted else None

        failure_types = []
        if tier == OutcomeTier.ALL_WRONG:
            # All wrong — classify by what the model predicted instead
            if most_common_pred and most_common_pred != gt_label:
                failure_types.append(f"systematic_misclass_as_{most_common_pred}")
        if tier == OutcomeTier.CONTESTED and SubType.METHOD_REGRESSION in sub_types:
            failure_types.append("method_induced_regression")
        if tier == OutcomeTier.CONTESTED and SubType.PROMPT_SPLIT in sub_types:
            failure_types.append("prompt_sensitive")
        if tier == OutcomeTier.CONTESTED and SubType.MODEL_SPLIT in sub_types:
            failure_types.append("model_sensitive")
        if any(_bool(r.get("hedge_detected")) for r in image_rows):
            failure_types.append("hedge")
        if any(_bool(r.get("refusal_detected")) for r in image_rows):
            failure_types.append("refusal")
        n_parse_fail = sum(1 for r in image_rows if not _bool(r.get("parse_success", True)))
        if n_parse_fail > n // 2:
            failure_types.append("parse_fail")

        primary_failure_type = failure_types[0] if failure_types else None

        # ── Judge-based tier (parallel to regex tier, using judge_state_correct) ──
        j_rows = [r for r in image_rows if r.get("judge_state_correct") is not None]
        n_j = len(j_rows)
        n_j_correct = sum(1 for r in j_rows if _bool(r.get("judge_state_correct")))
        if n_j == 0:
            judge_tier = None
            judge_consensus = None
        elif n_j_correct == n_j:
            judge_tier = 1
            judge_consensus = "all_correct"
        elif n_j_correct == 0:
            judge_tier = 2
            judge_consensus = "all_wrong"
        else:
            judge_tier = 3
            judge_consensus = "contested"

        tier_disagree = (tier != judge_tier) if (tier is not None and judge_tier is not None) else None

        results.append({
            "image": image,
            "gt_label": gt_label,
            "n_combos": n,
            "n_correct": n_correct,
            "difficulty_score": round(difficulty_score, 4) if difficulty_score is not None else None,
            "tier": tier,
            "consensus": consensus,
            "sub_types": "|".join(sub_types) if sub_types else None,
            "primary_failure_type": primary_failure_type,
            "failure_types": "|".join(failure_types) if failure_types else None,
            "judge_tier": judge_tier,
            "judge_consensus": judge_consensus,
            "judge_n_correct": n_j_correct,
            "judge_n": n_j,
            "tier_disagree": tier_disagree,
        })

    return results


def compute_prompt_stability(rows):
    """
    Kendall's tau correlation of prompt accuracy rankings across (model × method) combos.
    Returns rows for eval/outcome_analysis/prompt_stability.csv.
    """
    # Build {(model, method): {prompt_stem: accuracy}} mapping
    combo_prompt_acc = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["model_tag"], row["method"])
        combo_prompt_acc[key][row["prompt_stem"]].append(_bool(row.get("regex_correct", False)))

    # Average accuracy per (model, method, prompt)
    combo_accs = {}
    for combo, prompt_recs in combo_prompt_acc.items():
        combo_accs[combo] = {ps: mean(vals) for ps, vals in prompt_recs.items()}

    # All prompts (consistent ordering)
    all_prompts = sorted({ps for acc_map in combo_accs.values() for ps in acc_map})
    if len(all_prompts) < 2:
        return []

    # Compute Kendall's tau between each pair of combos
    def _kendall_tau(rank_a: list, rank_b: list):
        """Kendall's tau-a between two rankings: +1 identical, -1 reversed, 0 unrelated.

        Measures whether two (model, method) combinations rank the prompts the
        same way. High tau means prompt quality is a property of the prompt;
        low tau means it is entangled with the model, and "best prompt" is not
        a transferable claim.

        This is tau-a, which makes no tie correction — accuracies are
        coarse-grained and ties are common, so values are conservative
        (biased toward 0) when several prompts score identically.
        """
        n = len(rank_a)
        concordant = discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                sign_a = rank_a[i] - rank_a[j]
                sign_b = rank_b[i] - rank_b[j]
                if sign_a * sign_b > 0:
                    concordant += 1
                elif sign_a * sign_b < 0:
                    discordant += 1
        denom = n * (n - 1) / 2
        return (concordant - discordant) / denom if denom > 0 else 0

    combos = sorted(combo_accs.keys())
    results = []
    for i, c1 in enumerate(combos):
        for c2 in combos[i+1:]:
            acc1 = [combo_accs[c1].get(ps, 0) for ps in all_prompts]
            acc2 = [combo_accs[c2].get(ps, 0) for ps in all_prompts]
            tau = _kendall_tau(acc1, acc2)
            results.append({
                "combo_a": f"{c1[0]}×{c1[1]}",
                "combo_b": f"{c2[0]}×{c2[1]}",
                "kendall_tau": round(tau, 4),
                "n_prompts": len(all_prompts),
            })
    return results


def compute_clip_similarities(rows, benchybench_root, output_root=None):
    """
    Compute CLIP cosine similarity between original shipwreck image and DeGF SD image.
    SD image path is looked up in two ways (in order):
      1. degf_sd_image_path field in the record (explicit stored path).
      2. output_root/inference/sd_images/{image_flat}.png reconstructed from image name.
    Returns {image_key: similarity_score}.
    """
    sd_images_dir = (output_root / "inference" / "sd_images") if output_root else None
    degf_rows = [r for r in rows if r.get("method") == "degf"]
    if not degf_rows:
        return {}

    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        print("  [CLIP] skipping — torch/transformers/Pillow not available")
        return {}

    device = "cpu"
    print("  [CLIP] loading openai/clip-vit-base-patch32 …")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    image_dir = benchybench_root / "shipwreck_wiki_images" / "sorted_images"
    results = {}
    seen_sd = {}  # image → sd_path (SD is prompt-independent; cache per image)

    with torch.no_grad():
        for row in degf_rows:
            img_field = row["image"]
            orig_path = image_dir / img_field

            # Resolve SD image path — per-run subfolder: sd_images/{model}_degf_{prompt_stem}/
            sd_path = None
            stored = row.get("degf_sd_image_path")
            if stored:
                sd_path = Path(stored)
            elif sd_images_dir:
                cache_key = (img_field, row.get("model_tag", ""), row.get("prompt_stem", ""))
                if cache_key not in seen_sd:
                    img_flat = os.path.splitext(img_field.replace("/", "_").replace("\\", "_"))[0]
                    subfolder = f"{row.get('model_tag', 'unknown')}_degf_{row.get('prompt_stem', 'unknown')}"
                    candidate = sd_images_dir / subfolder / (img_flat + ".png")
                    seen_sd[cache_key] = candidate if candidate.exists() else None
                sd_path = seen_sd.get(cache_key)

            if sd_path is None or not orig_path.exists() or not sd_path.exists():
                continue
            try:
                orig_img = Image.open(orig_path).convert("RGB")
                sd_img = Image.open(sd_path).convert("RGB")
                inputs = processor(images=[orig_img, sd_img], return_tensors="pt").to(device)
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                sim = float(torch.dot(features[0], features[1]).item())
                key = f"{img_field}|{row['model_tag']}|{row['method']}|{row['prompt_stem']}"
                results[key] = round(sim, 4)
            except Exception as e:
                print(f"  [CLIP] skipping {img_field}: {e}")

    print(f"  [CLIP] computed {len(results)} similarities")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase report — HTML generation
# ─────────────────────────────────────────────────────────────────────────────

def _img_to_b64(path, size = 150):
    """Load image, thumbnail to {size}px, return base64 data URI."""
    try:
        from PIL import Image as PilImage
        img = PilImage.open(path).convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def _acc_table_html(summary_rows):
    combos_by_model = defaultdict(list)
    for row in summary_rows:
        combos_by_model[row["model_tag"]].append(row)

    prompts = sorted({r["prompt_stem"] for r in summary_rows})
    methods = ["baseline", "degf", "only"]

    html = ["<table class='data-table'>",
            "<thead><tr><th>Model</th><th>Method</th>"]
    for ps in prompts:
        html.append(f"<th>{ps}<br><small>(acc / F1)</small></th>")
    html.append("<th>Avg acc</th></tr></thead><tbody>")

    for model in sorted(combos_by_model):
        for method in methods:
            recs = [r for r in combos_by_model[model] if r["method"] == method]
            cells = []
            accs = []
            for ps in prompts:
                rec = next((r for r in recs if r["prompt_stem"] == ps), None)
                if rec:
                    acc = float(rec["accuracy"]) if rec.get("accuracy") else 0
                    f1 = float(rec["macro_f1"]) if rec.get("macro_f1") else 0
                    accs.append(acc)
                    pct_acc = f"{acc*100:.1f}%"
                    pct_f1 = f"{f1*100:.1f}%"
                    cells.append(f"<td>{pct_acc} / {pct_f1}</td>")
                else:
                    cells.append("<td>—</td>")
            avg = f"{mean(accs)*100:.1f}%" if accs else "—"
            html.append(f"<tr><td>{model}</td><td>{method}</td>")
            html.extend(cells)
            html.append(f"<td><b>{avg}</b></td></tr>")

    html.append("</tbody></table>")
    return "\n".join(html)


def _tier_summary_html(per_image_rows):
    tier_counts = defaultdict(int)
    sub_counts = defaultdict(int)
    for row in per_image_rows:
        t = row.get("tier")
        if t:
            tier_counts[str(t)] += 1
        subs = (row.get("sub_types") or "").split("|")
        for s in subs:
            if s:
                sub_counts[s] += 1
    total = sum(tier_counts.values())

    html = ["<table class='data-table'><thead><tr><th>Tier</th><th>Count</th><th>%</th><th>Description</th></tr></thead><tbody>"]
    descs = {
        "1": "Universal success — all combos correct",
        "2": "Universal failure — all combos wrong",
        "3": "Contested — some combos correct, some wrong",
    }
    for tier in ["1", "2", "3"]:
        n = tier_counts.get(tier, 0)
        pct = f"{100*n/total:.1f}%" if total else "—"
        html.append(f"<tr><td>Tier {tier}</td><td>{n}</td><td>{pct}</td><td>{descs[tier]}</td></tr>")
    html.append("</tbody></table>")

    if sub_counts:
        html.append("<h3>Tier 3 sub-types</h3>")
        html.append("<table class='data-table'><thead><tr><th>Sub-type</th><th>Images</th></tr></thead><tbody>")
        for k, v in sorted(sub_counts.items(), key=lambda x: -x[1]):
            html.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
        html.append("</tbody></table>")

    return "\n".join(html)


def _confusion_html(summary_rows, per_record_rows):
    # One confusion matrix per model (averaged across methods and prompts)
    html = []
    for model in sorted({r["model_tag"] for r in per_record_rows}):
        model_rows = [r for r in per_record_rows if r["model_tag"] == model]
        counts = {s: {s2: 0 for s2 in VALID_STATES + ["UNPARSEABLE"]}
                                    for s in VALID_STATES}
        for r in model_rows:
            gt = r.get("gt_label", "")
            pred = r.get("parsed_label") or "UNPARSEABLE"
            if gt in counts:
                if pred in counts[gt]:
                    counts[gt][pred] += 1
                else:
                    counts[gt]["UNPARSEABLE"] += 1

        html.append(f"<h3>Confusion matrix — {model}</h3>")
        html.append("<table class='data-table'><thead><tr><th>GT \\ Pred</th>")
        cols = VALID_STATES + ["UNPARSEABLE"]
        for col in cols:
            html.append(f"<th>{col}</th>")
        html.append("</tr></thead><tbody>")
        for gt in VALID_STATES:
            html.append(f"<tr><td><b>{gt}</b></td>")
            for col in cols:
                n = counts[gt][col]
                style = " style='background:#cfc'" if col == gt and n > 0 else (
                    " style='background:#fcc'" if col != gt and n > 0 else "")
                html.append(f"<td{style}>{n}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")
    return "\n".join(html)


def _update_summary_with_judge(rows, summary_path):
    """Add per-combo judge accuracy columns to summary.csv."""
    JUDGE_FIELDS = [
        ("judge_state_correct",       "judge_state_accuracy"),
        ("judge_vessel_type_correct", "judge_vessel_type_accuracy"),
        ("judge_size_correct",        "judge_size_accuracy"),
        ("judge_cargo_correct",       "judge_cargo_accuracy"),
    ]

    # Group rows by combo
    combos = defaultdict(list)
    for r in rows:
        combos[(r["model_tag"], r["method"], r["prompt_stem"])].append(r)

    def _kappa_combo(recs):
        pairs = [(1 if _bool(r.get("regex_correct")) else 0,
                  1 if _bool(r.get("judge_state_correct")) else 0)
                 for r in recs if r.get("judge_state_correct") is not None]
        if not pairs: return None
        n = len(pairs)
        p_o = sum(a == b for a, b in pairs) / n
        p_r = sum(a for a, _ in pairs) / n
        p_j = sum(b for _, b in pairs) / n
        p_e = p_r * p_j + (1 - p_r) * (1 - p_j)
        return round((p_o - p_e) / (1 - p_e), 4) if (1 - p_e) > 0 else 1.0

    summary = read_csv(summary_path)
    for row in summary:
        key = (row["model_tag"], row["method"], row["prompt_stem"])
        recs = combos.get(key, [])
        j_recs = [r for r in recs if r.get("judge_state_correct") is not None]
        row["judge_n"] = len(j_recs)
        for src_field, col in JUDGE_FIELDS:
            n_correct = sum(1 for r in j_recs if _bool(r.get(src_field)))
            row[col] = round(n_correct / len(j_recs), 4) if j_recs else None
        row["judge_state_kappa"] = _kappa_combo(recs)
    write_csv(summary_path, summary)
    print(f"  summary.csv updated with judge columns ({len(summary)} rows)")


def _judge_section_html(per_record_rows):
    judge_rows = [r for r in per_record_rows if r.get("judge_verdict") is not None]
    if not judge_rows:
        return "<p>No judge data available for this run.</p>"

    prompts = sorted({r["prompt_stem"] for r in per_record_rows})
    models  = sorted({r["model_tag"]   for r in per_record_rows})
    methods = ["baseline", "degf", "only"]

    # Per-combo tallies
    combos = defaultdict(lambda: {"n": 0, "j_correct": 0, "r_correct": 0,
                                  "both_correct": 0, "both_wrong": 0, "disagree": 0})
    for r in per_record_rows:
        if r.get("judge_state_correct") is None:
            continue
        key = (r["model_tag"], r["method"], r["prompt_stem"])
        c = combos[key]
        c["n"] += 1
        j_ok = _bool(r.get("judge_state_correct"))
        r_ok = _bool(r.get("regex_correct"))
        if j_ok: c["j_correct"] += 1
        if r_ok: c["r_correct"] += 1
        if j_ok and r_ok:      c["both_correct"] += 1
        elif not j_ok and not r_ok: c["both_wrong"]  += 1
        else:                       c["disagree"]    += 1

    # Per-combo tallies for all four fields
    field_combos = defaultdict(lambda: defaultdict(lambda: {"n": 0, "correct": 0}))
    for r in per_record_rows:
        key = (r["model_tag"], r["method"], r["prompt_stem"])
        for fld in ["judge_state_correct", "judge_vessel_type_correct",
                    "judge_size_correct", "judge_cargo_correct"]:
            if r.get(fld) is not None:
                field_combos[fld][key]["n"] += 1
                if _bool(r.get(fld)):
                    field_combos[fld][key]["correct"] += 1

    JUDGE_FIELDS = [
        ("judge_state_correct",       "State"),
        ("judge_vessel_type_correct", "Vessel type"),
        ("judge_size_correct",        "Size"),
        ("judge_cargo_correct",       "Cargo"),
    ]

    # ── Per-field accuracy tables ─────────────────────────────────────────────
    html = []
    for fld, label in JUDGE_FIELDS:
        html.append(f"<h3>Judge accuracy — {label}</h3>")
        html.append("<table class='data-table'><thead><tr><th>Model</th><th>Method</th>")
        for ps in prompts:
            html.append(f"<th>{ps}</th>")
        html.append("<th>Avg</th></tr></thead><tbody>")
        for model in models:
            for method in methods:
                cells, accs = [], []
                for ps in prompts:
                    c = field_combos[fld].get((model, method, ps))
                    if c and c["n"]:
                        acc = c["correct"] / c["n"]
                        accs.append(acc)
                        cells.append(f"<td>{acc*100:.1f}%</td>")
                    else:
                        cells.append("<td>—</td>")
                avg = f"{mean(accs)*100:.1f}%" if accs else "—"
                html.append(f"<tr><td>{model}</td><td>{method}</td>")
                html.extend(cells)
                html.append(f"<td><b>{avg}</b></td></tr>")
        html.append("</tbody></table>")

    # ── Regex–Judge tier disagreement summary ────────────────────────────────
    # Count images where regex tier ≠ judge tier (from per_record_rows proxy)
    image_regex = defaultdict(list)
    image_judge = defaultdict(list)
    for r in per_record_rows:
        if r.get("judge_state_correct") is not None:
            image_regex[r["image"]].append(_bool(r.get("regex_correct")))
            image_judge[r["image"]].append(_bool(r.get("judge_state_correct")))
    disagree_count = 0
    for img in image_regex:
        r_tier = sum(image_regex[img]) == len(image_regex[img])  # all correct?
        j_tier = sum(image_judge[img]) == len(image_judge[img])
        if r_tier != j_tier:
            disagree_count += 1
    n_images = len(image_regex)
    if n_images:
        html.append(f"<p><b>Regex–Judge tier disagreement:</b> {disagree_count} / {n_images} images "
                    f"({100*disagree_count/n_images:.1f}%) have different tier assignments. "
                    f"See <code>per_image.csv</code> column <code>tier_disagree</code> for details.</p>")

    # ── Regex–Judge Cohen's κ ─────────────────────────────────────────────────
    def _kappa(c):
        """Cohen's kappa between the regex and judge verdicts for one combination.

        Chance-corrected agreement, which raw agreement is not: when a
        combination is 90% correct, regex and judge agree ~82% of the time by
        chance alone, so a high raw agreement can mean nothing.

        Returns None for an empty combination. Returns 1.0 in the degenerate
        case where expected agreement is 1 — both raters always give the same
        verdict, leaving no variance to disagree over — since 0/0 is otherwise
        undefined and reporting perfect agreement is the honest reading.
        """
        n = c["n"]
        if n == 0: return None
        p_o = (c["both_correct"] + c["both_wrong"]) / n
        p_r  = c["r_correct"] / n
        p_j  = c["j_correct"] / n
        p_e  = p_r * p_j + (1 - p_r) * (1 - p_j)
        return (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0

    def _kappa_color(k):
        if k is None: return ""
        return "green" if k >= 0.8 else ("darkorange" if k >= 0.6 else "red")

    html.append("<h3>Regex–Judge agreement (Cohen's κ)</h3>")
    html.append("<table class='data-table'><thead><tr>"
                "<th>Model</th><th>Method</th><th>Prompt</th>"
                "<th>N</th><th>κ</th><th>Agree</th><th>Disagree</th>"
                "</tr></thead><tbody>")

    all_r, all_j = [], []
    for model in models:
        for method in methods:
            for ps in prompts:
                c = combos.get((model, method, ps))
                if not c or not c["n"]: continue
                k = _kappa(c)
                agree    = c["both_correct"] + c["both_wrong"]
                disagree = c["disagree"]
                col = _kappa_color(k)
                html.append(
                    f"<tr><td>{model}</td><td>{method}</td><td>{ps}</td>"
                    f"<td>{c['n']}</td>"
                    f"<td style='color:{col}'>{k:.3f}</td>"
                    f"<td>{agree} ({100*agree/c['n']:.0f}%)</td>"
                    f"<td>{disagree}</td></tr>"
                )

    # Collect for overall κ
    for r in per_record_rows:
        if r.get("judge_state_correct") is None: continue
        all_r.append(1 if _bool(r.get("regex_correct"))        else 0)
        all_j.append(1 if _bool(r.get("judge_state_correct"))  else 0)

    if all_r:
        n = len(all_r)
        agree = sum(a == b for a, b in zip(all_r, all_j))
        p_o = agree / n
        p_r = sum(all_r) / n; p_j = sum(all_j) / n
        p_e = p_r * p_j + (1 - p_r) * (1 - p_j)
        k_overall = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 1.0
        col = _kappa_color(k_overall)
        html.append(
            f"<tr style='font-weight:bold'><td colspan='3'>Overall</td>"
            f"<td>{n}</td>"
            f"<td style='color:{col}'>{k_overall:.3f}</td>"
            f"<td>{agree} ({100*agree/n:.0f}%)</td>"
            f"<td>{n - agree}</td></tr>"
        )

    html.append("</tbody></table>")
    return "\n".join(html)


def _degf_firstpass_html(per_record_rows):
    degf = [r for r in per_record_rows if r.get("method") == "degf"
            and r.get("degf_first_pass_label")]
    if not degf:
        return "<p>No DeGF first-pass data available.</p>"

    uplift = sum(1 for r in degf
                 if not _bool(r.get("degf_first_pass_correct")) and _bool(r.get("regex_correct")))
    regress = sum(1 for r in degf
                  if _bool(r.get("degf_first_pass_correct")) and not _bool(r.get("regex_correct")))
    no_effect = len(degf) - uplift - regress

    html = [f"<p>Total DeGF records with first-pass data: {len(degf)}</p>",
            "<table class='data-table'><thead><tr><th>Outcome</th><th>Count</th><th>%</th></tr></thead><tbody>"]
    for label, n in [("SD guidance helped (uplift)", uplift),
                     ("SD guidance hurt (regression)", regress),
                     ("No effect", no_effect)]:
        pct = f"{100*n/len(degf):.1f}%" if degf else "—"
        html.append(f"<tr><td>{label}</td><td>{n}</td><td>{pct}</td></tr>")
    html.append("</tbody></table>")
    return "\n".join(html)


def _render_html_report(output_root, run_name, benchybench_root):
    eval_dir = output_root / "eval"
    per_record_rows = read_csv(eval_dir / "regex" / "per_record.csv")
    summary_rows = read_csv(eval_dir / "regex" / "summary.csv")
    per_image_rows = read_csv(eval_dir / "outcome_analysis" / "per_image.csv")
    prompt_stab_rows = read_csv(eval_dir / "outcome_analysis" / "prompt_stability.csv")

    image_dir = benchybench_root / "shipwreck_wiki_images" / "sorted_images"

    # Tier-2 image gallery (all combos wrong)
    tier2_rows = [r for r in per_image_rows if str(r.get("tier")) == "2"]
    tier2_gallery = ""
    for r in tier2_rows[:20]:
        img_path = image_dir / r["image"]
        b64 = _img_to_b64(img_path)
        if b64:
            tier2_gallery += (
                f'<figure style="display:inline-block;margin:4px;text-align:center">'
                f'<img src="{b64}" style="height:150px"><br>'
                f'<figcaption style="font-size:0.75em">{r["image"]}<br>'
                f'GT:{r.get("gt_label","?")}</figcaption></figure>'
            )

    sections = {
        "S1: Overview": f"""
<p><b>Run:</b> {run_name}</p>
<p><b>Output root:</b> {output_root}</p>
<p><b>Total records:</b> {len(per_record_rows)}</p>
<p><b>Unique images:</b> {len({r['image'] for r in per_record_rows})}</p>
<p><b>Combos:</b> {len(set((r['model_tag'],r['method'],r['prompt_stem']) for r in per_record_rows))}</p>
""",
        "S2: Main accuracy table (regex)": _acc_table_html(summary_rows),
        "S3: Judge accuracy & agreement": _judge_section_html(per_record_rows),
        "S4: Outcome tiers": _tier_summary_html(per_image_rows),
        "S5: Confusion matrices": _confusion_html(summary_rows, per_record_rows),
        "S6: DeGF first-pass analysis": _degf_firstpass_html(per_record_rows),
        "S7: Tier-2 universal failure gallery": tier2_gallery or "<p>No Tier-2 images.</p>",
        "S8: Prompt stability (Kendall's tau)": _prompt_stab_html(prompt_stab_rows),
        "S9: Per-class breakdown": _per_class_html(per_record_rows),
    }

    # Assemble HTML
    nav = "\n".join(
        f'<a href="#{i}" class="nav-link">{title}</a>'
        for i, title in enumerate(sections)
    )

    body_parts = []
    for i, (title, content) in enumerate(sections.items()):
        body_parts.append(f'<section id="{i}"><h2>{title}</h2>{content}</section>')

    css = """
body{font-family:system-ui,sans-serif;margin:0;padding:0;background:#f9f9f9;color:#222}
.sidebar{position:fixed;top:0;left:0;width:220px;height:100vh;overflow-y:auto;
  background:#2c3e50;padding:12px;box-sizing:border-box}
.sidebar h1{color:#ecf0f1;font-size:0.9em;margin:0 0 12px}
.nav-link{display:block;color:#bdc3c7;padding:4px 6px;text-decoration:none;font-size:0.8em;border-radius:3px}
.nav-link:hover{background:#34495e;color:#fff}
main{margin-left:230px;padding:20px;max-width:1100px}
h2{border-bottom:2px solid #2c3e50;padding-bottom:4px;color:#2c3e50}
h3{color:#34495e}
.data-table{border-collapse:collapse;margin:12px 0;font-size:0.85em;overflow-x:auto;display:block}
.data-table th,.data-table td{border:1px solid #ccc;padding:4px 8px}
.data-table th{background:#2c3e50;color:#fff}
.data-table tr:nth-child(even){background:#f2f2f2}
section{margin-bottom:40px;padding-bottom:20px;border-bottom:1px solid #ddd}
@media(prefers-color-scheme:dark){body{background:#1a1a2e;color:#e0e0e0}
  .sidebar{background:#16213e}.nav-link{color:#a0aec0}
  .data-table th{background:#16213e}.data-table tr:nth-child(even){background:#0f3460}
  h2,h3{color:#a0c4ff}}
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>visual_classification — {run_name}</title>
<style>{css}</style>
</head>
<body>
<aside class="sidebar">
  <h1>visual_classification<br>{run_name}</h1>
  {nav}
</aside>
<main>
{"".join(body_parts)}
</main>
</body>
</html>"""
    return html


def _prompt_stab_html(rows):
    if not rows:
        return "<p>Not enough prompts for stability analysis (&lt;2).</p>"
    html = ["<table class='data-table'><thead><tr><th>Combo A</th><th>Combo B</th><th>Kendall's τ</th><th>N prompts</th></tr></thead><tbody>"]
    for row in rows:
        tau = float(row.get("kendall_tau", 0))
        color = "style='color:green'" if tau > 0.6 else ("style='color:red'" if tau < 0.2 else "")
        html.append(f"<tr><td>{row['combo_a']}</td><td>{row['combo_b']}</td>"
                    f"<td {color}>{tau:.3f}</td><td>{row.get('n_prompts','')}</td></tr>")
    html.append("</tbody></table>")
    return "\n".join(html)


def _per_class_html(per_record_rows):
    html = []
    for cls in VALID_STATES:
        cls_rows = [r for r in per_record_rows if r.get("gt_label") == cls]
        if not cls_rows:
            continue
        n = len(cls_rows)
        n_correct = sum(1 for r in cls_rows if _bool(r.get("regex_correct")))
        acc = f"{100*n_correct/n:.1f}%" if n else "—"
        html.append(f"<p><b>{cls}</b>: n={n}, accuracy={acc}</p>")
    return "\n".join(html)


# ─────────────────────────────────────────────────────────────────────────────
# Phase runners
# ─────────────────────────────────────────────────────────────────────────────

def run_outcome(output_root, benchybench_root, run_name):
    """Phase 1 of two: merge judge verdicts and classify per-image outcomes.

    Reads eval/regex/per_record.csv, merges judge consensus when available,
    then writes eval/outcome_analysis/ — per_image.csv plus a CSV per tier.

    Images are grouped by how the combinations agreed:

      Tier 1  every combination correct    — the task's floor
      Tier 2  every combination wrong      — candidates for bad ground truth
      Tier 3  combinations disagree        — where method comparison lives

    Tier 3 is the reason this phase exists. A run-level accuracy delta says one
    method beat another; the tier-3 split says on which images, and whether the
    difference tracks the method, the model, or the prompt. Sub-types are
    multi-label, since an image can split along several axes at once.

    `run_name` is REQUIRED to merge judge output — it keys the consensus files
    under Eval_CASTOR/results/p5_judge/<run_name>/. Omitting it is silent: the
    phase completes normally and every judge column is empty, which is
    indistinguishable from a --skip-judge run. Callers must pass it whenever
    the judge phase ran.
    """
    eval_dir = output_root / "eval"
    outcome_dir = eval_dir / "outcome_analysis"
    outcome_dir.mkdir(parents=True, exist_ok=True)

    per_record_path = eval_dir / "regex" / "per_record.csv"
    if not per_record_path.exists():
        sys.exit(f"ERROR: per_record.csv not found: {per_record_path}\n"
                 f"       Run regex_eval.py first.")

    print(f"Loading per_record.csv …")
    rows = read_csv(per_record_path)
    print(f"  {len(rows)} rows")

    # ── Merge judge results if available ──────────────────────────────────────
    if run_name:
        eval_castor_root = benchybench_root / "Eval_CASTOR"
        print(f"Loading judge consensus for run: {run_name} …")
        judge_map = load_judge_consensus(eval_castor_root, run_name)
        if judge_map:
            print(f"  {len(judge_map)} judge records found")
            n_merged = 0
            for row in rows:
                key = (row["image"], row["model_tag"], row["method"], row["prompt_stem"])
                jdata = judge_map.get(key)
                if jdata:
                    row["judge_verdict"]             = jdata["judge_verdict"]
                    row["judge_score"]               = jdata["judge_score"]
                    row["judge_state_correct"]       = jdata["judge_state_correct"]
                    row["judge_vessel_type_correct"] = jdata["judge_vessel_type_correct"]
                    row["judge_size_correct"]        = jdata["judge_size_correct"]
                    row["judge_cargo_correct"]       = jdata["judge_cargo_correct"]
                    n_merged += 1
            print(f"  Merged {n_merged} judge verdicts into per_record rows")
            write_csv(per_record_path, rows)
            summary_path = eval_dir / "regex" / "summary.csv"
            if summary_path.exists():
                _update_summary_with_judge(rows, summary_path)
        else:
            print("  No judge results found (judge may still be running, or --skip-judge used)")

    # ── Per-image tier analysis ───────────────────────────────────────────────
    print("Computing per-image tiers …")
    per_image = compute_per_image_tiers(rows)
    print(f"  {len(per_image)} images")
    tier_counts = defaultdict(int)
    for r in per_image:
        tier_counts[r["tier"]] += 1
    print(f"  Tier 1 (all correct)  : {tier_counts[1]}")
    print(f"  Tier 2 (all wrong)    : {tier_counts[2]}")
    print(f"  Tier 3 (contested)    : {tier_counts[3]}")

    write_csv(outcome_dir / "per_image.csv", per_image)

    # Write tier sub-files
    write_csv(outcome_dir / "tier1_successes.csv",
              [r for r in per_image if r["tier"] == 1])
    write_csv(outcome_dir / "tier2_failures.csv",
              [r for r in per_image if r["tier"] == 2])
    for sub in ["model_split", "method_split", "method_regression", "prompt_split", "combo_split"]:
        write_csv(outcome_dir / f"tier3_{sub}.csv",
                  [r for r in per_image if r["tier"] == 3 and sub in (r.get("sub_types") or "")])

    # ── Prompt stability ──────────────────────────────────────────────────────
    print("Computing prompt stability …")
    stab = compute_prompt_stability(rows)
    write_csv(outcome_dir / "prompt_stability.csv", stab)
    print(f"  {len(stab)} combo pairs")

    # ── CLIP plausibility (DeGF only, if available) ───────────────────────────
    print("Computing CLIP similarities (DeGF SD images) …")
    clip_sims = compute_clip_similarities(rows, benchybench_root, output_root)
    if clip_sims:
        clip_rows = [{"key": k, "clip_similarity": v} for k, v in clip_sims.items()]
        write_csv(outcome_dir / "clip_similarities.csv", clip_rows)

    # Merge CLIP back into per_record rows (initialise for all so fieldnames are uniform)
    for row in rows:
        key = f"{row['image']}|{row['model_tag']}|{row['method']}|{row['prompt_stem']}"
        row["clip_similarity"] = clip_sims.get(key) if row.get("method") == "degf" else None

    # Re-write per_record.csv with all new fields
    write_csv(per_record_path, rows)
    print(f"per_record.csv updated: {per_record_path}")
    print("\nOutcome phase complete.")


def run_report(output_root, benchybench_root, run_name):
    """Phase 2 of two: render the human-readable report from phase 1's CSVs.

    Writes report/ — report.html plus report.json and run_meta.json for
    programmatic use. Reads only files on disk, so it is safe to re-run after
    editing presentation without recomputing any analysis.

    Must run after run_outcome(); it reads eval/outcome_analysis/. Sections
    whose inputs are missing are skipped rather than raising, so a
    --skip-judge run still produces a complete report minus the judge
    sections.

    The HTML is self-contained — plots and image thumbnails are inlined as
    base64 — so it can be copied off the cluster as a single file.
    """
    print(f"Generating HTML report for run: {run_name}")
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    html = _render_html_report(output_root, run_name, benchybench_root)
    report_path = report_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"report.html written: {report_path}  ({len(html)//1024} KB)")

    # Summary JSON
    per_record_rows = read_csv(output_root / "eval" / "regex" / "per_record.csv")
    summary_rows = read_csv(output_root / "eval" / "regex" / "summary.csv")
    per_image_rows = read_csv(output_root / "eval" / "outcome_analysis" / "per_image.csv")

    tier_counts = defaultdict(int)
    for r in per_image_rows:
        tier_counts[str(r.get("tier", "none"))] += 1

    run_summary = {
        "run_name": run_name,
        "n_records": len(per_record_rows),
        "n_images": len(per_image_rows),
        "tier_1_count": tier_counts.get("1", 0),
        "tier_2_count": tier_counts.get("2", 0),
        "tier_3_count": tier_counts.get("3", 0),
        "combos": [
            {k: v for k, v in r.items()}
            for r in summary_rows
        ],
    }
    (report_dir / "report.json").write_text(json.dumps(run_summary, indent=2))
    print(f"report.json written: {report_dir / 'report.json'}")
    print("\nReport phase complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=["outcome", "report"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--benchybench-root")
    parser.add_argument("--run-name",
                        help="Required for --phase report; used by --phase outcome to find judge output")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    benchybench_root = (
        Path(args.benchybench_root)
        if args.benchybench_root
        else Path(__file__).resolve().parent.parent
    )

    if args.phase == "outcome":
        run_outcome(output_root, benchybench_root, args.run_name)
    elif args.phase == "report":
        if not args.run_name:
            sys.exit("ERROR: --run-name is required for --phase report")
        run_report(output_root, benchybench_root, args.run_name)


if __name__ == "__main__":
    main()
