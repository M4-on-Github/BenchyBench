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
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
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
                image = rec.get("image", "")
                model_tag = rec.get("model_tag", "")
                method = rec.get("method", "")
                prompt_stem = rec.get("prompt_stem", "")
                key = (image, model_tag, method, prompt_stem)
                records[key] = {
                    "judge_verdict": rec.get("judge_verdict"),
                    "judge_score": rec.get("mean_score"),
                    "judge_state_correct": (
                        rec.get("field_consensus", {}).get("state_correct")
                        if rec.get("field_consensus")
                        else None
                    ),
                }
    return records


def compute_per_image_tiers(rows):
    """
    Group rows by image, compute per-image tier and sub-types.

    Tier 1: all combos correct
    Tier 2: all combos wrong
    Tier 3: contested — with sub-types:
      model_split       correct for one model, wrong for the other (same method+prompt)
      method_split      correct for one method, wrong for another (same model+prompt)
      method_regression correct in baseline but wrong in DeGF or ONLY
      prompt_split      same model+method, different answers across prompts
      combo_split       doesn't fit neatly into above categories
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

        difficulty_score = 1 - (n_correct / n_with_gt) if n_with_gt else None

        # Tier assignment
        if n_with_gt == 0:
            tier = None
            consensus = "no_gt"
        elif n_correct == n_with_gt:
            tier = 1
            consensus = "all_correct"
        elif n_correct == 0:
            tier = 2
            consensus = "all_wrong"
        else:
            tier = 3
            consensus = "contested"

        # Sub-types (Tier 3 only)
        sub_types = []
        if tier == 3:
            # model_split: does correctness vary by model (same method+prompt)?
            mp_groups = defaultdict(list)
            for r in image_rows:
                mp_groups[(r["method"], r["prompt_stem"])].append(r)
            for (meth, ps), grp in mp_groups.items():
                models = {r["model_tag"] for r in grp}
                if len(models) > 1:
                    corr_by_model = {r["model_tag"]: _bool(r.get("regex_correct")) for r in grp}
                    if len(set(corr_by_model.values())) > 1:
                        sub_types.append("model_split")
                        break

            # method_split: does correctness vary by method (same model+prompt)?
            for r_model in {r["model_tag"] for r in image_rows}:
                for r_prompt in {r["prompt_stem"] for r in image_rows}:
                    grp = [r for r in image_rows if r["model_tag"] == r_model and r["prompt_stem"] == r_prompt]
                    methods = {r["method"] for r in grp}
                    if len(methods) > 1:
                        corr = {r["method"]: _bool(r.get("regex_correct")) for r in grp}
                        if len(set(corr.values())) > 1:
                            sub_types.append("method_split")
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
                            sub_types.append("method_regression")
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
                        sub_types.append("prompt_split")
                        break
                else:
                    continue
                break

            if not sub_types:
                sub_types = ["combo_split"]

        sub_types = sorted(set(sub_types))

        # Failure type labeling (from health flags and pattern)
        all_predicted = [r.get("parsed_label") for r in image_rows if r.get("parsed_label")]
        most_common_pred = max(set(all_predicted), key=all_predicted.count) if all_predicted else None

        failure_types = []
        if tier == 2:
            # All wrong — classify by what the model predicted instead
            if most_common_pred and most_common_pred != gt_label:
                failure_types.append(f"systematic_misclass_as_{most_common_pred}")
        if tier == 3 and "method_regression" in sub_types:
            failure_types.append("method_induced_regression")
        if tier == 3 and "prompt_split" in sub_types:
            failure_types.append("prompt_sensitive")
        if tier == 3 and "model_split" in sub_types:
            failure_types.append("model_sensitive")
        if any(_bool(r.get("hedge_detected")) for r in image_rows):
            failure_types.append("hedge")
        if any(_bool(r.get("refusal_detected")) for r in image_rows):
            failure_types.append("refusal")
        n_parse_fail = sum(1 for r in image_rows if not _bool(r.get("parse_success", True)))
        if n_parse_fail > n // 2:
            failure_types.append("parse_fail")

        primary_failure_type = failure_types[0] if failure_types else None

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


def compute_clip_similarities(rows, benchybench_root):
    """
    Compute CLIP cosine similarity between original shipwreck image and DeGF SD image.
    Only runs for DeGF rows that have a valid degf_sd_image_path.
    Returns {image_key: similarity_score}.
    """
    degf_rows = [r for r in rows if r.get("method") == "degf" and r.get("degf_sd_image_path")]
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

    with torch.no_grad():
        for row in degf_rows:
            orig_path = image_dir / row["image"]
            sd_path = Path(row["degf_sd_image_path"])
            if not orig_path.exists() or not sd_path.exists():
                continue
            try:
                orig_img = Image.open(orig_path).convert("RGB")
                sd_img = Image.open(sd_path).convert("RGB")
                inputs = processor(images=[orig_img, sd_img], return_tensors="pt").to(device)
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                sim = float(torch.dot(features[0], features[1]).item())
                key = f"{row['image']}|{row['model_tag']}|{row['method']}|{row['prompt_stem']}"
                results[key] = round(sim, 4)
            except Exception as e:
                print(f"  [CLIP] skipping {row['image']}: {e}")

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
        "S2: Main accuracy table": _acc_table_html(summary_rows),
        "S3: Outcome tiers": _tier_summary_html(per_image_rows),
        "S4: Confusion matrices": _confusion_html(summary_rows, per_record_rows),
        "S5: DeGF first-pass analysis": _degf_firstpass_html(per_record_rows),
        "S6: Tier-2 universal failure gallery": tier2_gallery or "<p>No Tier-2 images.</p>",
        "S7: Prompt stability (Kendall's tau)": _prompt_stab_html(prompt_stab_rows),
        "S8: Per-class breakdown": _per_class_html(per_record_rows),
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
                    row["judge_verdict"] = jdata["judge_verdict"]
                    row["judge_score"] = jdata["judge_score"]
                    row["judge_state_correct"] = jdata["judge_state_correct"]
                    n_merged += 1
            print(f"  Merged {n_merged} judge verdicts into per_record rows")
            # Re-write per_record.csv with judge fields
            write_csv(per_record_path, rows)
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
    clip_sims = compute_clip_similarities(rows, benchybench_root)
    if clip_sims:
        clip_rows = [{"key": k, "clip_similarity": v} for k, v in clip_sims.items()]
        write_csv(outcome_dir / "clip_similarities.csv", clip_rows)

    # Merge CLIP back into per_record rows
    for row in rows:
        if row.get("method") == "degf":
            key = f"{row['image']}|{row['model_tag']}|{row['method']}|{row['prompt_stem']}"
            row["clip_similarity"] = clip_sims.get(key)

    # Re-write per_record.csv with all new fields
    write_csv(per_record_path, rows)
    print(f"per_record.csv updated: {per_record_path}")
    print("\nOutcome phase complete.")


def run_report(output_root, benchybench_root, run_name):
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
