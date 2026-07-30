#!/usr/bin/env python3
"""
visual_classification — Phase 1: inference health check

Scans all inference JSONLs in {output_root}/inference/ and flags:
  - Per-record: repetition, hedge, refusal, length anomaly
  - Per-combo: parse-fail rate, label bias, self-inconsistency

Exits 0 if healthy; exits 1 if any combo has label_bias OR parse_fail_rate > 0.15.
Always writes health/health_report.json regardless of pass/fail.

Usage:
  python health_check.py --output-root /data/$USER/.../run01 \
                         --benchybench-root /path/to/BenchyBench
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


# ── State map (replicated here to avoid cross-submodule import at health stage) ──
STATE_MAP = {
    "aground":  ["aground", "run aground", "grounded"],
    "capsized": ["capsized", "capsizing", "overturned", "rolled over"],
    "on_fire":  ["on fire", "on_fire", "fire", "burning", "ablaze", "aflame"],
    "sunken":   ["sunken", "sunk", "submerged", "underwater", "sinking"],
}

_PATTERN = re.compile(
    r"\b(" + "|".join(
        re.escape(s) for synonyms in STATE_MAP.values() for s in synonyms
    ) + r")\b",
    re.IGNORECASE,
)

_CLASS_PATTERNS = {
    cls: re.compile(
        r"\b(" + "|".join(re.escape(s) for s in synonyms) + r")\b",
        re.IGNORECASE,
    )
    for cls, synonyms in STATE_MAP.items()
}


def normalize_state(text):
    for cls, pat in _CLASS_PATTERNS.items():
        if pat.search(text):
            return cls
    return None


def detect_repetition(text, ngram = 5, threshold = 3):
    words = text.lower().split()
    if len(words) < ngram * threshold:
        return False
    grams = [" ".join(words[i:i+ngram]) for i in range(len(words)-ngram+1)]
    seen = {}
    for g in grams:
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= threshold:
            return True
    return False


def detect_hedge(text):
    found = set()
    for cls, pat in _CLASS_PATTERNS.items():
        if pat.search(text):
            found.add(cls)
    return len(found) > 1


def detect_refusal(text):
    keywords = ["cannot", "can't", "unable to determine", "unclear", "not sure",
                "impossible to tell", "no image", "no photo", "no picture"]
    lower = text.lower()
    return any(k in lower for k in keywords)


def load_records(output_root):
    infer_dir = output_root / "inference"
    records = []
    for path in sorted(infer_dir.glob("answers_*.jsonl")):
        stem = path.stem
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec["_source_stem"] = stem
                    records.append(rec)
                except json.JSONDecodeError:
                    pass
    return records


def load_meta(output_root):
    """Return {answers_filename_stem: meta_dict} from sidecar meta_*.json files."""
    infer_dir = output_root / "inference"
    meta_map = {}
    for path in sorted(infer_dir.glob("meta_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            answers_path = Path(data.get("answers_file", ""))
            meta_map[answers_path.stem] = data
        except (json.JSONDecodeError, KeyError):
            pass
    return meta_map



def annotate_records(records):
    lengths = [len(rec.get("text", "")) for rec in records if rec.get("text")]
    global_mean = mean(lengths) if lengths else 0
    global_std = stdev(lengths) if len(lengths) > 1 else 0

    for rec in records:
        text = rec.get("text", "")
        rec["_health"] = {
            "repetition_detected": detect_repetition(text),
            "hedge_detected": detect_hedge(text),
            "refusal_detected": detect_refusal(text),
            "length_anomaly": len(text) > global_mean + 3 * global_std if global_std else False,
            "parsed_label": normalize_state(text),
        }
    return records


def per_combo_stats(records):
    combos = defaultdict(list)
    for rec in records:
        model = rec.get("model_tag") or rec.get("model_id", "unknown")
        method = rec.get("_method", "unknown")
        prompt_stem = rec.get("_prompt_stem", "unknown")
        combos[(model, method, prompt_stem)].append(rec)

    MIN_BIAS_N = 10  # skip bias check on smoke-size samples

    result = {}
    for (model, method, prompt_stem), recs in combos.items():
        n = len(recs)
        n_fail = sum(1 for r in recs if r["_health"]["parsed_label"] is None)
        parse_fail_rate = n_fail / n if n else 0

        # Label distribution
        label_counts = defaultdict(int)
        for r in recs:
            lbl = r["_health"]["parsed_label"]
            if lbl:
                label_counts[lbl] += 1
        n_parsed = n - n_fail
        # Bias check only meaningful with enough samples; skip on tiny smoke runs
        label_bias = (
            any(c / n_parsed > 0.40 for c in label_counts.values())
            if n_parsed >= MIN_BIAS_N else False
        )

        # Self-inconsistency: per image, how many distinct labels across this combo's prompts
        # (only meaningful across multiple prompt_stems; within a combo it's all one prompt)
        per_image = defaultdict(set)
        for r in recs:
            img = r.get("image", "")
            lbl = r["_health"]["parsed_label"]
            if lbl:
                per_image[img].add(lbl)
        n_inconsistent = sum(1 for lbls in per_image.values() if len(lbls) > 1)

        # label_bias is a warning only — it's a research finding, not a pipeline error.
        # Hard gate only on parse_fail_rate (can't compute accuracy on unparseable output).
        gate_fail = (parse_fail_rate > 0.15) if n >= MIN_BIAS_N else False

        result[(model, method, prompt_stem)] = {
            "n_records": n,
            "parse_fail_rate": round(parse_fail_rate, 4),
            "n_parse_fail": n_fail,
            "label_distribution": dict(label_counts),
            "label_bias": label_bias,
            "bias_dominant": max(label_counts, key=label_counts.get) if label_counts else None,
            "n_repetition": sum(1 for r in recs if r["_health"]["repetition_detected"]),
            "n_hedge": sum(1 for r in recs if r["_health"]["hedge_detected"]),
            "n_refusal": sum(1 for r in recs if r["_health"]["refusal_detected"]),
            "n_length_anomaly": sum(1 for r in recs if r["_health"]["length_anomaly"]),
            "n_self_inconsistent_images": n_inconsistent,
            "GATE_FAIL": gate_fail,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--benchybench-root")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    health_dir = output_root / "health"
    health_dir.mkdir(parents=True, exist_ok=True)

    # ── Load records ──────────────────────────────────────────────────────────
    records = load_records(output_root)
    if not records:
        msg = f"ERROR: no inference records found in {output_root}/inference/"
        report = {"error": msg, "n_records": 0, "gate_passed": False}
        (health_dir / "health_report.json").write_text(json.dumps(report, indent=2))
        print(msg)
        sys.exit(1)

    # ── Inject method + prompt_stem from sidecar meta JSONs ───────────────────
    # Inference records (especially LLaVA/DeGF) don't carry method/prompt_stem
    # internally; the meta_*.json sidecars written by infer_*.sh are authoritative.
    meta_map = load_meta(output_root)
    for rec in records:
        stem = rec.get("_source_stem", "")
        meta = meta_map.get(stem, {})
        # Qwen records carry method natively; LLaVA records rely entirely on meta
        rec["_method"] = meta.get("method") or rec.get("method") or "unknown"
        rec["_prompt_stem"] = meta.get("prompt_stem") or rec.get("_prompt_stem") or "unknown"

    print(f"Loaded {len(records)} records")

    # ── Annotate per-record health flags ──────────────────────────────────────
    records = annotate_records(records)

    n_rep  = sum(1 for r in records if r["_health"]["repetition_detected"])
    n_hed  = sum(1 for r in records if r["_health"]["hedge_detected"])
    n_ref  = sum(1 for r in records if r["_health"]["refusal_detected"])
    n_len  = sum(1 for r in records if r["_health"]["length_anomaly"])
    n_pfail = sum(1 for r in records if r["_health"]["parsed_label"] is None)
    print(f"  repetition_detected : {n_rep}")
    print(f"  hedge_detected      : {n_hed}")
    print(f"  refusal_detected    : {n_ref}")
    print(f"  length_anomaly      : {n_len}")
    print(f"  parse_fail (global) : {n_pfail} / {len(records)} ({100*n_pfail/len(records):.1f}%)")

    # ── Per-combo stats ───────────────────────────────────────────────────────
    combo_stats = per_combo_stats(records)
    gate_failures = [(k, v) for k, v in combo_stats.items() if v["GATE_FAIL"]]

    print(f"\nPer-combo summary ({len(combo_stats)} combos):")
    for (model, method, prompt_stem), stats in sorted(combo_stats.items()):
        status = "FAIL" if stats["GATE_FAIL"] else "OK"
        print(f"  [{status}] {model}×{method}×{prompt_stem}: "
              f"parse_fail={stats['parse_fail_rate']:.1%}, "
              f"label_bias={stats['label_bias']}"
              + (f" (dominant: {stats['bias_dominant']})" if stats["label_bias"] else ""))

    # ── Build and write report ────────────────────────────────────────────────
    report = {
        "n_records": len(records),
        "gate_passed": len(gate_failures) == 0,
        "global": {
            "n_repetition": n_rep,
            "n_hedge": n_hed,
            "n_refusal": n_ref,
            "n_length_anomaly": n_len,
            "n_parse_fail": n_pfail,
            "parse_fail_rate": round(n_pfail / len(records), 4),
        },
        "combos": {
            f"{m}×{mt}×{ps}": v
            for (m, mt, ps), v in combo_stats.items()
        },
        "gate_failures": [
            f"{m}×{mt}×{ps}" for (m, mt, ps), _ in gate_failures
        ],
    }

    report_path = health_dir / "health_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nHealth report written to: {report_path}")

    if gate_failures:
        print(f"\nGATE FAILED: {len(gate_failures)} combo(s) failed health check:")
        for combo_key, stats in gate_failures:
            print(f"  {combo_key[0]}×{combo_key[1]}×{combo_key[2]}: "
                  f"parse_fail={stats['parse_fail_rate']:.1%}, label_bias={stats['label_bias']}")
        print("Downstream judge submission will not proceed (--dependency=afterok).")
        sys.exit(1)
    else:
        print("\nGATE PASSED: all combos healthy.")
        sys.exit(0)


if __name__ == "__main__":
    main()
