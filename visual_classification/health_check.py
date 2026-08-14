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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

# NOTE ON PYTHON VERSION
# This module runs inside castor.sif (Python 3.10), so dataclasses and typing
# generics are available. judge_submit.py is NOT in the container — it runs on
# the node's bare python3, which is 3.6 — so it must not import dataclasses.
# Keep that split in mind before moving code between the two.


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


class StateClassifier:
    """Extracts casualty states from free-form model output.

    Owns the synonym map and its compiled patterns, which were previously
    module-level globals shared by several free functions. Encapsulating them
    means an alternative synonym set can be supplied — useful for testing a
    prompt that elicits different vocabulary — without mutating global state
    that every other caller also reads.

    Patterns are compiled once per instance and match on word boundaries, so
    'fire' does not match inside 'misfired'.
    """

    def __init__(self, state_map: Optional[Dict[str, List[str]]] = None):
        self.state_map = state_map if state_map is not None else STATE_MAP
        self._patterns = {
            cls: re.compile(
                r"\b(" + "|".join(re.escape(s) for s in synonyms) + r")\b",
                re.IGNORECASE,
            )
            for cls, synonyms in self.state_map.items()
        }

    def states_mentioned(self, text: str) -> set:
        """Return every distinct state named in `text`.

        Synonyms collapse to their canonical state, so "the sunk vessel is
        submerged" yields {'sunken'} rather than two entries.
        """
        if not text:
            return set()
        return {cls for cls, pat in self._patterns.items() if pat.search(text)}

    def normalize(self, text: str) -> Optional[str]:
        """Return the first state named in `text`, or None.

        First match wins in state_map order, so hedged output naming several
        states still yields a single label. That is why is_hedged() exists:
        hedging does not show up as a parse failure and would otherwise
        inflate apparent parse success.
        """
        if not text:
            return None
        for cls, pat in self._patterns.items():
            if pat.search(text):
                return cls
        return None

    def is_hedged(self, text: str) -> bool:
        """True if `text` names more than one distinct state."""
        return len(self.states_mentioned(text)) > 1


class RefusalDetector:
    """Detects output that declines to answer.

    Deliberately conservative keyword matching: it catches explicit refusals
    ("cannot determine", "no image") but never a confidently stated wrong
    answer, which is indistinguishable from a correct one without ground
    truth and is left to the evaluation pipelines.
    """

    DEFAULT_KEYWORDS = (
        "cannot", "can't", "unable to determine", "unclear", "not sure",
        "impossible to tell", "no image", "no photo", "no picture",
    )

    def __init__(self, keywords: Optional[Tuple[str, ...]] = None):
        self.keywords = keywords if keywords is not None else self.DEFAULT_KEYWORDS

    def detect(self, text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        return any(k in lower for k in self.keywords)


class RepetitionDetector:
    """Detects degenerate decoding loops via repeated n-grams.

    A model stuck in a loop emits the same phrase until it hits the token
    limit. Thresholds are instance state rather than call-site defaults so a
    run can be scanned consistently without every caller repeating them.
    """

    def __init__(self, ngram: int = 5, threshold: int = 3):
        self.ngram = ngram
        self.threshold = threshold

    def detect(self, text: str) -> bool:
        """True if any `ngram`-word sequence occurs at least `threshold` times."""
        if not text:
            return False
        words = text.lower().split()
        # Too short to contain the required number of repeats — skip the scan.
        if len(words) < self.ngram * self.threshold:
            return False
        seen: Dict[str, int] = {}
        for i in range(len(words) - self.ngram + 1):
            gram = " ".join(words[i:i + self.ngram])
            seen[gram] = seen.get(gram, 0) + 1
            if seen[gram] >= self.threshold:
                return True
        return False


# Module-level instances backing the free functions below, which are kept as a
# compatibility facade for existing callers and tests.
_CLASSIFIER = StateClassifier()
_REFUSAL = RefusalDetector()
_REPETITION = RepetitionDetector()


# ── Compatibility facade ─────────────────────────────────────────────────────
# These delegate to the module-level instances above. They exist so callers and
# tests written against the original procedural API keep working; new code
# should construct the classes directly, which allows non-default configuration.

def normalize_state(text):
    """Return the first casualty state named in `text`, or None.

    Delegates to StateClassifier.normalize(). See that method for why a hedged
    answer still yields a single label.
    """
    return _CLASSIFIER.normalize(text)


def detect_repetition(text, ngram=5, threshold=3):
    """True if any `ngram`-word sequence repeats at least `threshold` times.

    Delegates to RepetitionDetector. A non-default ngram or threshold builds a
    throwaway detector, since the shared instance carries the defaults.
    """
    if ngram == _REPETITION.ngram and threshold == _REPETITION.threshold:
        return _REPETITION.detect(text)
    return RepetitionDetector(ngram=ngram, threshold=threshold).detect(text)


def detect_hedge(text):
    """True if `text` names more than one casualty state.

    Delegates to StateClassifier.is_hedged(). Hedged records still parse, so
    this is the only signal that separates a committed answer from an
    equivocal one.
    """
    return _CLASSIFIER.is_hedged(text)


def detect_refusal(text):
    """True if `text` declines to answer. Delegates to RefusalDetector."""
    return _REFUSAL.detect(text)


def load_records(output_root):
    """Load every inference record from output_root/inference/answers_*.jsonl.

    Each record gains a `_source_stem` field naming the file it came from,
    which downstream grouping uses to recover the (model, method, prompt)
    combination.

    Malformed JSON lines are skipped silently: a partial write from an
    interrupted job should not block the health check that exists to report it.
    Line counts therefore reflect parseable records, not bytes written.
    """
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



@dataclass
class HealthFlags:
    """Per-record health signals produced by RecordAnnotator.

    Separating these from the record dict gives the five signals a name and a
    type. They are not independent: `hedge_detected` can be True while
    `parsed_label` is non-None, because hedged output still parses. Reading
    `parsed_label` alone therefore overstates how many records gave a committed
    answer.
    """

    repetition_detected: bool = False
    hedge_detected: bool = False
    refusal_detected: bool = False
    length_anomaly: bool = False
    parsed_label: Optional[str] = None

    @property
    def parse_failed(self) -> bool:
        """True if no state could be extracted."""
        return self.parsed_label is None

    def to_dict(self) -> dict:
        """Serialise to the `_health` dict shape that downstream code reads."""
        return asdict(self)


class RecordAnnotator:
    """Applies the health detectors across a full set of records.

    Exists as a class because `length_anomaly` cannot be judged one record at a
    time — it needs the mean and standard deviation of the whole run, which is
    fitted in `fit()` and then applied in `flags_for()`. That two-phase shape is
    exactly what an object models well and a free function does not.

    The baseline is run-wide, not per combination, so a uniformly more verbose
    method shifts the baseline rather than having all of its records flagged.
    The flag marks outliers relative to the run.
    """

    SIGMA = 3  # how many standard deviations above the mean counts as anomalous

    def __init__(self, classifier=None, refusal=None, repetition=None):
        self.classifier = classifier or _CLASSIFIER
        self.refusal = refusal or _REFUSAL
        self.repetition = repetition or _REPETITION
        self._mean = 0.0
        self._std = 0.0

    def fit(self, records: List[dict]) -> "RecordAnnotator":
        """Compute the run-wide length baseline. Returns self for chaining."""
        lengths = [len(r.get("text", "")) for r in records if r.get("text")]
        self._mean = mean(lengths) if lengths else 0
        self._std = stdev(lengths) if len(lengths) > 1 else 0
        return self

    def flags_for(self, text: str) -> HealthFlags:
        """Evaluate every detector against one record's text."""
        text = text or ""
        # With zero variance every record is the same length, so nothing is an
        # outlier — guarding also avoids flagging everything when std is 0.
        anomalous = bool(self._std) and len(text) > self._mean + self.SIGMA * self._std
        return HealthFlags(
            repetition_detected=self.repetition.detect(text),
            hedge_detected=self.classifier.is_hedged(text),
            refusal_detected=self.refusal.detect(text),
            length_anomaly=anomalous,
            parsed_label=self.classifier.normalize(text),
        )

    def annotate(self, records: List[dict]) -> List[dict]:
        """Attach a `_health` dict to every record, in place. Returns `records`."""
        self.fit(records)
        for rec in records:
            rec["_health"] = self.flags_for(rec.get("text", "")).to_dict()
        return records


def annotate_records(records):
    """Attach a `_health` dict to each record, in place. Returns `records`.

    Compatibility facade over RecordAnnotator; see that class for why the
    length baseline is run-wide.
    """
    return RecordAnnotator().annotate(records)


@dataclass
class ComboStats:
    """Aggregate health for one (model, method, prompt_stem) combination.

    Two judgements live here, and they are deliberately NOT equivalent:

    `label_bias` is a WARNING. A model predicting one class for over 40% of
    parsed records may be broken — or may be right. The CASTOR image set is
    unbalanced and 'aground' genuinely dominates, so treating this as an error
    would fail a run for a property of the dataset. It is reported, never gated.

    `gate_fail` is a HARD FAILURE, triggering only on parse failure above 15%.
    That is not a quality bar: accuracy cannot be computed from output no
    parser can read, so continuing would compute metrics over a silently
    shrinking denominator.

    Both are suppressed below MIN_SAMPLE_N, since neither a distribution nor a
    rate means anything on smoke-size samples.
    """

    MIN_SAMPLE_N = 10       # below this, neither check is meaningful
    BIAS_THRESHOLD = 0.40   # share of one class that counts as biased
    PARSE_FAIL_LIMIT = 0.15 # parse-failure rate that gates the run

    model: str
    method: str
    prompt_stem: str
    n_records: int = 0
    n_parse_fail: int = 0
    label_distribution: Dict[str, int] = field(default_factory=dict)
    n_repetition: int = 0
    n_hedge: int = 0
    n_refusal: int = 0
    n_length_anomaly: int = 0
    n_self_inconsistent_images: int = 0

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.model, self.method, self.prompt_stem)

    @property
    def parse_fail_rate(self) -> float:
        return self.n_parse_fail / self.n_records if self.n_records else 0.0

    @property
    def n_parsed(self) -> int:
        return self.n_records - self.n_parse_fail

    @property
    def bias_dominant(self) -> Optional[str]:
        """The most-predicted label, or None when nothing parsed."""
        if not self.label_distribution:
            return None
        return max(self.label_distribution, key=self.label_distribution.get)

    @property
    def label_bias(self) -> bool:
        """True if one class dominates the parsed predictions. Warning only."""
        if self.n_parsed < self.MIN_SAMPLE_N:
            return False
        return any(c / self.n_parsed > self.BIAS_THRESHOLD
                   for c in self.label_distribution.values())

    @property
    def gate_fail(self) -> bool:
        """True if parse failure is high enough to invalidate the run."""
        if self.n_records < self.MIN_SAMPLE_N:
            return False
        return self.parse_fail_rate > self.PARSE_FAIL_LIMIT

    def to_dict(self) -> dict:
        """Serialise to the health_report.json shape.

        `GATE_FAIL` is upper-case purely for backward compatibility — it is the
        key already written to health_report.json and read by downstream
        tooling. The attribute is `gate_fail`, matching every other name here.
        """
        return {
            "n_records": self.n_records,
            "parse_fail_rate": round(self.parse_fail_rate, 4),
            "n_parse_fail": self.n_parse_fail,
            "label_distribution": dict(self.label_distribution),
            "label_bias": self.label_bias,
            "bias_dominant": self.bias_dominant,
            "n_repetition": self.n_repetition,
            "n_hedge": self.n_hedge,
            "n_refusal": self.n_refusal,
            "n_length_anomaly": self.n_length_anomaly,
            "n_self_inconsistent_images": self.n_self_inconsistent_images,
            "GATE_FAIL": self.gate_fail,
        }


class ComboAggregator:
    """Groups annotated records into ComboStats, one per combination."""

    @staticmethod
    def combo_key(rec: dict) -> Tuple[str, str, str]:
        """Identify which combination a record belongs to.

        `model_tag` is preferred over `model_id`: the former is the short label
        the harness assigns, the latter the model's own name, which varies by
        checkpoint and would split one combination into several.
        """
        model = rec.get("model_tag") or rec.get("model_id", "unknown")
        return (model,
                rec.get("_method", "unknown"),
                rec.get("_prompt_stem", "unknown"))

    def aggregate(self, records: List[dict]) -> Dict[Tuple[str, str, str], ComboStats]:
        grouped = defaultdict(list)
        for rec in records:
            grouped[self.combo_key(rec)].append(rec)

        out = {}
        for (model, method, prompt_stem), recs in grouped.items():
            labels = defaultdict(int)
            # Distinct labels seen per image; more than one means the combination
            # contradicted itself on the same input.
            per_image = defaultdict(set)
            stats = ComboStats(model=model, method=method, prompt_stem=prompt_stem,
                               n_records=len(recs))

            for r in recs:
                health = r["_health"]
                label = health["parsed_label"]
                if label:
                    labels[label] += 1
                    per_image[r.get("image", "")].add(label)
                else:
                    stats.n_parse_fail += 1
                stats.n_repetition     += bool(health["repetition_detected"])
                stats.n_hedge          += bool(health["hedge_detected"])
                stats.n_refusal        += bool(health["refusal_detected"])
                stats.n_length_anomaly += bool(health["length_anomaly"])

            stats.label_distribution = dict(labels)
            stats.n_self_inconsistent_images = sum(
                1 for lbls in per_image.values() if len(lbls) > 1)
            out[stats.key] = stats
        return out


def per_combo_stats(records):
    """Aggregate annotated records by (model, method, prompt_stem).

    Compatibility facade over ComboAggregator, returning plain dicts. See
    ComboStats for why label_bias only warns while parse failure gates.
    """
    return {k: v.to_dict() for k, v in ComboAggregator().aggregate(records).items()}
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
        print(f"\nHEALTH WARNINGS: {len(gate_failures)} combo(s) have elevated parse_fail or label_bias:")
        for combo_key, stats in gate_failures:
            print(f"  {combo_key[0]}×{combo_key[1]}×{combo_key[2]}: "
                  f"parse_fail={stats['parse_fail_rate']:.1%}, label_bias={stats['label_bias']}")
        print("Proceeding — health check is informational; see health_report.json for details.")
    else:
        print("\nAll combos healthy.")
    sys.exit(0)


if __name__ == "__main__":
    main()
