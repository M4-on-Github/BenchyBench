# Path Sweep Findings

Audit of path assumptions across all four submodules, checking for drift
introduced when the standalone repos were wrapped by BenchyBench.

## Status

| # | Finding | Status |
|---|---------|--------|
| 1 | P8+ `config.yaml` points at a directory that does not exist | **open** |
| 2 | "run from `~/<Repo>/`" instructions assume the pre-BenchyBench layout | **open** — docs |
| 3 | QWEN-Maritime docs say `~/QWEN`; the repo is `QWEN-Maritime` | **open** — docs |
| 4 | Cross-repo container binds assume siblings at `~/` | **partly fixed** — `DEGF_REPO` corrected in code; comments still stale |
| 5 | `SLURM_SUBMIT_DIR` trusted without validation | **fixed** |
| 6 | Stale layout descriptions in READMEs and SPECs | **open** — docs |

Finding 5 is resolved by `CASTOR/benchybench_paths.sh`, deployed identically to
DeGF, ONLY and QWEN-Maritime, and covered by `tests/test_paths.sh` (28 assertions
including a simulated SLURM spool execution). The library also removes the
config.json cwd-relative fallback by passing `--image-folder` explicitly.

The layout question below is now settled as policy: **nested is canonical**,
with standalone supported via `BENCHYBENCH_ROOT` or `--image-folder`.

Scope note: `DeGF/experiments/lavis/`, `ONLY/experiments/llava/`, and
`ONLY/transformers/` are vendored upstream code. Their `/export/home/...` and
`/home/kchen/...` paths are third-party and out of scope.

---

## The question everything depends on

Every finding below hinges on one fact this audit cannot determine from the
filesystem: **on the cluster, do the method repos live inside BenchyBench?**

| Layout | Meaning |
|---|---|
| **A — nested** | `~/BenchyBench/DeGF`, `~/BenchyBench/ONLY`, … |
| **B — siblings** | `~/DeGF`, `~/ONLY`, … with BenchyBench separate |

The code implies **A**: every `submit_job.sh` derives the images directory as
`$(dirname $REPO)/shipwreck_wiki_images`, which only resolves correctly when the
repo sits one level below the directory holding the images.

The docs consistently say **B**: "run from `~/DeGF/`", "run from `~/ONLY/`",
"run from `~/Eval_CASTOR/`".

Under A, the doc instructions are wrong. Under B, the images path breaks unless
a second copy exists. They cannot both be right.

---

## Findings

### 1. P8+ `config.yaml` points at a directory that does not exist — **broken**

`Eval_CASTOR/pipelines/plan_coherence/improved/config.yaml`

```yaml
line 13  images_dir:   /home/${USER}/ONLY/CASTOR/shipwreck_wiki_images/sorted_images
line 21  pipeline_dir: /home/${USER}/Eval_CASTOR/pipelines/plan_coherence/improved
line 24  gt_csv:       /home/${USER}/Eval_CASTOR/human_ground_truth_label/human_gt.csv
```

`images_dir` is wrong in two independent ways: it assumes ONLY sits at `~/ONLY`,
and that images live inside `ONLY/CASTOR/`. **No method repo contains images** —
verified: all 112 are tracked once, at the BenchyBench root. This path cannot
resolve under either layout.

Mitigating: line 3 says "Edit the paths section before running run_all.sh", so
it is a template. But the defaults encode a layout that no longer exists, and
this is a ⭐ pipeline.

### 2. Every "run from `~/<Repo>/`" instruction assumes layout B

| Repo | Locations |
|---|---|
| DeGF | `CASTOR/submit.sh:6`, `submit_job.sh:8,19,21,34` |
| ONLY | `CASTOR/submit.sh:6`, `submit_job.sh:8,19,21,34` |
| QWEN-Maritime | `degf_ablate/` and `self_verify/` submit scripts + `generate_assets_sd.py` |
| Eval_CASTOR | 5 scripts under `containers/` |

If the cluster uses layout A, every one of these is a broken copy-paste.

### 3. QWEN-Maritime docs say `~/QWEN`, but the repo is `QWEN-Maritime`

`CASTOR/degf_ablate/submit.sh:6`, `submit_degf_ablate.sh:8,22`,
`self_verify/submit.sh:7`, `submit_self_verify.sh:8,21`,
`degf_ablate/run_degf_ablate.py:18`, `generate_assets_sd.py:5`

Same class as the `benchybench` / `BenchyBench` mismatch already fixed in the
parent repo: a case- and name-sensitive path that silently fails on Linux.

### 4. Cross-repo container binds assume sibling repos at `~/`

QWEN-Maritime binds ONLY and DeGF into its containers:

```
generate_assets_sd.py:9        --bind ~/DeGF:~/DeGF --bind ~/ONLY:~/ONLY
submit_degf_ablate.sh:25       --bind ~/ONLY:~/ONLY
submit_self_verify.sh:24       --bind ~/ONLY:~/ONLY
```

Under layout A these resolve to nonexistent paths, and the bind silently
provides nothing rather than erroring loudly.

### 5. `SLURM_SUBMIT_DIR` override — latent, all three inference repos

```bash
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
...
IMAGE_DIR="$(dirname $REPO)/shipwreck_wiki_images/sorted_images"
```

`DeGF/CASTOR/submit_job.sh:36,91` · `ONLY/CASTOR/submit_job.sh:35,92` ·
`QWEN-Maritime/CASTOR/submit_job.sh:37,72`

The fallback is correct — it derives the repo root from the script's own
location. But `SLURM_SUBMIT_DIR` is whatever directory `sbatch` was invoked
from, and it silently wins. Submit from the BenchyBench root instead of the repo
root and `REPO` is one level too high, so `$(dirname $REPO)` climbs above it and
images resolve to `~/shipwreck_wiki_images`.

Images are located by **two independent routes** that happen to agree:

1. `prepare_dataset.py` receives `--image-dir "$IMAGE_DIR"`
2. `run_inference.py` gets **no** `--image-folder`, so it falls back to
   `config.json`'s `"../shipwreck_wiki_images/sorted_images"`, resolved against
   `cwd` — which line 37 set to `$REPO`

`run_inference.py:222` applies only `expandvars`/`expanduser`, never `abspath`
or a join against the config file's own location. Both routes break together,
so the failure is at least consistent rather than silently mismatched.

`visual_classification/` is unaffected: it exports `BENCHYBENCH_ROOT` and calls
`run_inference.py` directly rather than going through these scripts.

### 6. Stale layout descriptions in docs

- `DeGF/CASTOR/README.md:113` — `--image-folder CASTOR/shipwreck_wiki_images/subset`
- `DeGF/CASTOR/README.md:13` — images "travel with the repo"
- `ONLY/SPEC.md:13,26,27,69,127,128` — describes
  `CASTOR/shipwreck_wiki_images/`, and line 69 instructs symlinking from
  `../DeGF/CASTOR/shipwreck_wiki_images/`

All describe the pre-BenchyBench layout where each repo carried its own copy.

---

## Suggested fix order

Once the layout question is answered:

| Priority | Fix | Risk |
|---|---|---|
| 1 | P8+ `config.yaml` paths | Low — config only |
| 2 | `~/QWEN` → `QWEN-Maritime` | Low — comments |
| 3 | `~/<Repo>` doc paths across all four | Low — comments |
| 4 | Cross-repo binds in QWEN | Medium — touches container invocation |
| 5 | `SLURM_SUBMIT_DIR` override | Medium — changes runtime behavior; needs a cluster test |

Items 1–3 are text-only and safe. Item 5 changes how a running job resolves
paths and should not be committed without a cluster smoke test.
