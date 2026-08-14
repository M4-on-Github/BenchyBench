# BenchyBench

ONR maritime disaster VLM research — hallucination mitigation methods applied
to the CASTOR shipwreck classification and salvage planning task.

This is a **meta-repository**. The research code lives in four independent
submodules, each with its own pipelines and documentation. This repo pins them
together, holds the resources they share (prompts, dataset), and provides one
integrated benchmark harness.

**Looking for a specific pipeline?** See [PIPELINES.md](PIPELINES.md) for a
directory of what exists in which repo.

## Repositories

| Repo | Paper | Method | Branch |
|------|-------|--------|--------|
| [DeGF](https://github.com/M4-on-Github/DeGF) | ICLR 2025 | JS-divergence contrastive decoding via Stable Diffusion reference images | `interactive_session` |
| [ONLY](https://github.com/M4-on-Github/ONLY) | ICCV 2025 | Single transformer-layer suppression (no diffusion) | `ONLY_CASTOR` |
| [Eval_CASTOR](https://github.com/M4-on-Github/Eval_CASTOR) | — | Eight evaluation pipelines (P1–P8) over CASTOR inference output | `main` |
| [QWEN-Maritime](https://github.com/M4-on-Github/QWEN-Maritime) | — | Qwen3-VL-8B baseline + self-verification pipeline | `main` |

All four repos apply their methods to the same task: classify shipwreck images
(`aground / capsized / on_fire / sunken`) using the CASTOR dataset on the
AART Lab `pleiades` SLURM cluster.

## Clone Everything

```bash
git clone --recurse-submodules https://github.com/M4-on-Github/BenchyBench.git
cd BenchyBench
```

This gives you all four repos at the commits this master repo currently pins.

## Pull Latest from All Branches

To update all submodules to the latest commit on their tracked branch:

```bash
git submodule update --remote --merge
```

Then commit the updated pointers in the parent:

```bash
git add DeGF ONLY Eval_CASTOR QWEN-Maritime
git commit -m "Update submodule pointers to latest"
git push
```

## If You Already Have the Repos Cloned

```bash
git clone https://github.com/M4-on-Github/BenchyBench.git
cd BenchyBench
git submodule update --init --recursive
```

Or point the submodules at your existing local clones:

```bash
git clone https://github.com/M4-on-Github/BenchyBench.git
cd BenchyBench
git config submodule.DeGF.url /path/to/your/DeGF
git config submodule.ONLY.url /path/to/your/ONLY
git config submodule.Eval_CASTOR.url /path/to/your/Eval_CASTOR
git config submodule.QWEN-Maritime.url /path/to/your/QWEN-Maritime
git submodule update --init
```

## CASTOR Task

Images: `shipwreck_wiki_images/sorted_images/` **in this repo** (not inside the
method repos), sorted into `aground / capsized / on_fire / sunken`, ~110 images
total. Method repos read them from here. See
[shipwreck_wiki_images/README.md](shipwreck_wiki_images/README.md).

Ground truth: `Eval_CASTOR/human_ground_truth_label/human_gt.csv`, joined on the
`image` column (e.g. `aground/00017.jpg`).

Cluster: AART Lab `pleiades` (`head1.condo.cs.cmu.edu`), RTX6000Ada GPUs.
Inference and the vLLM-backed evaluation pipelines are cluster-only.

## Repo Contents

| Path | What it is |
|------|-----------|
| `PIPELINES.md` | Index of every pipeline across all four repos |
| `METHOD_NOTES.md` | How DeGF and ONLY decoding actually work, read from the code |
| `MODELS.md` | Every model used, where it lives, and how it is obtained |
| `tests/` | Local test suites — no cluster or GPU required |
| `visual_classification/` | Integrated benchmark harness (see below) |
| `all_maritime_prompts/` | Shared prompt library |
| `shipwreck_wiki_images/` | CASTOR dataset (~110 images) |

## Tests

```bash
bash tests/run_all.sh
```

565 assertions across 22 suites, all running locally — no cluster, no GPU, no
model weights, no network. Covers path resolution, the `visual_classification`
pipeline, and every Eval_CASTOR pipeline's core logic. See
[PIPELINES.md](PIPELINES.md#tests) for what each suite covers and what none of
them can.

The method implementations in `DeGF/degf_utils/` and `ONLY/only_utils/` are
kept **byte-identical to the published papers' code** and are not restructured,
so results stay directly comparable with the upstream repositories. What was
learned from reading them is recorded in [METHOD_NOTES.md](METHOD_NOTES.md)
rather than by editing the files.

## Integrated Harness

`visual_classification/` runs every (model × method × prompt) combination over
the CASTOR images in one submission and produces a unified report — regex
metrics, LLM-judge panel scores, and outcome tier analysis.

It is **one** integration, not the entry point to the project. Most pipelines
live in the submodules and are run directly from there. See
[visual_classification/README.md](visual_classification/README.md) for when to
use it, and [PIPELINES.md](PIPELINES.md) for everything else.

## Data Flow

```
DeGF/ or ONLY/ or QWEN-Maritime/   ← inference (.jsonl output)
         ↓
  results/castor_results/          ← handoff point (gitignored)
         ↓
     Eval_CASTOR/                  ← P1–P8 evaluation pipelines
         ↓
  compared against human_ground_truth_label/human_gt.csv
```

Inference output and evaluation results are **not tracked** — they live under
`/data/$USER/` on the cluster and are gitignored locally.

## Shared Prompts

`all_maritime_prompts/` contains prompt variants used across experiments (CoT,
1-shot, separated-field, visual-grounded planning). See
[all_maritime_prompts/README.md](all_maritime_prompts/README.md).

Each experiment copies the prompts it needs into its own directory —
`ONLY/CASTOR/prompts/`, `QWEN-Maritime/CASTOR/prompts/`, or
`visual_classification/prompts/` — where they are auto-discovered by glob and
indexed by SLURM array task ID. Those copies are frozen once a run starts;
changing them invalidates comparisons.
