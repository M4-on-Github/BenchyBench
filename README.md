# benchybench

ONR maritime disaster VLM research — hallucination mitigation methods applied
to the CASTOR shipwreck classification and salvage planning task.

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
git clone --recurse-submodules https://github.com/M4-on-Github/benchybench.git
cd benchybench
```

This gives you all four repos at the commits this master repo currently pins.

## Pull Latest from All Branches

To update all submodules to the latest commit on their tracked branch:

```bash
git submodule update --remote --merge
```

Then commit the updated pointers in the parent:

```bash
git add DeGF ONLY Eval_CASTOR QWEN
git commit -m "Update submodule pointers to latest"
git push
```

## If You Already Have the Repos Cloned

```bash
git clone https://github.com/M4-on-Github/benchybench.git
cd benchybench
git submodule update --init --recursive
```

Or point the submodules at your existing local clones:

```bash
git clone https://github.com/M4-on-Github/benchybench.git
cd benchybench
git config submodule.DeGF.url /path/to/your/DeGF
git config submodule.ONLY.url /path/to/your/ONLY
git config submodule.Eval_CASTOR.url /path/to/your/Eval_CASTOR
git config submodule.QWEN.url /path/to/your/QWEN
git submodule update --init
```

## CASTOR Task

Images: `shipwreck_wiki_images/sorted_images/` inside each method repo
(aground / capsized / on_fire / sunken, ~110 images total).

Cluster: AART Lab `pleiades` (`head1.condo.cs.cmu.edu`), RTX6000Ada GPUs.

## Data Flow

```
DeGF/ or ONLY/ or QWEN/   ← inference (.jsonl output)
         ↓
  results/castor_results/  ← handoff point (gitignored)
         ↓
     Eval_CASTOR/          ← P1–P8 evaluation pipelines
         ↓
  compared against human_ground_truth_label/human_gt.csv
```

## Shared Prompts

`all_prompts/` contains prompt variants used across experiments (CoT, 1-shot,
visual-grounded planning). The active prompt for each method's cluster job
lives in `<method>/CASTOR/prompts/`.
