# Our_Pipeline

Fraud detection with a graph-enhanced LLM, built on DGP's dual-granularity prompting ("DGP: A Dual-Granularity
Prompting Framework for Fraud Detection with Graph-Enhanced LLMs", AAAI 2026). No official DGP code exists, so the DGP
recipe here is rebuilt from the paper's description; details the paper does not state are marked in `configs/base.yaml`.

Our pipeline changes three DGP settings, each one line in `configs/ours/`, so each can be switched off:

| Setting | DGP recipe (`configs/datasets/`) | Our pipeline (`configs/ours/`) |
|---|---|---|
| Metapath hops | K = 2 | K = 1, with K = 2 kept as a comparison |
| Summary writer and classifier | Qwen3-8B | Gemma-2-9B |
| Epoch selection | Lowest validation loss, patience 2 | Best validation AUROC, patience 3 |

## The method

For each target node:

1. **Trim.** List every metapath up to K hops, then keep the M neighbours closest to the target under a Markov diffusion
   kernel over DeBERTaV3 and numeric features.
2. **Summarise.** A frozen LLM (Qwen3-8B in the DGP recipe, Gemma-2-9B in ours) summarises each kept neighbour, then each metapath's neighbours, within a token budget.
3. **Prompt.** The target keeps its raw text. Each metapath adds one line with its summary and neighbour means.
4. **Classify.** The same LLM with LoRA reads the Yes and No logits at the first generated token.

DGP recipe: K = 2, M = 4, B = 10, AdamW, batch size 4, up to 10 epochs, lowest validation loss, patience 2, seeds 0-4.

## Layout

```
src/our_pipeline/
  config.py            YAML configs with inheritance, and the run hash
  pipeline.py          runs the stages in order, with caching: trim, summarise, prompt
  runs.py              one seed: train, evaluate, write the manifest
  evaluate.py          threshold, metrics, paired bootstrap, Holm
  graph/               data.py graphs and splits, raw.py build AmazonVideo and YelpChi, trimming.py metapaths,
                       diffusion, top-M and DeBERTaV3 embeddings
  text/                summarise.py node and metapath summaries, prompt.py the final prompt, cache.py resumable
                       stores, tokenizer.py, templates/ the three prompt templates
  model/               classifier.py LoRA classifier and first-token readout, train.py loop and stopping rules
configs/
  base.yaml            the DGP recipe, each value tagged PAPER, GRID or RECON
  datasets/            one file per dataset; every dataset difference lives here
  ours/                our pipeline: the three changes as overlays, one config per dataset and K
scripts/               prepare_data, generate_summaries, train, evaluate, run_queue
tests/                 unit tests on a tiny synthetic graph; no GPU needed
docs/                  RESULTS.md our results, DGP_BASELINE.md the DGP recipe, results.json the comparison statistics
```

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                                        # CPU, under a minute

python scripts/prepare_data.py --config configs/datasets/amazonvideo.yaml
python scripts/run_queue.py configs/ours/amazonvideo_k1.yaml --gpus 0 1 2 3
python scripts/evaluate.py configs/ours/amazonvideo_k1.yaml
```

Models are downloaded from the Hugging Face Hub on first use, at the revisions pinned in the configs:
`Qwen/Qwen3-8B`, `microsoft/deberta-v3-base`, and `google/gemma-2-9b-it` for our pipeline (a gated model: accept its
licence on the Hub and log in with `hf auth login`).

Data is not included. `configs/datasets/` gives each source. Reddit, Instagram and Amazon Instruments must not be
re-hosted; they come from public processed graphs, placed as `data/processed/<name>/` together with their `split.json`. YelpChi needs the 67,395-review release with text, not `YelpChi.mat`.

Every run folder holds `metrics.json`, `predictions.csv`, `training_log.jsonl` and `run_manifest.json` (config, git
commit, seed, GPU, runtime, peak memory, mean and max prompt tokens). Epoch and threshold use validation data only, and
`tests/test_train_evaluate.py` checks that changing test data moves neither.

## Run everything

```bash
python scripts/run_queue.py configs/ours/[a-z]*_k?.yaml --gpus 0 1 2 3 4 5 6 7     # our pipeline, both K, five datasets
python scripts/run_queue.py configs/datasets/*.yaml --gpus 0 1 2 3          # the DGP recipe
```

## Results

Test set, mean ± population std over seeds 0-4, Macro-F1 / AUROC / AUPRC. Epoch and threshold are chosen on validation
only. `docs/RESULTS.md` has every arm, paired bootstrap CIs and Holm-corrected tests.

| Dataset | Ours K = 1 | Ours K = 2 | DGP reproduced | Reference |
|---|---|---|---|---|
| AmazonVideo | 65.05 ± 1.45 / 76.56 ± 0.25 / 34.01 ± 0.77 | 64.86 ± 0.83 / 76.27 ± 0.37 / 34.31 ± 1.00 | 65.16 ± 0.52 / 75.07 ± 0.50 / 32.33 ± 0.65 | DGP paper 66.91 / 77.32 / 34.63 |
| YelpChi | 71.40 ± 0.64 / 85.30 ± 0.63 / 56.57 ± 1.49 | 71.44 ± 0.19 / 85.55 ± 0.35 / 55.98 ± 0.99 | 61.76 ± 1.86 / 78.38 ± 0.88 / 31.05 ± 1.31 | DGP paper 69.07 / 84.28 / 48.87 |
| Instagram | 53.86 ± 0.87 / 58.86 ± 1.19 / 14.18 ± 1.26 | 52.66 ± 2.59 / 60.20 ± 1.76 / 14.90 ± 1.48 | 52.75 ± 2.10 / 58.27 ± 2.64 / 12.46 ± 1.08 | FLAG 53.41 / 59.01 / n/a |
| Reddit | 53.58 ± 3.20 / 65.37 ± 1.89 / 15.17 ± 1.23 | 53.97 ± 1.08 / 65.01 ± 1.00 / 14.68 ± 0.68 | 53.61 ± 1.01 / 62.26 ± 1.52 / 12.92 ± 0.81 | FLAG 53.66 / 62.25 / n/a |
| Amazon Instruments | 88.85 ± 1.01 / 97.67 ± 0.19 / 86.65 ± 1.09 | 88.54 ± 1.30 / 97.74 ± 0.14 / 85.68 ± 1.42 | 86.25 ± 1.01 / 97.02 ± 0.32 / 80.27 ± 1.74 | FLAG 76.45 / 89.95 / n/a |

FLAG numbers are means across 7 GNN backbones, so they are context, not a head-to-head result; `docs/RESULTS.md` also
gives the plain GNN means. The DGP reproduced runs are stored results; `docs/DGP_BASELINE.md` gives their provenance and
the parity checks against this code. The stored run manifests predate the rename of the config section `dgp` to `method`;
the values are otherwise the same, and run folders are named by the current config hash.

## Limitations

1. **The DGP recipe falls short of the paper** on both of its public datasets, by 5.90 AUROC on YelpChi and 2.25 on
   AmazonVideo. Most of the YelpChi gap goes with the stopping rule, which is itself reconstructed: under the DGP rule our
   K = 2 arm drops from 85.55 to 82.34 ± 4.00 AUROC.
2. **Three changes at once.** Gemma is both writer and classifier, so the model effect cannot be split from the other two
   changes, or the writer effect from the classifier effect.
3. **Weak statistics on some gains.** Instagram and Reddit gains do not survive Holm correction, and five seeds give
   little power.
4. **Settings are inherited, not validated.** M, B, LoRA rank and learning rate were never tuned; the paper's grid was
   not run.
5. **Reconstructed details.** The prompt layout, patience, LoRA alpha and others are not stated in the paper.
6. **Extension datasets are new ground.** Reddit, Instagram and Amazon Instruments are not in the paper and have no numeric
   summary. Reddit and Instagram have user nodes and one relation each.
7. **Amazon Instruments text is cut.** 565 of 8,639 text nodes keep only their first 3,900 tokens, and text alone is a
   strong signal there (see `docs/DGP_BASELINE.md`).
8. **The FLAG comparison is indirect.** It compares against GNN means, and Amazon uses a collapsed graph there.
9. **YelpChi comes from an unofficial mirror** (67,395 reviews), and differs from the YelpChi used in the FLAG runs (45,954 nodes).
10. **One fixed split per dataset**, so standard deviations reflect training seeds only.
11. **Mixed software stacks.** The stored DGP runs and our runs used different torch and CUDA builds.

## Future study

1. Separate the writer and the classifier: Qwen writer with Gemma classifier, and the reverse.
2. The paper's grid.
3. User-node metapath designs for Reddit and Instagram.
4. GNN baselines on the same splits for AmazonVideo and YelpChi.

## Licence

Code: MIT. Gemma, Qwen3-8B, DeBERTaV3 and the datasets keep their own terms.
