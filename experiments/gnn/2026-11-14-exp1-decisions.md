# Experiment 1 decisions log: GNN embeddings for neighbour selection

Every assumption and deviation, with who decided and why. "Ours" = the pipeline in `configs/ours/` (K = 1, Gemma-2-9B
writer and classifier, `auroc_es3` epoch rule, seeds 0-4, split seed 0).

## Scope

| ID | Decision | Decided by | Reason |
|---|---|---|---|
| S1 | Datasets: Instagram and Reddit only. Amazon Instruments waits for promising results. | User, 2026-11-14 | Cost: Amazon Instruments is ~22 GPU-h per variant. |
| S2 | GNN: PMP only. BWGNN deferred. If added later, its "layers" setting is the wavelet order C in {1, 2}, since BWGNN has no stacked layers. | User | Scope. |
| S3 | GNN layers k in {1, 2} = PMP `n_layer`. Independent of metapath K, which stays 1. | User | Task spec. |
| S4 | Experiment code, configs, docs and outputs live in `experiments/gnn/`; the core package only gains a generic hook (`method.selection: embedding_cosine`). | User / Claude | One folder per experiment; core stays GNN-agnostic. |
| S5 | Reddit skipped: no Reddit pipeline job is run. Existing Reddit embeddings, diagnostics and config are kept. Scope is Instagram, PMP-L1. Amazon Instruments and BWGNN stay deferred. | User, 2026-11-14 | ~9.4 GPU-h; only 12.1% of Reddit targets have more than 4 candidate neighbours; Reddit PMP-L1's best GNN epoch is 0 (G5). See also G6. |
| S6 | Original DGP Instagram runs: the 5 DGP-Reproduce runs (commit `df393c4`, `repro_instagram_seed*_9dcd4dad8e`) copied to `runs/dgp_reference/`; recipe identical to `configs/datasets/instagram.yaml` (K 2, M 4, Qwen3-8B, LoRA 8, lr 1e-4, patience 2). | User, 2026-11-14 | Made by other code, so no run of this repo's config exists. |
| S7 | Reddit run after all: PMP-L1 and BWGNN-C1, 5 seeds each, same design as Instagram. Reverses S5 and G6. Amazon Instruments still not run. | User, 2026-11-14 ("run BWGNN and PMP on reddit as well, same as instagram") | User decision. |

## Pipeline hook

| ID | Decision | Reason |
|---|---|---|
| H1 | `method.selection` defaults to `diffusion` in code, not in `configs/base.yaml`. | A new key in `base.yaml` would change every config hash and orphan the 50 stored runs. |
| H2 | Trims cache key gains `selection` and the SHA-256 (16 hex) of the embedding file only when selection is not diffusion. | Baseline caches stay valid; verified on the real Instagram and Reddit K=1 configs (same folders `c5a0a54a...`, `14c89288...`). |
| H3 | Cosine on the raw last hidden layer, largest first, ties by node id, zero vectors score 0 (norm clipped at 1e-12). | Task spec. |
| H4 | The node-summary cache key never included the trims folder, so node summaries are already reused across selections; only newly selected nodes are summarised. No change made. | Checked in `pipeline.node_store`. |

## PMP

Source: github.com/Xtra-Computing/PMP, commit `3f7629f6c180891a0bc1bba3c66d94d288a1ddae` (no licence file; reimplemented, not
copied). Paper cited as ICLR 2024, OpenReview `tEgrUrUuwA`, as in the official README.

| ID | Decision | Reason |
|---|---|---|
| D1 | **Feature normalisation: two grid configs on raw features.** The official `row_normalize` divides each row by (row sum + 0.01). DeBERTa row sums are often negative or near zero (Instagram: 239 negative, 28 with \|sum\| < 0.1; Reddit: 7,817 negative, 650 near zero), so normalised values reach 4.3e3 (Instagram) and 1.4e6 (Reddit) against a raw maximum of about 20. The grid keeps the four official-normalisation configs and adds A3 and Y3: A2 and Y2 with `norm_feat: false`. Validation AUROC still chooses. | Approved by the user at Gate 2. z-scoring is out of scope, so "off" means raw features. |
| D2 | Grid (6 configs, chosen on validation AUROC, ties to the lower index): A1 = official Amazon config (hid 256, dropout 0.6, 100 epochs, patience 20) unweighted; A2 = A1 weighted; Y1 = official Yelp config (hid 48, dropout 0, 500 epochs, patience 100) unweighted; Y2 = Y1 weighted; A3, Y3 = D1. | The official Amazon config uses `weighted_loss: false`; the task asked for weighted cross-entropy. Both are tried and validation decides. |
| D3 | Class weight = #train benign / #train fraud, from train labels only. | Official formula, no val/test labels. |
| D4 | Partition labels (`label_unk`): train nodes keep their label; every other node is unknown (2). | Official construction; no val or test label enters message passing. Tested. |
| D5 | Relations: `merge` (official `homo: true`). Instagram and Reddit have one relation, so merge and cat are identical here. | Official Amazon config. |
| D6 | Full batch, one Adam step per epoch. Official: mini-batches with full neighbourhoods. | Simpler; graphs are small. |
| D7 | The official "mean" aggregator sums each partition (no degree division). Reproduced. | Faithfulness. |
| D8 | The official model crashes on 768-dim input (LIMLPs sized for the input width after `fc_neigh` already reduced it) and at 2 layers (`trans_src` sized for the original features but fed the layer input). Here the LIMLP input is the hidden width and `trans_src` always reads the original features. Identical to the official code at layer 1 apart from the crash fix. | Needed to run at all. |
| D9 | Early stopping on validation AUROC every epoch, official `best <= now` rule (ties move later), state kept in memory. | Official `monitor: auc_gnn`. |
| D10 | Seed 0, CPU, deterministic algorithms; one GNN per dataset and variant shared by all 5 classifier seeds. | Task spec; keeps node and metapath summaries shared across seeds. Recorded as a limitation. |
| D11 | Embedding = hidden state before dropout + Conv1d head (`proj: true`). Scores = sigmoid(logit[:, 1]), as in the official code. | Task spec / official code. |

The full list of 12 implementation deviations is in the docstring of `experiments/gnn/gnn_embed.py`.

## Evaluation

| ID | Decision | Reason |
|---|---|---|
| E1 | Primary metric AUROC. Holm across all variant × dataset comparisons against Ours K=1. | Task spec. |
| E2 | `scripts/compare_arms.py` checked against the stored Ours K=1 vs K=2 statistics: exact on four datasets. Amazon Instruments K=1 `predictions.csv` is rounded to 6 decimals, which shifts its AUROC/AUPRC CIs by up to 0.43; `predictions_full.csv` reproduces them exactly. Not relevant until Amazon Instruments is run. | User: no action for now. |

## Phase C and D

| ID | Decision | Decided by | Reason |
|---|---|---|---|
| G1 | Variant: PMP-L1 for both datasets (best GNN validation AUROC: Instagram 62.86 vs 61.21 for L2; Reddit 67.66 vs 63.64). | User, Gate 3 | Task rule: choose by GNN validation AUROC only. |
| G2 | Config Y3 (raw features, weighted loss, Yelp-config widths) was chosen in all four variants. | Validation AUROC | D1 made the difference: best official-normalisation config was 61.67 (Instagram L1) and 58.85 (Reddit L1). |
| G3 | Run Instagram first; decide on Reddit after its results (option a). | User, Gate 3 | Reddit headroom is 12.1% of targets and mean Jaccard 0.51 with diffusion, so only ~6% of prompts change materially, for ~9.4 GPU-h. |
| G4 | Experiment configs `experiments/gnn/configs/<ds>_pmp_l1.yaml` = `configs/ours/<ds>_k1.yaml` + `method.selection: embedding_cosine` + `method.selection_embeddings` + `paths.results: experiments/gnn/runs/results`. Verified: no other key differs. | Claude | One change only. |
| G5 | Caveat: Reddit PMP-L1's best epoch is 0, so its embedding is one optimiser step from initialisation. | Recorded | Limits what a Reddit result could say about trained GNN embeddings. |

## Operations

| ID | Decision | Decided by | Reason |
|---|---|---|---|
| O1 | Dashboard served on the instance at `0.0.0.0:10100` (public `http://80.188.223.202:11469/`), **without authentication**; no file outside the repo was changed. | User, 2026-11-14 | User chose open access over the token-protected Caddy route, which would have needed an edit to `/etc/portal.yaml`. Anyone with the URL can see logs, configs and process lists. |
| O2 | Phase D stopped after 3 identical failures: the Hugging Face model cache `/workspace/.hf_home` is unreadable (`Errno 116 Stale file handle`; only 1.2 GB present, no Gemma weights). Fix pending the user's decision. | Rule: two failures = stop | Not a pipeline fault; repo data and caches are intact. |
| O3 | O2 resolved: Gemma-2-9B-it re-downloaded at the pinned revision `11c9b309` into a new cache `/workspace/.hf_home_exp1` (18 GB, every file read back); the queue ran with `HF_HOME` set to it and `HF_HUB_OFFLINE=1`. The broken `/workspace/.hf_home` was left untouched. The queue resumed from `node_0`; all 9 Instagram jobs finished with rc 0 at 02:45 UTC. | User, 2026-11-14 (option 1) | Fresh cache avoids the stale file handle without deleting or repairing anything. |

## Phase E

| ID | Decision | Decided by | Reason |
|---|---|---|---|
| E3 | Instagram comparison: "Ours + PMP-L1 cosine" against "Ours K=1", Holm over the three metrics of the one dataset run (`--holm-family all`). Primary metric AUROC: Δ −0.29, 95% CI [−1.49, +0.90], Holm 0.89. Macro-F1 Δ −1.96 (Holm 0.033). | Task spec (E1) | With Reddit not run, the family is this dataset's three tests. |
| G6 | Reddit and Amazon Instruments full runs not made; experiment closed after Instagram. | User, 2026-11-14 ("proceed phase E" after the recommendation to skip Reddit) | No AUROC gain on Instagram, the dataset with the most headroom (51.8%); Reddit has 12.1% headroom and a best GNN epoch of 0 (G5), for ~9.4 GPU-h. |
| E4 | `runs/compare/significance.json` holds only the 3 comparisons against Ours K=1 (main Holm family: Macro-F1, AUROC, AUPRC on Instagram), converted from `scripts/compare_arms.py` JSON output without recomputing anything. No significance test against the original DGP; the dashboard shows only the mean difference for that row. | User, 2026-11-14 | User: "do not compare with original DGP, leave it". |

Report: [`2026-11-14-exp1-gnn-selection-report.md`](2026-11-14-exp1-gnn-selection-report.md).

## BWGNN

Source: github.com/squareRoot3/Rethinking-Anomaly-Detection, commit `de0631f039bbd19c1890b483cc01f1007f596af7` (ICML 2022;
no licence file; reimplemented, not copied). Code: `bwgnn_embed.py`, tests `tests/test_bwgnn.py` (16).

| ID | Decision | Decided by | Reason |
|---|---|---|---|
| BW1 | Homogeneous BWGNN with the official settings: hid 64, Adam lr 0.01, 100 epochs, full batch, cross-entropy weighted by #benign / #fraud train labels, no dropout, raw features, merged graph plus one self-loop per node. Embedding = ReLU(linear3(...)), 64 dims; score = softmax[:, 1]. Seven deviations (plain PyTorch, shared L^k, repo split, kept state, seed 0 on CPU, closed-form wavelet coefficients, merge mode only) are listed in the `bwgnn_embed.py` docstring. | Claude, plan approved by the user 2026-11-14 | Faithful to the official `--homo 1` model; Instagram has one relation, so homo and hetero coincide. |
| BW2 | Epoch rule: official BWGNN rule, first epoch with the highest validation Macro-F1 over thresholds 0.05..0.95, scored on the logits before that epoch's step; the state that produced them is kept. | User, 2026-11-14 ("use official for both") | Faithfulness, as for PMP. |
| BW3 | PMP checked against its official code (`main.py`: `monitor: auc_gnn`, `eval_interval: 1`, `best_metric <= now_metric`, patience, best checkpoint reloaded; `convert_probs`: sigmoid, column 1). The PMP runs already used this rule (D9, D11), so no PMP re-run. | User asked for the check, 2026-11-14 | Same rule as official. |
| BW4 | Variants C = 1 and C = 2 (wavelet order, S2). C = 1 chosen on GNN validation AUROC, 57.98 against 55.46 (rule of G1); official validation Macro-F1 56.58 against 54.26; best epochs 26 and 27. | Task rule | Validation only. |
| BW5 | Observations, not used for any choice: BWGNN-C1's validation AUROC (57.98) is below PMP-L1's (62.86), and below its own best epoch on AUROC (61.27 at epoch 7). Its neighbour sets differ from diffusion (Jaccard 0.302) and from PMP-L1 (0.288); validation label agreement 89.4 (diffusion 90.9, random 91.8); fraud targets 19.0 (diffusion 21.1; only 28 targets). | Recorded | `runs/diagnostics/bwgnn/`. |
| BW6 | Run config `configs/instagram_bwgnn_c1.yaml` (hash `771aa150b9`) differs from `instagram_pmp_l1.yaml` only in `method.selection_embeddings`. Estimated run: 37 new node summaries, 6,522 metapath summaries, prompts, 5 seeds, about 1.9 GPU-h (as the PMP run). Launch waits for the user. | Claude | Same one-change design as G4. |
| BW7 | Run finished: 5/5 seeds (hash `771aa150b9`), 05:17–07:05 UTC, all jobs rc 0. Test mean ± std: Macro-F1 51.88 ± 2.66, AUROC 59.04 ± 0.68, AUPRC 14.01 ± 0.88. | Recorded | `runs/results/raw/instagram_seed*_771aa150b9/`. |
| BW8 | Result against Ours K=1 (Holm over 6 tests, E5): AUROC Δ +0.18, 95% CI [−1.05, +1.44], Holm 1.0; Macro-F1 Δ −1.98 (p 0.29, Holm 1.0); AUPRC Δ −0.17 (Holm 1.0). BWGNN-C1 does not improve Ours on Instagram. Seeds 1 and 4 chose the lowest validation threshold (0.05) and account for the Macro-F1 gap (observation only). | Task spec (E1) | `runs/compare/2026-11-14-compare-arms-instagram-pmp-bwgnn.{md,json}`. |
| BW9 | No direct BWGNN-C1 vs PMP-L1 test (not asked for). | Claude | Offered to the user; not part of the E1 family. |
| E5 | Main Holm family widened from 3 to 6 tests (PMP-L1 and BWGNN-C1, 3 metrics each, against Ours K=1 on Instagram), as E1 requires. `significance.json` rewritten from the new compare_arms output (old file kept as `runs/compare/2026-11-14-significance.json.bak`). PMP-L1 Macro-F1 Holm moves from 0.033 to 0.065, so E3's "significantly worse" no longer holds; all other Holm values are 1.0. E3 is kept as the record of the 3-test analysis. | User approved, 2026-11-14 ("approve") | E1: all variant × dataset comparisons against Ours K=1 in one family. |
| BW10 | Reddit BWGNN: C = 1 chosen over C = 2 on GNN validation AUROC, 68.19 against 67.36 (best epochs 16 and 11). Config `configs/reddit_bwgnn_c1.yaml` (hash `98520ee79e`) differs from `reddit_pmp_l1.yaml` (hash `22ce4ff044`) only in `method.selection_embeddings`. | Task rule (G1, BW4) | Validation only. `runs/diagnostics/bwgnn_reddit/`. |
| BW11 | Reddit runs finished: 10/10 seeds (PMP-L1 and BWGNN-C1), queue 09:28 UTC 2026-11-14 to 07:24 UTC 2026-11-15, all jobs rc 0. Test mean ± std: PMP-L1 Macro-F1 53.62 ± 2.77, AUROC 64.32 ± 1.80, AUPRC 14.56 ± 0.94; BWGNN-C1 54.31 ± 1.58, 65.82 ± 1.33, 15.20 ± 0.74 (Ours K=1: 53.58 ± 3.20, 65.37 ± 1.89, 15.17 ± 1.23). | Recorded | `runs/results/raw/reddit_seed*_{22ce4ff044,98520ee79e}/`. |
| BW12 | Reddit result against Ours K=1 (Holm over 12 tests, E6): PMP-L1 AUROC Δ −1.05, 95% CI [−1.73, −0.35], lower on 5/5 seeds, raw p 0.009, Holm 0.107; BWGNN-C1 AUROC Δ +0.46, CI [−0.32, +1.16], Holm 1.0; BWGNN-C1 Macro-F1 Δ +0.73, CI [+0.22, +1.24], p 0.71, Holm 1.0. Neither GNN improves Ours on Reddit. | Task spec (E1) | `runs/compare/2026-11-15-compare-arms-instagram-reddit-pmp-bwgnn.{md,json}`. |
| E6 | Main Holm family widened from 6 to 12 tests (PMP-L1 and BWGNN-C1, 3 metrics each, against Ours K=1 on Instagram and Reddit), as E1 requires. `significance.json` rewritten from the new compare_arms output (6-test file kept as `runs/compare/2026-11-15-significance.json.bak`). Only existing value that moves: Instagram PMP-L1 Macro-F1 Holm 0.065 → 0.119. E3 and E5 are kept as the record of the 3- and 6-test analyses. | User approved, 2026-11-15 ("approve") | E1. |

Reports: [`2026-11-14-exp1-bwgnn-report.md`](2026-11-14-exp1-bwgnn-report.md) (Instagram),
[`2026-11-15-exp1-reddit-report.md`](2026-11-15-exp1-reddit-report.md) (Instagram and Reddit, 12-test family).
