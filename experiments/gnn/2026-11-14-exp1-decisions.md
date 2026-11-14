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

Report: [`2026-11-14-exp1-gnn-selection-report.md`](2026-11-14-exp1-gnn-selection-report.md).
