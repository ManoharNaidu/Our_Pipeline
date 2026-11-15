"""BWGNN (Tang et al., "Rethinking Graph Neural Networks for Anomaly Detection", ICML 2022) for node embeddings.

Source: github.com/squareRoot3/Rethinking-Anomaly-Detection, commit de0631f039bbd19c1890b483cc01f1007f596af7 (no licence
file; reimplemented from the paper and that code, not copied). Homogeneous BWGNN only: the official `--homo 1` model.

The model, as in the official BWGNN class:
    h = ReLU(W2 ReLU(W1 x))
    for i = 0..C:   h_i = sum_k theta_i[k] L^k h          (beta-wavelet filter i of order C)
    z = ReLU(W3 [h_0 | ... | h_C])                         <- the embedding written by train_gnn.py
    logits = W4 z
with L = I - D^-1/2 A D^-1/2 over the merged graph plus one self-loop per node, D the in-degrees clamped at 1.

Training, as in the official main.py: full batch, Adam (lr 0.01), cross-entropy weighted by #benign / #fraud over the
train labels, 100 epochs, no dropout, no weight decay, raw features. Epoch rule: the epoch whose validation Macro-F1,
at the best of the 19 thresholds 0.05, 0.10, ..., 0.95, is strictly higher than every earlier epoch's (so the earliest
epoch wins ties). The official loop scores the logits computed *before* that epoch's optimiser step; the state kept
here is the one that produced the scored logits, so the embedding matches the selected validation scores exactly.

Deviations from the official code (each recorded in the decisions log):
  1. Plain PyTorch sparse matrices instead of DGL. Same arithmetic: copy_u + sum into the destination node.
  2. L^k h is computed once per order k and shared by the C + 1 filters; the official code recomputes it per filter.
     Same values.
  3. The repo's split (train / validation / test from split.json) instead of a random 40% / 20% / 40% split.
  4. The selected state is kept and used for the embedding; the official code only stores the test metrics of that
     epoch. No test metric is ever computed here.
  5. Seed 0, deterministic algorithms, CPU by default (gnn_embed.seed_everything).
  6. Wavelet coefficients from a closed form (binomial expansion) instead of sympy; tested against a sympy derivation.
  7. Only relation mode "merge" (official to_homogeneous + add_self_loop). The heterogeneous BWGNN is not supported.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import gnn_embed as G
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

GRID = {
    # Official README / main.py settings for BWGNN (homo): hid_dim 64, order from --layers, 100 epochs, Adam lr 0.01.
    "B1": {"hid_dim": 64, "lr": 0.01, "optimizer": "Adam", "weight_decay": 0.0, "epochs": 100, "weighted_loss": True,
           "self_loops": True},
}
SELECTION = ("epoch: first epoch with the highest validation Macro-F1 over thresholds 0.05..0.95 (official BWGNN); "
             "config: highest validation AUROC, ties go to the lower grid index")
EMBEDDING_WHAT = "ReLU(linear3(concatenated beta-wavelet filters)), the hidden layer before the output layer"
THRESHOLDS = np.linspace(0.05, 0.95, 19)


def beta_wavelet_thetas(order: int) -> list[list[float]]:
    """Coefficients, in increasing powers of L, of p_i(L) = (L/2)^i (1 - L/2)^(C-i) / B(i+1, C+1-i), i = 0..C."""
    thetas = []
    for i in range(order + 1):
        beta = math.gamma(i + 1) * math.gamma(order + 1 - i) / math.gamma(order + 2)
        coeff = [0.0] * (order + 1)
        for j in range(order - i + 1):                      # (L/2)^i * sum_j comb(C-i, j) (-L/2)^j
            coeff[i + j] += math.comb(order - i, j) * (-1) ** j / 2 ** (i + j) / beta
        thetas.append(coeff)
    return thetas


class Laplacian:
    """x -> x - D^-1/2 A D^-1/2 x, aggregating source features into destination nodes (DGL copy_u + sum)."""

    def __init__(self, src: np.ndarray, dst: np.ndarray, num_nodes: int, device: str = "cpu"):
        src_t = torch.as_tensor(src, dtype=torch.long)
        dst_t = torch.as_tensor(dst, dtype=torch.long)
        ones = torch.ones(len(src_t), dtype=torch.float32)
        self.adj = torch.sparse_coo_tensor(torch.stack([dst_t, src_t]), ones, (num_nodes, num_nodes)).coalesce().to(device)
        in_deg = torch.bincount(dst_t, minlength=num_nodes).float().clamp(min=1)
        self.d_invsqrt = in_deg.pow(-0.5).unsqueeze(-1).to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x - torch.sparse.mm(self.adj, x * self.d_invsqrt) * self.d_invsqrt


def homogeneous_edges(relations, num_nodes: int, self_loops: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Merged multigraph over all relations (an edge in two relations counts twice), plus one self-loop per node."""
    (src, dst), = G.relation_edges(relations, "merge")
    if self_loops:
        loops = np.arange(num_nodes, dtype=src.dtype)
        src, dst = np.concatenate([src, loops]), np.concatenate([dst, loops])
    return src, dst


class BWGNN(nn.Module):
    def __init__(self, in_feats: int, h_feats: int, order: int):
        super().__init__()
        self.thetas = beta_wavelet_thetas(order)
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.thetas), h_feats)
        self.linear4 = nn.Linear(h_feats, 2)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, lap: Laplacian) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.linear2(self.act(self.linear(x))))
        powers = [h]
        for _ in range(1, len(self.thetas)):
            powers.append(lap(powers[-1]))
        filtered = []
        for theta in self.thetas:
            out = theta[0] * powers[0]
            for k in range(1, len(theta)):
                out = out + theta[k] * powers[k]
            filtered.append(out)
        z = self.act(self.linear3(torch.cat(filtered, dim=-1)))
        return z, self.linear4(z)


def fraud_scores(logits: torch.Tensor) -> np.ndarray:
    """Official: softmax over the two classes, column 1."""
    return logits.softmax(1)[:, 1].detach().cpu().double().numpy()


def best_macro_f1(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Official get_best_f1: highest Macro-F1 over THRESHOLDS (strictly greater, so the lowest threshold wins ties)."""
    best_f1, best_thr = 0.0, 0.0
    for thr in THRESHOLDS:
        f1 = f1_score(y, (p > thr).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_f1, best_thr


@dataclass
class BWTrainResult(G.TrainResult):
    val_macro_f1_curve: list[float] = field(default_factory=list)
    best_val_macro_f1: float = float("nan")


def train_bwgnn(features: np.ndarray, relations, nodes: G.LabelledNodes, hp: dict, order: int,
                relation_mode: str = "merge", seed: int = 0, device: str = "cpu",
                early_stopping: bool = True, max_epochs: int | None = None) -> BWTrainResult:
    """Train one configuration full batch and return all-node embeddings from the selected state.

    Train labels: loss and class weight. Validation labels: the epoch rule only (none at all if early_stopping is
    False, when the last state is kept). Test labels: never seen (nodes.test_idx carries ids only).
    """
    if relation_mode != "merge":
        raise ValueError("BWGNN here is the official homogeneous model: relation_mode must be 'merge'")
    G.seed_everything(seed, device)
    x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)
    n = x.shape[0]
    src, dst = homogeneous_edges(relations, n, hp.get("self_loops", True))
    lap = Laplacian(src, dst, n, device)

    model = BWGNN(x.shape[1], hp["hid_dim"], order).to(device)
    optimizer = getattr(torch.optim, hp.get("optimizer", "Adam"))(model.parameters(), lr=hp["lr"],
                                                                     weight_decay=hp.get("weight_decay", 0.0))
    weight = G.class_weight(nodes.train_y).to(device) if hp.get("weighted_loss", True) else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    train_idx = torch.as_tensor(nodes.train_idx, dtype=torch.long, device=device)
    train_y = torch.as_tensor(nodes.train_y, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(nodes.val_idx, dtype=torch.long, device=device)

    epochs = hp["epochs"] if max_epochs is None else min(hp["epochs"], max_epochs)
    best_f1, best_epoch, best_state = 0.0, -1, None
    f1_curve, auroc_curve, losses, epoch = [], [], [], -1
    for epoch in range(epochs):
        model.train()
        _, logits = model(x, lap)
        loss = loss_fn(logits[train_idx], train_y)
        if early_stopping:                       # score these logits, as the official loop does, before the step
            p_val = fraud_scores(logits[val_idx])
            f1, _ = best_macro_f1(nodes.val_y, p_val)
            f1_curve.append(f1)
            auroc_curve.append(G._auroc(nodes.val_y, p_val))
            if best_f1 < f1:                     # official: strictly better, so the earliest epoch wins ties
                best_f1, best_epoch = f1, epoch
                best_state = copy.deepcopy(model.state_dict())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch
    model.eval()
    with torch.no_grad():
        z, logits = model(x, lap)
    val_scores = fraud_scores(logits[val_idx])
    return BWTrainResult(embeddings=z.detach().cpu().numpy().astype(np.float32), val_scores=val_scores,
                         val_auroc=G._auroc(nodes.val_y, val_scores), best_epoch=best_epoch, epochs_run=epoch + 1,
                         val_auroc_curve=auroc_curve, train_loss_curve=losses, val_macro_f1_curve=f1_curve,
                         best_val_macro_f1=best_macro_f1(nodes.val_y, val_scores)[0])


TRAIN = train_bwgnn
