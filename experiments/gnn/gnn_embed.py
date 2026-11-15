"""PMP (Partitioning Message Passing, Zhuo et al., ICLR 2024) in plain PyTorch, for node embeddings.

A reimplementation of the official LA-SAGE-S model (github.com/Xtra-Computing/PMP, commit 3f7629f), full batch, with
the cached DeBERTa text embeddings as node features. The embedding of a node is its last hidden representation before
the classification head: with proj=True, h right before dropout + Conv1d(k=1); with proj=False, see Deviations.

What one PMP layer computes (official LASAGESConv, aggregator 'mean'), for destination node v with layer input h_v and
original input features x_v:

    m_u      = fc_neigh(h_u) if in_dim > out_dim else h_u                 (lin_before_mp)
    S_c(v)   = sum over in-edges u -> v with label_unk[u] == c of m_u     c in {fraud=1, benign=0, unknown=2}
    a(v)     = sigmoid(fc_balance(h_v))                                   (2-layer MLP, ReLU, out 1)
    h'_v     = ReLU( fc_self(h_v) + F(S_1, x_v) + B(S_0, x_v) + a(v) F(S_2, x_v) + (1 - a(v)) B(S_2, x_v) )

where F / B are the fraud / benign LIMLPs. A LILinear is out = W (input * trans_src(src)) + b, with trans_src a
Linear(origin_infeat -> in_features) on the destination node's features `src`. Each relation runs the same layer (shared
weights); the per-relation outputs are concatenated ('cat') and passed through relation_mlp[l]; ReLU between layers,
not after the last one. The head is dropout + Conv1d(hid, 2, k=1). Scores are sigmoid(logit[:, 1]) as in the official
convert_probs (not a softmax).

Message direction: as DGL update_all, messages flow src -> dst along edges. A relation matrix A gives an edge u -> v for
every stored non-zero A[u, v] (dgl.from_scipy convention, row = src, col = dst), so v aggregates column v (A^T h). The
repo's matrices are symmetric, so the direction does not matter for them; graph_checks reports symmetry.

label_unk: a train node carries its true label (0 benign, 1 fraud); every other node (validation, test, unlabelled)
carries 2 = unknown. Validation labels are used only for early stopping/selection; test labels are never read.

Deviations from the official code
---------------------------------
1. Full batch: one Adam step per epoch on all train nodes, instead of DGL mini-batches of `batch_size` train nodes with
   MultiLayerFullNeighborSampler. The forward computation per node is the same (full neighbourhoods), but the number
   of optimiser steps per epoch differs (official: ceil(#train / batch_size)).
2. agg_func name/behaviour: the official 'mean' aggregator actually SUMS the messages of each partition (mailbox * mask,
   .sum(1)); no division by degree. This is reproduced as is (sums, not means). Nodes without in-edges get zeros, as DGL
   does.
3. lin_before_mp with in_dim > out_dim (768 -> 256 / 48 here): the official code applies fc_neigh before message
   passing but still builds the LIMLPs with input size in_dim, which crashes with a shape error (the partition sums are
   out_dim wide). Here the LIMLP input size is out_dim when lin_before_mp holds (in_dim otherwise), and trans_src maps
   origin_infeat -> that size. The official configs never hit this path (Amazon 25 -> 256, Yelp 32 -> 48).
4. trans_src input across layers: the official LASAGESConv passes src = h_self (the layer input of the destination node)
   while trans_src is Linear(origin_infeat, .), so any second layer with hid != input size crashes. Here src is the
   ORIGINAL input features x_v (after norm_feat) of the destination node in every layer, which is what origin_infeat
   implies; for layer 1 this is identical to the official code. fc_balance and fc_self still take the layer input h_v.
5. mlp_activation is accepted but, as in the official code, has no effect: LISeq only applies LILinear, Dropout and
   GraphNorm modules and skips activation modules, and with num_trans = 1 there is no hidden layer anyway.
6. The unused LILinear parameter w_src (initialised, never used in forward) is not created. Parameter initialisation
   otherwise follows the official code (LILinear kaiming_uniform(a=sqrt 5) weight, fan-in uniform bias, default
   nn.Linear trans_src; xavier_uniform(gain=relu) on fc_self and fc_neigh weights; default init elsewhere), but the
   random stream is not bit-compatible with the official run (different module order/count, no DGL, seed 0).
7. relation_mlp: the official loop builds every relation_mlp[j] with the last layer's width (it tests `i`, not `j`).
   With proj=True all layers are hid -> hid so this is harmless and reproduced; with proj=False the intermediate
   relation_mlps here use hid and only the last uses num_classes (the official code would crash for n_layer >= 2).
8. proj=False: the last layer outputs num_classes = 2 and there is no head, so the "embedding" is the 2-dim logit
   vector (num_nodes, 2). The grid only uses proj=True.
9. relation_mode 'merge' = official homo=True (dgl.to_homogeneous): the edge lists of all relations are concatenated,
   so an edge present in r relations is counted r times (multigraph, no de-duplication). 'cat' = official hetero with
   relation_agg 'cat'. Edge weights (matrix values) are ignored, as in the official model (no edge_weight is passed).
   With a single relation the two modes give the same architecture and the same numbers.
10. Early stopping: validation AUROC (official monitor auc_gnn) every epoch; the official update rule
   `best <= now` is kept (a tie moves the best epoch later), patience counts strictly worse epochs and training stops
   when it reaches `patience`. The best state is kept in memory instead of a checkpoint file. `early_stopping: false`
   (not in the official code) trains exactly `epochs` epochs and keeps the final state.
11. norm_feat replicates DatasetHelper.row_normalize: x / (rowsum(x) + 0.01), inf -> 0, computed in float64 and cast to
   float32. It was designed for non-negative features; on DeBERTa embeddings row sums can be negative or near -0.01,
   which flips signs and inflates rows (see graph/feature checks in the manifest). Reproduced as specified.
12. Seed 0 (official 1234 then a random per-run seed), CPU by default, torch.use_deterministic_algorithms(True) (on CUDA
   with warn_only=True). Numeric node attributes are not used (features are the DeBERTa embeddings only; Instagram and
   Reddit have none). AUROC uses sklearn roc_auc_score, equal to the official metrics.auc(roc_curve).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

BENIGN, FRAUD, UNKNOWN = 0, 1, 2

# The four grid configurations. A = config/amazon.yml (LA-SAGE-S), Y = config/yelp.yml (LA-SAGE-S); the second of each
# pair switches weighted_loss on. relation_mode is set separately (default merge). yelp.yml has no mlp_activation key:
# prepare_model defaults it to relu.
_AMAZON = {"hid_dim": 256, "dropout": 0.6, "lr": 0.01, "optimizer": "Adam", "weight_decay": 0.0, "epochs": 100,
           "patience": 20, "mlp_activation": "elu", "num_trans": 1, "proj": True, "norm_feat": True, "agg": "mean"}
_YELP = {"hid_dim": 48, "dropout": 0.0, "lr": 0.01, "optimizer": "Adam", "weight_decay": 0.0, "epochs": 500,
         "patience": 100, "mlp_activation": "relu", "num_trans": 1, "proj": True, "norm_feat": True, "agg": "mean"}
GRID: dict[str, dict] = {
    "A1": {**_AMAZON, "weighted_loss": False},
    "A2": {**_AMAZON, "weighted_loss": True},
    "Y1": {**_YELP, "weighted_loss": False},
    "Y2": {**_YELP, "weighted_loss": True},
    # Decision (2026-11-14, see 2026-11-14-exp1-decisions.md D1): the official row_normalize divides by the row sum,
    # which explodes on DeBERTa features (negative / near-zero sums; max |x| 4.3e3 on Instagram, 1.4e6 on Reddit).
    # The two remaining grid slots repeat A2 and Y2 on raw features. Validation AUROC still chooses the config.
    "A3": {**_AMAZON, "weighted_loss": True, "norm_feat": False},
    "Y3": {**_YELP, "weighted_loss": True, "norm_feat": False},
}


# ----------------------------------------------------------------------------------------------------------------------
# Determinism, features, labels
# ----------------------------------------------------------------------------------------------------------------------

def seed_everything(seed: int, device: str = "cpu") -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(True)


def row_normalize(features: np.ndarray) -> np.ndarray:
    """Official DatasetHelper.row_normalize: divide each row by (its sum + 0.01); an infinite inverse becomes 0."""
    x = np.asarray(features, dtype=np.float64)
    rowsum = x.sum(1) + 0.01
    with np.errstate(divide="ignore"):
        inv = np.power(rowsum, -1)
    inv[np.isinf(inv)] = 0.0
    return (x * inv[:, None]).astype(np.float32)


def feature_checks(features: np.ndarray) -> dict:
    rowsum = np.asarray(features, dtype=np.float64).sum(1) + 0.01
    return {"shape": list(features.shape), "rowsum_plus_0.01_min": float(rowsum.min()),
            "rowsum_plus_0.01_max": float(rowsum.max()), "rows_negative_rowsum": int((rowsum < 0).sum()),
            "rows_abs_rowsum_below_0.1": int((np.abs(rowsum) < 0.1).sum())}


def make_label_unk(num_nodes: int, train_idx: np.ndarray, train_y: np.ndarray) -> torch.Tensor:
    """2 everywhere, then the true label on train nodes only. Takes no validation or test labels at all."""
    train_y = np.asarray(train_y).astype(np.int64)
    if set(np.unique(train_y)) - {BENIGN, FRAUD}:
        raise ValueError("train labels must be 0 or 1")
    label_unk = torch.full((num_nodes,), UNKNOWN, dtype=torch.long)
    label_unk[torch.as_tensor(np.asarray(train_idx), dtype=torch.long)] = torch.as_tensor(train_y)
    return label_unk


def class_weight(train_y: np.ndarray) -> torch.Tensor:
    """Official prepare.py: [1, #benign / #fraud], from train labels only."""
    y = torch.as_tensor(np.asarray(train_y), dtype=torch.float32)
    return torch.tensor([1.0, float((1 - y).sum() / y.sum())])


@dataclass
class LabelledNodes:
    train_idx: np.ndarray
    train_y: np.ndarray
    val_idx: np.ndarray
    val_y: np.ndarray
    test_idx: np.ndarray            # ids only: test labels are never carried into training


def labelled_nodes(labels: np.ndarray, split) -> LabelledNodes:
    """Pick train and validation labels out of the full label vector; test nodes keep only their ids."""
    labels = np.asarray(labels)
    train_idx, val_idx = np.asarray(split.train, dtype=np.int64), np.asarray(split.val, dtype=np.int64)
    out = LabelledNodes(train_idx, labels[train_idx].astype(np.int64), val_idx, labels[val_idx].astype(np.int64),
                        np.asarray(split.test, dtype=np.int64))
    for name, y in (("train", out.train_y), ("val", out.val_y)):
        if set(np.unique(y)) - {0, 1}:
            raise ValueError(f"{name} split holds nodes without a 0/1 label")
    return out


def _expected_embedding_path(graph, cfg: dict) -> Path:
    """The path pipeline.embedding_path returns, computed without its side effect of creating the cache folder."""
    settings = {k: cfg["embedder"].get(k) for k in ("model_id", "revision", "max_length", "backend")}
    meta = {"dataset": graph.name, "nodes": graph.num_nodes, **settings}
    key = hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return Path(cfg["paths"]["cache"]) / "embeddings" / key / "text_embeddings.npy"


def load_text_embeddings(graph, cfg: dict) -> tuple[np.ndarray, Path]:
    """The cached DeBERTa embeddings of every node. A missing cache is an error: embeddings are never recomputed here."""
    from our_pipeline.pipeline import embedding_path

    expected = _expected_embedding_path(graph, cfg)
    if not expected.exists():
        raise FileNotFoundError(f"cached text embeddings not found at {expected}; build them with the pipeline first")
    path = embedding_path(graph, cfg)
    if path.resolve() != expected.resolve():
        raise RuntimeError(f"embedding_path gave {path}, expected {expected}")
    embeddings = np.load(path)
    if embeddings.shape[0] != graph.num_nodes:
        raise ValueError(f"{path} has {embeddings.shape[0]} rows for {graph.num_nodes} nodes")
    return np.asarray(embeddings, dtype=np.float32), path


# ----------------------------------------------------------------------------------------------------------------------
# Graph structure
# ----------------------------------------------------------------------------------------------------------------------

def _as_list(relations) -> list[tuple[str, sp.spmatrix]]:
    if isinstance(relations, Mapping):
        return [(str(k), v) for k, v in relations.items()]
    if isinstance(relations, Sequence):
        return [(f"r{i}", m) for i, m in enumerate(relations)]
    return [("r0", relations)]


def edge_list(matrix: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Edges u -> v for every stored non-zero A[u, v] (row = src, col = dst). Values are ignored."""
    coo = sp.coo_matrix(matrix)
    keep = coo.data != 0
    return coo.row[keep].astype(np.int64), coo.col[keep].astype(np.int64)


def relation_edges(relations, mode: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """'cat': one edge list per relation. 'merge': one concatenated edge list (multi-edges kept, as to_homogeneous)."""
    lists = [edge_list(m) for _, m in _as_list(relations)]
    if mode == "cat":
        return lists
    if mode == "merge":
        return [(np.concatenate([s for s, _ in lists]), np.concatenate([d for _, d in lists]))]
    raise ValueError(f"relation_mode must be 'merge' or 'cat', got {mode!r}")


def _structure_checks(src: np.ndarray, dst: np.ndarray, n: int) -> dict:
    a = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()   # duplicates summed = multiplicity
    off = a - sp.diags(a.diagonal())
    off.eliminate_zeros()
    degree = np.asarray((off != 0).sum(0)).ravel() + np.asarray((off != 0).sum(1)).ravel()
    return {"edges": int(len(src)), "distinct_edges": int(a.nnz), "duplicate_edges": int(len(src) - a.nnz),
            "symmetric": bool(abs(a - a.T).sum() == 0), "self_loops": int((a.diagonal() != 0).sum()),
            "isolated_nodes": int((degree == 0).sum())}


def graph_checks(relations) -> dict:
    """Per relation and for the merged graph: symmetry, self-loop count, isolated nodes (no edge to another node)."""
    named = _as_list(relations)
    n = named[0][1].shape[0]
    out = {"num_nodes": int(n), "relations": {}}
    for name, matrix in named:
        src, dst = edge_list(matrix)
        out["relations"][name] = _structure_checks(src, dst, n)
    src, dst = relation_edges(relations, "merge")[0]
    out["merged"] = _structure_checks(src, dst, n)
    return out


class PartitionGraph:
    """One relation's in-edges, split by the source node's label_unk into fraud / benign / unknown operators.

    partition_sums(h) returns (S_fraud, S_benign, S_unknown), S_c[v] = sum_{u -> v, label_unk[u] = c} h[u]: exactly the
    official mp_func / agg_func pair, computed as three sparse (dst x src) matrix products.
    """

    def __init__(self, src: np.ndarray, dst: np.ndarray, label_unk: torch.Tensor, num_nodes: int, device="cpu"):
        src_t, dst_t = torch.as_tensor(src, dtype=torch.long), torch.as_tensor(dst, dtype=torch.long)
        src_label = label_unk[src_t]
        self.ops = []
        for c in (FRAUD, BENIGN, UNKNOWN):
            mask = src_label == c
            idx = torch.stack([dst_t[mask], src_t[mask]])
            op = torch.sparse_coo_tensor(idx, torch.ones(int(mask.sum()), dtype=torch.float32),
                                         (num_nodes, num_nodes), check_invariants=False).coalesce()  # sums duplicates
            self.ops.append(op.to(device))

    def partition_sums(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fr, be, unk = (torch.sparse.mm(op, h) for op in self.ops)
        return fr, be, unk


# ----------------------------------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------------------------------

class LILinear(nn.Module):
    """out = W (input * trans_src(src)) + b; trans_src: Linear(origin_infeat, in_features)."""

    def __init__(self, in_features: int, out_features: int, origin_infeat: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.trans_src = nn.Linear(origin_infeat, in_features)
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        return F.linear(x * self.trans_src(src), self.weight, self.bias)


class LIMLP(nn.Module):
    """Chain of LILinears sharing the same src; like the official LISeq, no activation is applied in between."""

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_layers: int, origin_infeat: int):
        super().__init__()
        dims = [in_dim] + [hid_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(LILinear(a, b, origin_infeat) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src)
        return x


class PMPConv(nn.Module):
    """Official LASAGESConv with aggregator 'mean' (which sums per partition), feat_drop 0, activation ReLU."""

    def __init__(self, in_dim: int, out_dim: int, origin_infeat: int, num_trans: int, hid_feats: int | None = None):
        super().__init__()
        hid_feats = hid_feats or out_dim
        self.lin_before_mp = in_dim > out_dim
        mp_dim = out_dim if self.lin_before_mp else in_dim          # Deviation 3
        self.fc_neigh_benign = LIMLP(mp_dim, hid_feats, out_dim, num_trans, origin_infeat)
        self.fc_neigh_fraud = LIMLP(mp_dim, hid_feats, out_dim, num_trans, origin_infeat)
        self.fc_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_balance = nn.Sequential(nn.Linear(in_dim, hid_feats), nn.ReLU(), nn.Linear(hid_feats, 1))
        self.fc_self = nn.Linear(in_dim, out_dim)
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, h: torch.Tensor, x0: torch.Tensor, graph: PartitionGraph) -> torch.Tensor:
        messages = self.fc_neigh(h) if self.lin_before_mp else h
        s_fr, s_be, s_unk = graph.partition_sums(messages)
        neigh_fr = self.fc_neigh_fraud(s_fr, x0)
        neigh_be = self.fc_neigh_benign(s_be, x0)
        balance = torch.sigmoid(self.fc_balance(h))
        neigh_unk = balance * self.fc_neigh_fraud(s_unk, x0) + (1 - balance) * self.fc_neigh_benign(s_unk, x0)
        return F.relu(self.fc_self(h) + neigh_fr + neigh_be + neigh_unk)


class PMP(nn.Module):
    """Official LASAGE_S: n_layer PMPConvs shared across relations, relation_mlp per layer, dropout + Conv1d head."""

    def __init__(self, in_dim: int, hid_dim: int, n_layer: int, num_relations: int, num_trans: int = 1,
                 dropout: float = 0.0, proj: bool = True, num_classes: int = 2):
        super().__init__()
        self.proj, self.dropout = proj, dropout
        out_dim = hid_dim if proj else num_classes
        widths = [hid_dim] * (n_layer - 1) + [out_dim]
        self.layers = nn.ModuleList()
        d = in_dim
        for w in widths:
            self.layers.append(PMPConv(d, w, origin_infeat=in_dim, num_trans=num_trans, hid_feats=w))
            d = w
        self.relation_mlp = nn.ModuleList(nn.Linear(w * num_relations, w) for w in widths)   # Deviation 7
        self.conv = nn.Conv1d(out_dim, num_classes, kernel_size=1) if proj else None

    def embed(self, x: torch.Tensor, graphs: list[PartitionGraph]) -> torch.Tensor:
        h = x
        for i, (layer, mlp) in enumerate(zip(self.layers, self.relation_mlp)):
            h = mlp(torch.cat([layer(h, x, g) for g in graphs], dim=1))
            if i != len(self.layers) - 1:
                h = F.relu(h)
        return h

    def head(self, h: torch.Tensor) -> torch.Tensor:
        if not self.proj:
            return h
        if self.dropout > 0:
            h = F.dropout(h, self.dropout, training=self.training)
        return self.conv(h.t().unsqueeze(0)).squeeze(0).t()        # Conv1d(k=1) over nodes, as the official permutes

    def forward(self, x: torch.Tensor, graphs: list[PartitionGraph]) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(x, graphs)
        return h, self.head(h)


# ----------------------------------------------------------------------------------------------------------------------
# Training and selection
# ----------------------------------------------------------------------------------------------------------------------

def fraud_scores(logits: torch.Tensor) -> np.ndarray:
    """Official convert_probs: sigmoid on the logits, column 1."""
    return torch.sigmoid(logits[:, 1]).detach().cpu().double().numpy()


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


@dataclass
class TrainResult:
    embeddings: np.ndarray
    val_scores: np.ndarray
    val_auroc: float
    best_epoch: int                 # 0-based, as the official code counts
    epochs_run: int
    val_auroc_curve: list[float] = field(default_factory=list)
    train_loss_curve: list[float] = field(default_factory=list)


def train_pmp(features: np.ndarray, relations, nodes: LabelledNodes, hp: dict, n_layer: int,
              relation_mode: str = "merge", seed: int = 0, device: str = "cpu",
              early_stopping: bool = True, max_epochs: int | None = None) -> TrainResult:
    """Train one configuration full batch and return all-node embeddings from the selected state.

    Uses train labels for label_unk, the loss and the class weight; validation labels only for the stopping rule.
    """
    seed_everything(seed, device)
    x_np = row_normalize(features) if hp.get("norm_feat", False) else np.asarray(features, dtype=np.float32)
    n = x_np.shape[0]
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    label_unk = make_label_unk(n, nodes.train_idx, nodes.train_y)
    graphs = [PartitionGraph(s, d, label_unk, n, device) for s, d in relation_edges(relations, relation_mode)]

    model = PMP(x.shape[1], hp["hid_dim"], n_layer, num_relations=len(graphs), num_trans=hp.get("num_trans", 1),
                dropout=hp.get("dropout", 0.0), proj=hp.get("proj", True)).to(device)
    optimizer = getattr(torch.optim, hp.get("optimizer", "Adam"))(model.parameters(), lr=hp["lr"],
                                                                     weight_decay=hp.get("weight_decay", 0.0))
    weight = class_weight(nodes.train_y).to(device) if hp.get("weighted_loss", False) else None
    loss_fn = nn.CrossEntropyLoss(weight=weight, reduction="mean")
    train_idx = torch.as_tensor(nodes.train_idx, dtype=torch.long, device=device)
    train_y = torch.as_tensor(nodes.train_y, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(nodes.val_idx, dtype=torch.long, device=device)

    epochs = hp["epochs"] if max_epochs is None else min(hp["epochs"], max_epochs)
    patience = hp.get("patience", 0)
    best_metric, best_epoch, best_state, patience_cnt = -float("inf"), -1, None, 0
    curve, losses, epoch = [], [], -1
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        _, logits = model(x, graphs)
        loss = loss_fn(logits[train_idx], train_y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if not early_stopping:
            continue
        model.eval()
        with torch.no_grad():
            _, logits = model(x, graphs)
        metric = _auroc(nodes.val_y, fraud_scores(logits[val_idx]))
        curve.append(metric)
        if best_metric <= metric:                                  # official tie rule: later epoch wins
            best_metric, best_epoch, patience_cnt = metric, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
        if patience > 0 and patience_cnt >= patience:
            break
    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch
    model.eval()
    with torch.no_grad():
        h, logits = model(x, graphs)
    val_scores = fraud_scores(logits[val_idx])
    return TrainResult(embeddings=h.detach().cpu().numpy().astype(np.float32), val_scores=val_scores,
                       val_auroc=_auroc(nodes.val_y, val_scores), best_epoch=best_epoch, epochs_run=epoch + 1,
                       val_auroc_curve=curve, train_loss_curve=losses)


def select_config(results: dict[str, TrainResult], order: list[str]) -> str:
    """Highest validation AUROC; ties go to the earlier config in `order`."""
    best = order[0]
    for name in order[1:]:
        if results[name].val_auroc > results[best].val_auroc:
            best = name
    return best


TRAIN = train_pmp
SELECTION = "highest validation AUROC; ties go to the lower grid index"
EMBEDDING_WHAT = "last hidden representation before dropout + Conv1d head"
