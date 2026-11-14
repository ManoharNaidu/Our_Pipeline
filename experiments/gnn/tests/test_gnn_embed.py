"""PMP embedding code on a tiny synthetic graph: shapes, determinism, label leakage, partition sums, graph checks."""

from types import SimpleNamespace

import gnn_embed as G
import numpy as np
import pytest
import scipy.sparse as sp
import torch

N, D = 40, 12
HP = {"hid_dim": 8, "dropout": 0.5, "lr": 0.01, "optimizer": "Adam", "weight_decay": 0.0, "epochs": 12,
      "patience": 4, "mlp_activation": "relu", "num_trans": 1, "proj": True, "norm_feat": True, "agg": "mean",
      "weighted_loss": True}


def _symmetric(rng, n, p):
    upper = sp.triu(sp.random(n, n, density=p, random_state=rng, data_rvs=np.ones), k=1)
    return ((upper + upper.T) > 0).astype(np.float32).tocsr()


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    features = rng.random((N, D)).astype(np.float32)
    relations = {"R1": _symmetric(1, N, 0.1), "R2": _symmetric(2, N, 0.08)}
    labels = np.zeros(N, dtype=np.int64)
    labels[rng.choice(N, 10, replace=False)] = 1
    order = rng.permutation(N)
    split = SimpleNamespace(train=np.sort(order[:16]), val=np.sort(order[16:28]), test=np.sort(order[28:]), seed=0)
    for part in (split.train, split.val, split.test):        # both classes in every part
        assert 0 < labels[part].sum() < len(part)
    return features, relations, labels, split


def _run(toy, labels=None, **kw):
    features, relations, base_labels, split = toy
    nodes = G.labelled_nodes(base_labels if labels is None else labels, split)
    args = {"n_layer": 1, "relation_mode": "merge", "early_stopping": True, **kw}
    hp = {**HP, **args.pop("hp", {})}
    return G.train_pmp(features, relations, nodes, hp, **args)


def _flip(labels, idx, seed):
    out = labels.copy()
    out[idx] = 1 - np.random.default_rng(seed).permutation(labels[idx])
    return out


@pytest.mark.parametrize("n_layer", [1, 2])
@pytest.mark.parametrize("mode", ["merge", "cat"])
@pytest.mark.parametrize("hid", [8, 16])          # 8 < D: fc_neigh before message passing; 16 > D: after
def test_shapes(toy, n_layer, mode, hid):
    res = _run(toy, n_layer=n_layer, relation_mode=mode, hp={"hid_dim": hid})
    assert res.embeddings.shape == (N, hid) and res.embeddings.dtype == np.float32
    assert np.isfinite(res.embeddings).all()
    assert len(res.val_scores) == len(toy[3].val) and np.isfinite(res.val_scores).all()


def test_proj_false_gives_logits(toy):
    res = _run(toy, hp={"proj": False})
    assert res.embeddings.shape == (N, 2)


def test_deterministic(toy):
    a, b = _run(toy, n_layer=2), _run(toy, n_layer=2)
    assert np.array_equal(a.embeddings, b.embeddings) and a.best_epoch == b.best_epoch


def test_test_labels_do_not_leak(toy):
    labels, split = toy[2], toy[3]
    changed = _flip(labels, split.test, 1)
    assert not np.array_equal(changed[split.test], labels[split.test])
    for n_layer in (1, 2):
        a, b = _run(toy, n_layer=n_layer), _run(toy, labels=changed, n_layer=n_layer)
        assert np.array_equal(a.embeddings, b.embeddings)


def test_val_and_test_labels_do_not_leak_without_early_stopping(toy):
    labels, split = toy[2], toy[3]
    changed = _flip(_flip(labels, split.test, 2), split.val, 3)
    assert not np.array_equal(changed[split.val], labels[split.val])
    for n_layer in (1, 2):
        a = _run(toy, n_layer=n_layer, early_stopping=False)
        b = _run(toy, labels=changed, n_layer=n_layer, early_stopping=False)
        assert a.epochs_run == HP["epochs"] and np.array_equal(a.embeddings, b.embeddings)


def test_label_unk_and_class_weight_use_train_labels_only(toy):
    labels, split = toy[2], toy[3]
    nodes = G.labelled_nodes(labels, split)
    label_unk = G.make_label_unk(N, nodes.train_idx, nodes.train_y).numpy()
    others = np.setdiff1d(np.arange(N), split.train)
    assert (label_unk[others] == 2).all()
    assert np.array_equal(label_unk[split.train], labels[split.train])
    expected = (labels[split.train] == 0).sum() / (labels[split.train] == 1).sum()
    weight = G.class_weight(nodes.train_y)
    assert weight[0] == 1.0 and weight[1].item() == pytest.approx(expected)
    changed = G.labelled_nodes(_flip(_flip(labels, split.test, 4), split.val, 5), split)
    assert torch.equal(G.class_weight(changed.train_y), weight)
    assert np.array_equal(G.make_label_unk(N, changed.train_idx, changed.train_y).numpy(), label_unk)


def test_partition_sums_match_dense_reference():
    rng = np.random.default_rng(7)
    n, d = 15, 5
    r1 = (rng.random((n, n)) < 0.3).astype(np.float32)          # deliberately asymmetric, with self-loops
    r2 = (rng.random((n, n)) < 0.3).astype(np.float32) * 3.0      # values ignored: binary structure only
    label_unk = torch.as_tensor(rng.integers(0, 3, n))
    h = torch.as_tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    relations = [sp.csr_matrix(r1), sp.csr_matrix(r2)]
    (src, dst), = G.relation_edges(relations, "merge")
    sums = G.PartitionGraph(src, dst, label_unk, n).partition_sums(h)
    counts = torch.as_tensor(r1 + (r2 != 0), dtype=torch.float32)    # merged multigraph: edge in both counts twice
    for c, s in zip((1, 0, 2), sums):
        mask = (label_unk == c).float()[:, None]
        # explicit mailbox reference: S[v] = sum over edges u -> v (A[u, v]) of h[u] * [label_unk[u] == c]
        loop = torch.zeros(n, d)
        for u in range(n):
            for v in range(n):
                loop[v] += counts[u, v] * mask[u, 0] * h[u]
        assert torch.allclose(s, (counts * mask).t() @ h, atol=1e-5)
        assert torch.allclose(s, loop, atol=1e-5)
    for rel, (s_r, d_r) in zip(relations, G.relation_edges(relations, "cat")):
        fr = G.PartitionGraph(s_r, d_r, label_unk, n).partition_sums(h)[0]
        dense = torch.as_tensor((rel.toarray() != 0).astype(np.float32))
        assert torch.allclose(fr, (dense * (label_unk == 1).float()[:, None]).t() @ h, atol=1e-5)


def test_graph_checks_known_matrix():
    a = sp.csr_matrix(([1, 1, 1, 1], ([0, 1, 2, 4], [1, 0, 3, 4])), shape=(5, 5))   # 0-1 both ways, 2->3, loop 4
    b = sp.csr_matrix(([1, 1], ([0, 1], [1, 0])), shape=(5, 5))
    out = G.graph_checks({"A": a, "B": b})
    assert out["num_nodes"] == 5
    assert out["relations"]["A"] == {"edges": 4, "distinct_edges": 4, "duplicate_edges": 0, "symmetric": False,
                                     "self_loops": 1, "isolated_nodes": 1}
    assert out["relations"]["B"] == {"edges": 2, "distinct_edges": 2, "duplicate_edges": 0, "symmetric": True,
                                     "self_loops": 0, "isolated_nodes": 3}
    assert out["merged"] == {"edges": 6, "distinct_edges": 4, "duplicate_edges": 2, "symmetric": False,
                             "self_loops": 1, "isolated_nodes": 1}


def test_row_normalize_matches_official_formula():
    x = np.array([[1.0, 2.0, 3.0], [-0.01, 0.0, 0.0], [0.5, -0.5, 0.0]], dtype=np.float64)
    out = G.row_normalize(x)
    assert np.allclose(out[0], x[0] / 6.01) and np.allclose(out[2], x[2] / 0.01)
    assert np.array_equal(out[1], np.zeros(3, dtype=np.float32))          # rowsum + 0.01 == 0 -> inverse inf -> 0


def test_select_config_ties_go_to_lower_index():
    r = {k: SimpleNamespace(val_auroc=v) for k, v in {"A1": 0.7, "A2": 0.8, "Y1": 0.8, "Y2": 0.5}.items()}
    assert G.select_config(r, ["A1", "A2", "Y1", "Y2"]) == "A2"


def test_grid_values():
    assert list(G.GRID) == ["A1", "A2", "Y1", "Y2", "A3", "Y3"] and len(G.GRID) <= 6
    assert G.GRID["A1"]["hid_dim"] == 256 and G.GRID["Y1"]["hid_dim"] == 48
    assert not G.GRID["A1"]["weighted_loss"] and G.GRID["A2"]["weighted_loss"]
    for raw, normed in (("A3", "A2"), ("Y3", "Y2")):           # D1: the raw-feature copies differ only in norm_feat
        assert {k: v for k, v in G.GRID[raw].items() if k != "norm_feat"} == \
               {k: v for k, v in G.GRID[normed].items() if k != "norm_feat"}
        assert G.GRID[normed]["norm_feat"] and not G.GRID[raw]["norm_feat"]


def test_missing_embeddings_raise(tmp_path):
    graph = SimpleNamespace(name="toy", num_nodes=3)
    cfg = {"paths": {"cache": str(tmp_path)}, "embedder": {"model_id": "m", "revision": "r", "max_length": 8}}
    with pytest.raises(FileNotFoundError):
        G.load_text_embeddings(graph, cfg)
    assert not any(tmp_path.iterdir())                                   # nothing created
