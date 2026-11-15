"""BWGNN embedding code on a tiny synthetic graph: wavelet coefficients, Laplacian, shapes, determinism, label leakage."""

from types import SimpleNamespace

import bwgnn_embed as B
import gnn_embed as G
import numpy as np
import pytest
import scipy.sparse as sp
import torch

N, D = 40, 12
HP = {**B.GRID["B1"], "hid_dim": 8, "epochs": 15}


def _symmetric(seed, n, p):
    upper = sp.triu(sp.random(n, n, density=p, random_state=seed, data_rvs=np.ones), k=1)
    return ((upper + upper.T) > 0).astype(np.float32).tocsr()


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    features = rng.standard_normal((N, D)).astype(np.float32)
    relations = {"R1": _symmetric(1, N, 0.1), "R2": _symmetric(2, N, 0.08)}
    labels = np.zeros(N, dtype=np.int64)
    labels[rng.choice(N, 10, replace=False)] = 1
    order = rng.permutation(N)
    split = SimpleNamespace(train=np.sort(order[:16]), val=np.sort(order[16:28]), test=np.sort(order[28:]), seed=0)
    for part in (split.train, split.val, split.test):        # both classes in every part
        assert 0 < labels[part].sum() < len(part)
    return features, relations, labels, split


def _run(toy, labels=None, order=2, **kw):
    features, relations, base_labels, split = toy
    nodes = G.labelled_nodes(base_labels if labels is None else labels, split)
    return B.train_bwgnn(features, relations, nodes, {**HP, **kw.pop("hp", {})}, order, **kw)


def _flip(labels, idx, seed):
    out = labels.copy()
    out[idx] = 1 - np.random.default_rng(seed).permutation(labels[idx])
    return out


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_thetas_match_sympy_derivation_of_official_formula(order):
    sympy = pytest.importorskip("sympy")
    from scipy.special import beta
    x = sympy.symbols("x")
    for i, theta in enumerate(B.beta_wavelet_thetas(order)):
        poly = sympy.Poly((x / 2) ** i * (1 - x / 2) ** (order - i) / beta(i + 1, order + 1 - i), x)
        expected = [float(c) for c in reversed(poly.all_coeffs())]          # increasing powers, as the official code
        assert theta == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_thetas_order_one_by_hand():
    # p_0 = (1 - L/2) / B(1, 2) = 2 - L ; p_1 = (L/2) / B(2, 1) = L
    assert B.beta_wavelet_thetas(1) == [pytest.approx([2.0, -1.0]), pytest.approx([0.0, 1.0])]


def test_laplacian_matches_dense_reference():
    rng = np.random.default_rng(3)
    n, d = 9, 4
    r1 = _symmetric(4, n, 0.3)
    r2 = _symmetric(5, n, 0.3)
    r1[8, :] = 0
    r1[:, 8] = 0
    r2[8, :] = 0
    r2[:, 8] = 0                                     # node 8 isolated before self-loops
    src, dst = B.homogeneous_edges({"R1": sp.csr_matrix(r1), "R2": sp.csr_matrix(r2)}, n, self_loops=True)
    a = np.zeros((n, n))
    np.add.at(a, (dst, src), 1.0)                    # a[v, u] = number of edges u -> v (duplicates across relations)
    assert np.array_equal(np.diag(a), np.ones(n) + np.diag(r1.toarray() + r2.toarray()))
    deg = np.maximum(a.sum(1), 1)
    dense_lap = np.eye(n) - np.diag(deg ** -0.5) @ a @ np.diag(deg ** -0.5)
    h = rng.standard_normal((n, d))
    lap = B.Laplacian(src, dst, n)
    got = lap(torch.as_tensor(h, dtype=torch.float32)).numpy()
    assert np.allclose(got, dense_lap @ h, atol=1e-5)
    assert np.allclose(got[8], 0.0, atol=1e-6)       # isolated node with its self-loop: x - x = 0


def test_merge_counts_shared_edges_twice():
    a = sp.csr_matrix(([1, 1], ([0, 1], [1, 0])), shape=(3, 3))
    src, dst = B.homogeneous_edges({"A": a, "B": a.copy()}, 3, self_loops=False)
    assert len(src) == 4 and sorted(zip(src.tolist(), dst.tolist())) == [(0, 1), (0, 1), (1, 0), (1, 0)]


@pytest.mark.parametrize("order", [1, 2, 3])
def test_shapes(toy, order):
    res = _run(toy, order=order)
    assert res.embeddings.shape == (N, HP["hid_dim"]) and res.embeddings.dtype == np.float32
    assert np.isfinite(res.embeddings).all() and len(res.val_scores) == len(toy[3].val)
    assert res.epochs_run == HP["epochs"] and len(res.val_macro_f1_curve) == HP["epochs"]


def test_deterministic(toy):
    a, b = _run(toy), _run(toy)
    assert np.array_equal(a.embeddings, b.embeddings) and a.best_epoch == b.best_epoch


def test_epoch_rule_is_first_strict_maximum_and_state_matches(toy):
    res = _run(toy)
    curve = res.val_macro_f1_curve
    assert res.best_epoch == int(np.argmax(curve))                     # argmax returns the first maximum
    assert res.best_val_macro_f1 == pytest.approx(curve[res.best_epoch])  # kept state reproduces the scored logits


def test_test_labels_do_not_leak(toy):
    labels, split = toy[2], toy[3]
    changed = _flip(labels, split.test, 1)
    assert not np.array_equal(changed[split.test], labels[split.test])
    for order in (1, 2):
        a, b = _run(toy, order=order), _run(toy, labels=changed, order=order)
        assert np.array_equal(a.embeddings, b.embeddings) and a.best_epoch == b.best_epoch


def test_val_and_test_labels_do_not_leak_without_early_stopping(toy):
    labels, split = toy[2], toy[3]
    changed = _flip(_flip(labels, split.test, 2), split.val, 3)
    assert not np.array_equal(changed[split.val], labels[split.val])
    for order in (1, 2):
        a = _run(toy, order=order, early_stopping=False)
        b = _run(toy, labels=changed, order=order, early_stopping=False)
        assert a.best_epoch == HP["epochs"] - 1 and np.array_equal(a.embeddings, b.embeddings)


def test_cat_mode_rejected(toy):
    with pytest.raises(ValueError):
        _run(toy, relation_mode="cat")


def test_grid_is_official_settings():
    assert list(B.GRID) == ["B1"]
    assert B.GRID["B1"] == {"hid_dim": 64, "lr": 0.01, "optimizer": "Adam", "weight_decay": 0.0, "epochs": 100,
                            "weighted_loss": True, "self_loops": True}
    assert B.TRAIN is B.train_bwgnn and G.TRAIN is G.train_pmp
