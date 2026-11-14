"""Neighbour selection by cosine similarity on precomputed node embeddings, and the untouched diffusion default."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from our_pipeline.graph.data import load_graph, load_split
from our_pipeline.graph.trimming import enumerate_metapaths, metapath_name, metapath_neighbours, top_m_cosine
from our_pipeline.pipeline import build_prompts, trims_for
from our_pipeline.runs import run_seed
from our_pipeline.text.cache import artifact_dir


def quiet(_message):
    pass


def load_tiny(cfg):
    directory = Path(cfg["paths"]["data"]) / cfg["dataset"]["name"]
    return load_graph(directory), load_split(directory)


def original_trims_meta(graph, split, cfg) -> dict:
    """The trims cache key exactly as trims_for built it before selection existed."""
    method = cfg["method"]
    targets = split.targets()
    return {"dataset": graph.name, "split_seed": split.seed, "K": method["K"], "M": method["M"],
            "targets": hashlib.sha1(np.sort(targets).astype(np.int64).tobytes()).hexdigest()[:16],
            "embedder": cfg["embedder"].get("model_id", "hashing")}


def assert_same_trims(a, b):
    assert a.keys() == b.keys()
    for name in a:
        assert a[name].keys() == b[name].keys()
        for v in a[name]:
            assert np.array_equal(a[name][v], b[name][v])


def with_embeddings(cfg, path):
    out = copy.deepcopy(cfg)
    out["method"]["selection"] = "embedding_cosine"
    out["method"]["selection_embeddings"] = str(path)
    return out


def write_embeddings(path, rows, seed=0, dim=8):
    np.save(path, np.random.default_rng(seed).normal(size=(rows, dim)))
    return path


def test_top_m_cosine_orders_by_similarity():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 0.1], [-1.0, 0.0]])
    assert top_m_cosine(np.array([1, 2, 3, 4]), embeddings, 0, 4).tolist() == [3, 2, 1, 4]


def test_top_m_cosine_breaks_ties_by_id_and_keeps_at_most_m():
    embeddings = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.5, 0.0], [0.0, 1.0]])
    out = top_m_cosine(np.array([4, 3, 2, 1], dtype=np.int64), embeddings, 0, 2)
    assert out.tolist() == [1, 2] and out.dtype == np.int64
    assert len(top_m_cosine(np.array([4, 3], dtype=np.int64), embeddings, 0, 10)) == 2


def test_top_m_cosine_treats_zero_vectors_as_similarity_zero():
    embeddings = np.array([[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.1]])
    # similarities: 1 -> 0 (zero vector), 2 -> -1, 3 -> 0, 4 -> ~1; zero-vector ties with node 3 by id
    assert top_m_cosine(np.array([1, 2, 3, 4]), embeddings, 0, 4).tolist() == [4, 1, 3, 2]
    with np.errstate(all="raise"):
        assert top_m_cosine(np.array([1, 2]), np.zeros((3, 2)), 0, 2).tolist() == [1, 2]   # zero target too


def test_default_selection_keeps_the_original_cache_key_and_trims(tiny_config):
    graph, split = load_tiny(tiny_config)
    expected = artifact_dir(tiny_config["paths"]["cache"], "trims", original_trims_meta(graph, split, tiny_config))
    assert "selection" not in tiny_config["method"]
    trims, directory = trims_for(graph, split, tiny_config, "cpu", quiet)
    assert directory == expected

    explicit = copy.deepcopy(tiny_config)
    explicit["method"]["selection"] = "diffusion"
    explicit["paths"]["cache"] = tiny_config["paths"]["cache"] + "_explicit"     # recompute, do not reuse the cache
    again, other = trims_for(graph, split, explicit, "cpu", quiet)
    assert other.name == directory.name
    assert_same_trims(trims, again)


def test_embedding_cosine_trims_match_a_direct_computation(tiny_config, tmp_path):
    graph, split = load_tiny(tiny_config)
    file = write_embeddings(tmp_path / "emb.npy", graph.num_nodes)
    cfg = with_embeddings(tiny_config, file)
    _, diffusion_dir = trims_for(graph, split, tiny_config, "cpu", quiet)
    trims, directory = trims_for(graph, split, cfg, "cpu", quiet)
    assert directory.name != diffusion_dir.name

    embeddings, targets, M = np.load(file), split.targets(), cfg["method"]["M"]
    for path in enumerate_metapaths(cfg["dataset"]["relations"], cfg["method"]["K"]):
        for v, found in zip(targets, metapath_neighbours(graph.relations, path, np.asarray(targets, dtype=np.int64))):
            expected = top_m_cosine(found, embeddings, int(v), M) if len(found) else found
            assert np.array_equal(trims[metapath_name(path)][int(v)], expected)

    write_embeddings(file, graph.num_nodes, seed=1)
    assert trims_for(graph, split, cfg, "cpu", quiet)[1].name != directory.name


def test_embedding_cosine_never_runs_the_text_embedder(tiny_config, tmp_path, monkeypatch):
    import our_pipeline.pipeline as pipeline

    def refuse(*args, **kwargs):
        raise AssertionError("diffusion features must not be computed")

    monkeypatch.setattr(pipeline, "_diffusion_features", refuse)
    graph, split = load_tiny(tiny_config)
    cfg = with_embeddings(tiny_config, write_embeddings(tmp_path / "emb.npy", graph.num_nodes))
    trims_for(graph, split, cfg, "cpu", quiet)


def test_unknown_selection_is_refused(tiny_config):
    graph, split = load_tiny(tiny_config)
    tiny_config["method"]["selection"] = "nearest"
    with pytest.raises(ValueError):
        trims_for(graph, split, tiny_config, "cpu", quiet)


def test_bad_embeddings_are_refused(tiny_config, tmp_path):
    graph, split = load_tiny(tiny_config)
    with pytest.raises(ValueError):
        trims_for(graph, split, with_embeddings(tiny_config, write_embeddings(tmp_path / "short.npy", graph.num_nodes - 1)),
                  "cpu", quiet)
    bad = np.ones((graph.num_nodes, 8))
    bad[3, 2] = np.nan
    np.save(tmp_path / "nan.npy", bad)
    with pytest.raises(ValueError):
        trims_for(graph, split, with_embeddings(tiny_config, tmp_path / "nan.npy"), "cpu", quiet)


def test_embedding_cosine_end_to_end(tiny_config, tmp_path):
    graph, _ = load_tiny(tiny_config)
    cfg = with_embeddings(tiny_config, write_embeddings(tmp_path / "emb.npy", graph.num_nodes))
    prompts = build_prompts(cfg, "cpu", log=quiet)
    run_seed(cfg, prompts, 0, "cpu")
    run_dir = next((Path(cfg["paths"]["results"]) / "raw").iterdir())
    assert (run_dir / "metrics.json").exists()
