"""Live progress dashboard for the GNN experiment (stdlib, plus optional numpy / our_pipeline).

Usage: python experiments/gnn/dashboard.py [--host 127.0.0.1] [--port 8765] [--root /workspace/Our_Pipeline]

Serves `/` (HTML, refreshes itself every 15 s in place) and `/api/state` (the same data as JSON). Every read is guarded,
so a missing or half-written file shows as "not started" / "unavailable" instead of breaking the page. A background
thread samples nvidia-smi and the summary stores every 15 s (GPU history: last 6 h, in memory only).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
from collections import deque
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - optional
    yaml = None

try:
    from our_pipeline.config import config_hash as _pkg_config_hash
    from our_pipeline.config import load_config as _pkg_load_config
except Exception:  # pragma: no cover - optional
    _pkg_config_hash = _pkg_load_config = None

ROOT = Path("/workspace/Our_Pipeline")
SAMPLE_SECONDS = 15
GPU_HISTORY = deque(maxlen=6 * 3600 // SAMPLE_SECONDS)
LOCK = threading.RLock()
STARTED = time.time()
JOB_RE = re.compile(r"^(?P<stem>.+?)_(?P<stage>trim|prompts|node_\d+|meta_\d+|train_s\d+)$")
RUN_RE = re.compile(r"^(?P<ds>.+)_seed(?P<seed>\d+)_(?P<hash>[0-9a-f]{6,})$")
STAGES = ("trim", "node", "meta", "prompts", "train")
STAGE_LABEL = {"trim": "trim", "node": "node shards", "meta": "meta shards", "prompts": "prompts",
               "train": "train seeds"}
PROC_PATTERNS = ("train_gnn.py", "generate_summaries.py", "scripts/train.py", "train.py", "run_queue.py",
                 "diagnose_selection.py", "compare_arms.py")


def gnn_dir() -> Path:
    return ROOT / "experiments" / "gnn"


def rel(p) -> str:
    try:
        return str(Path(p).relative_to(ROOT))
    except Exception:
        return str(p)


# ---------------------------------------------------------------- small helpers
def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def read_text(path: Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def read_jsonl(path: Path) -> list:
    out = []
    txt = read_text(path)
    for line in (txt or "").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def stat(path: Path):
    try:
        return Path(path).stat()
    except Exception:
        return None


def parse_t(s):
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except Exception:
        return None


def ts(t) -> float | None:
    dt = parse_t(t)
    return dt.timestamp() if dt else None


def iso(t: float | None) -> str | None:
    if t is None:
        return None
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def fmt_dur(sec) -> str:
    try:
        sec = int(round(float(sec)))
    except Exception:
        return ""
    if sec < 0:
        sec = 0
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}d{h:02d}h"
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def fmt_bytes(n) -> str:
    try:
        n = float(n)
    except Exception:
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def num(v):
    try:
        if isinstance(v, bool) or v is None:
            return None
        return float(v)
    except Exception:
        return None


def first(d: dict, *keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def scalar(v) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def tail_lines(path: Path, n: int = 40, max_bytes: int = 65536) -> str | None:
    """Last n lines, with carriage-return progress bars collapsed to their final state."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    lines = [ln.rsplit("\r", 1)[-1] if "\r" in ln.rstrip("\r") else ln.rstrip("\r") for ln in data.split("\n")]
    if size > max_bytes and lines:
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines[-n:])


def run_cmd(args: list, timeout: float = 5) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------- caches
_LINES: dict = {}       # path -> {ino, size, mtime, offset, count}
_GROWTH: dict = {}      # path -> deque[(t, lines)]
_SHA: dict = {}         # path -> (size, mtime, sha16)
_MEMO: dict = {}        # name -> (key, value)
_EPOCH_SEEN: dict = {}  # training_log path -> {epochs: first time seen}


def count_lines(path: Path) -> int | None:
    """Line count, read incrementally for append-only files (cached by inode/size/mtime)."""
    st = stat(path)
    if st is None:
        return None
    key = str(path)
    with LOCK:
        c = _LINES.get(key)
        if c and c["ino"] == st.st_ino and c["size"] == st.st_size and c["mtime"] == st.st_mtime:
            return c["count"]
        start, count = 0, 0
        if c and c["ino"] == st.st_ino and st.st_size >= c["offset"]:
            start, count = c["offset"], c["count"]
        try:
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read()
        except Exception:
            return c["count"] if c else None
        last_nl = data.rfind(b"\n")
        if last_nl >= 0:
            count += data.count(b"\n", 0, last_nl + 1)
            offset = start + last_nl + 1
        else:
            offset = start
        _LINES[key] = {"ino": st.st_ino, "size": st.st_size, "mtime": st.st_mtime, "offset": offset, "count": count}
        return count


def observe_growth(path: Path, lines: int | None) -> None:
    if lines is None:
        return
    with LOCK:
        h = _GROWTH.setdefault(str(path), deque(maxlen=720))
        now = time.time()
        if not h or h[-1][1] != lines or now - h[-1][0] >= SAMPLE_SECONDS - 1:
            h.append((now, lines))


def growth_rate(path: Path, window: float = 900.0) -> float | None:
    """Lines per second over the last `window` seconds of in-memory history."""
    with LOCK:
        h = list(_GROWTH.get(str(path), ()))
    if len(h) < 2:
        return None
    now_t, now_n = h[-1]
    old = [p for p in h if now_t - p[0] <= window] or h
    t0, n0 = old[0]
    if now_t - t0 < 20 or now_n <= n0:
        return 0.0 if now_t - t0 >= 120 and now_n == n0 else None
    return (now_n - n0) / (now_t - t0)


def file_sha16(path: Path) -> str | None:
    st = stat(path)
    if st is None:
        return None
    with LOCK:
        c = _SHA.get(str(path))
        if c and c[0] == st.st_size and c[1] == st.st_mtime:
            return c[2]
    try:
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except Exception:
        return None
    with LOCK:
        _SHA[str(path)] = (st.st_size, st.st_mtime, sha)
    return sha


def memo(name: str, key, fn):
    with LOCK:
        c = _MEMO.get(name)
        if c and c[0] == key:
            return c[1]
    val = fn()
    with LOCK:
        _MEMO[name] = (key, val)
    return val


# ---------------------------------------------------------------- config loading
def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def load_config(path: Path) -> dict:
    if _pkg_load_config is not None:
        return _pkg_load_config(path)
    if yaml is None:
        raise RuntimeError("neither our_pipeline nor yaml importable")
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    merged: dict = {}
    for parent in raw.pop("defaults", []) or []:
        merged = _deep_merge(merged, load_config(Path(path).parent / parent))
    return _deep_merge(merged, raw)


def config_hash(cfg: dict) -> str:
    if _pkg_config_hash is not None:
        return _pkg_config_hash(cfg)
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def flatten(d, prefix: str = "") -> dict:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v:
                out.update(flatten(v, key))
            else:
                out[key] = v
    return out


def yaml_files_key() -> tuple:
    files = list((ROOT / "configs").rglob("*.yaml")) + list((gnn_dir() / "configs").glob("*.yaml"))
    return tuple(sorted((str(p), getattr(stat(p), "st_mtime", 0)) for p in files))


def _load_configs() -> list:
    res = []
    for p in sorted((gnn_dir() / "configs").glob("*.yaml")):
        entry = {"stem": p.stem, "path": rel(p), "text": read_text(p), "overlay": p.name.startswith("_")}
        if entry["overlay"]:
            res.append(entry)
            continue
        try:
            cfg = load_config(p)
            ds = cfg["dataset"]["name"]
            entry.update(dataset=ds, hash=config_hash(cfg), seeds=list(cfg.get("seeds") or []),
                         max_epochs=(cfg.get("training") or {}).get("max_epochs"),
                         selection_rules=(cfg.get("training") or {}).get("selection_rules"),
                         primary_rule=(cfg.get("training") or {}).get("primary_rule"),
                         results=(cfg.get("paths") or {}).get("results", "runs/results"),
                         cache=(cfg.get("paths") or {}).get("cache", "runs/cache"),
                         selection=(cfg.get("method") or {}).get("selection"),
                         selection_embeddings=(cfg.get("method") or {}).get("selection_embeddings"),
                         summarizer=(cfg.get("summarizer") or {}).get("model_id"))
            bpath = ROOT / "configs" / "ours" / f"{ds}_k1.yaml"
            entry["baseline_path"] = rel(bpath)
            try:
                bcfg = load_config(bpath)
                entry["baseline_hash"] = config_hash(bcfg)
                entry["baseline_results"] = (bcfg.get("paths") or {}).get("results", "runs/results")
                fa, fb = flatten(cfg), flatten(bcfg)
                entry["diff"] = [{"key": k, "baseline": fb.get(k, "(absent)"), "experiment": fa.get(k, "(absent)")}
                                 for k in sorted(set(fa) | set(fb)) if fa.get(k, "(absent)") != fb.get(k, "(absent)")]
            except Exception as exc:
                entry["baseline_error"] = repr(exc)
        except Exception as exc:
            entry["error"] = repr(exc)
        res.append(entry)
    return res


def get_configs() -> list:
    try:
        return memo("configs", yaml_files_key(), _load_configs)
    except Exception as exc:
        return [{"stem": "?", "error": repr(exc)}]


def arms() -> list:
    return [c for c in get_configs() if not c.get("overlay") and c.get("hash")]


# ---------------------------------------------------------------- data collection
def get_progress() -> dict | None:
    d = read_json(gnn_dir() / "runs" / "progress.json")
    return d if isinstance(d, dict) else None


def get_estimates():
    return read_json(gnn_dir() / "runs" / "estimates.json")


def get_decisions() -> dict:
    files = sorted(gnn_dir().glob("*decisions*.md"))
    return {"files": [{"path": rel(p), "text": read_text(p)} for p in files]}


def norm_grid_entry(g: dict) -> dict:
    hp = first(g, "hyperparameters", "config", "params", "hparams", default=None)
    if not isinstance(hp, dict):
        hp = {k: v for k, v in g.items() if scalar(v) and k not in ("name",)}
    return {
        "name": first(g, "name", "config", "id", default="?") if not isinstance(g.get("config"), dict)
        else first(g, "name", "id", default="?"),
        "hp": hp,
        "val_auroc": num(first(g, "val_auroc", "auroc", "val_auc")),
        "val_macro_f1": num(first(g, "val_macro_f1", "macro_f1", "val_f1")),
        "val_threshold": num(first(g, "val_threshold", "threshold")),
        "best_epoch": first(g, "best_epoch_0based", "best_epoch"),
        "epochs_run": first(g, "epochs_run", "epochs_ran", "n_epochs"),
        "seconds": num(first(g, "seconds", "train_seconds", "runtime_seconds")),
        "val_auroc_curve": g.get("val_auroc_curve") if isinstance(g.get("val_auroc_curve"), list) else None,
        "train_loss_curve": g.get("train_loss_curve") if isinstance(g.get("train_loss_curve"), list) else None,
    }


def get_embeddings() -> list:
    res = []
    try:
        mfs = sorted((gnn_dir() / "runs" / "embeddings").glob("*/manifest.json"))
    except Exception:
        mfs = []
    trims = cosine_trims()
    for mf in mfs:
        d = read_json(mf)
        if not isinstance(d, dict):
            res.append({"folder": mf.parent.name, "error": "manifest unreadable"})
            continue
        try:
            grid_raw = first(d, "grid", "configs", "results", default=[])
            if isinstance(grid_raw, dict):
                grid_raw = [{"name": k, **v} if isinstance(v, dict) else {"name": k, "value": v}
                            for k, v in grid_raw.items()]
            grid = [norm_grid_entry(g) for g in grid_raw if isinstance(g, dict)]
            chosen = d.get("chosen")
            cc = d.get("chosen_config")
            if not isinstance(chosen, str):
                chosen = first(cc, "name") if isinstance(cc, dict) else (cc if isinstance(cc, str) else None)
            emb_out = d.get("embedding_output") if isinstance(d.get("embedding_output"), dict) else {}
            emb_path = ROOT / str(emb_out.get("path") or rel(mf.parent / "embeddings.npy"))
            if not emb_path.exists():
                emb_path = mf.parent / "embeddings.npy"
            sha = file_sha16(emb_path)
            st = stat(emb_path)
            res.append({
                "folder": mf.parent.name, "path": rel(mf),
                "dataset": d.get("dataset"), "model": d.get("model"), "layers": d.get("layers"),
                "seed": d.get("seed"), "device": d.get("device"), "relation_mode": d.get("relation_mode"),
                "config_file": d.get("config_file"), "selection_rule": d.get("selection"),
                "chosen": chosen, "grid": grid,
                "label_counts": d.get("label_counts"), "split": d.get("split"),
                "graph_checks": d.get("graph_checks"), "features": d.get("features"),
                "embedding": {**emb_out, "file": rel(emb_path), "bytes": st.st_size if st else None,
                              "sha256_16": sha},
                "used_by_trims": [t["folder"] for t in trims if sha and t.get("selection_embeddings_sha") == sha],
                "runtime_seconds": d.get("runtime_seconds"), "git_commit": d.get("git_commit"),
                "versions": d.get("versions"), "test_metrics": d.get("test_metrics"),
                "settings_scalars": {k: v for k, v in (d.get("settings") or {}).items() if scalar(v)},
            })
        except Exception as exc:
            res.append({"folder": mf.parent.name, "error": repr(exc)})
    return res


def get_diagnostics() -> dict:
    ddir = gnn_dir() / "runs" / "diagnostics"
    out = {"json": [], "md": []}
    try:
        for p in sorted(ddir.glob("*.json")):
            out["json"].append({"path": rel(p), "data": read_json(p)})
        for p in sorted(ddir.glob("*.md")):
            out["md"].append({"path": rel(p), "text": read_text(p)})
    except Exception:
        pass
    return out


def get_processes() -> list:
    txt = run_cmd(["ps", "-eo", "pid=,ppid=,etimes=,rss=,pcpu=,args="])
    res = []
    for line in (txt or "").splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        args = parts[5]
        if not any(p in args for p in PROC_PATTERNS) or args.startswith(("bash -c", "/bin/sh -c", "sh -c")):
            continue
        if "grep" in args.split()[0] or "dashboard.py" in args:
            continue
        try:
            res.append({"pid": int(parts[0]), "ppid": int(parts[1]), "elapsed_s": int(parts[2]),
                        "elapsed": fmt_dur(parts[2]), "rss_mb": round(int(parts[3]) / 1024, 1),
                        "cpu_pct": float(parts[4]), "args": args})
        except Exception:
            continue
    return res


def queue_cmdline_info(procs: list) -> dict:
    for p in procs:
        if "run_queue.py" in p["args"]:
            toks = p["args"].split()
            info = {"pid": p["pid"], "args": p["args"], "configs": [t for t in toks if t.endswith(".yaml")]}
            for flag in ("--shards", "--out", "--only"):
                if flag in toks and toks.index(flag) + 1 < len(toks):
                    info[flag.lstrip("-")] = toks[toks.index(flag) + 1]
            info["gpus"] = []
            if "--gpus" in toks:
                for t in toks[toks.index("--gpus") + 1:]:
                    if t.startswith("-"):
                        break
                    info["gpus"].append(t)
            return info
    return {}


def get_queue(procs: list) -> dict | None:
    qdir = gnn_dir() / "runs" / "queue"
    sf = qdir / "status.jsonl"
    events = read_jsonl(sf)
    done_markers: set = set()
    try:
        if (qdir / "done").is_dir():
            done_markers = {p.name for p in (qdir / "done").iterdir()}
    except Exception:
        pass
    if not events and not done_markers and not qdir.exists():
        return None
    now = time.time()
    qinfo = queue_cmdline_info(procs)
    alive = bool(qinfo)
    jobs: dict = {}
    meta = {"queue_start": None, "queue_end": None, "jobs_declared": None, "unfinished": None, "failed": None,
            "events": len(events), "process_alive": alive, "cmdline": qinfo}
    for ev in events:
        kind, t = ev.get("event") or ev.get("type"), ev.get("t")
        if kind == "queue_start":
            meta.update(queue_start=t, jobs_declared=ev.get("jobs"), queue_end=None)
            continue
        if kind == "queue_end":
            meta.update(queue_end=t, unfinished=ev.get("unfinished"), failed=ev.get("failed"))
            continue
        name = ev.get("job")
        if not name:
            continue
        j = jobs.setdefault(name, {"job": name, "state": "pending", "attempts": 0, "gpu": None, "start": None,
                                   "end": None, "seconds": None, "rc": None, "history": []})
        if kind == "start":
            j.update(state="running", gpu=ev.get("gpu"), start=t, end=None, seconds=None, rc=None)
            j["attempts"] = max(j["attempts"] + 1, int(ev.get("attempt") or 0))
            j["history"].append({"attempt": j["attempts"], "start": t, "end": None, "rc": None, "seconds": None})
        elif kind == "end":
            rc = ev.get("rc")
            j.update(rc=rc, end=t, seconds=ev.get("seconds"), state="done" if rc == 0 else "failed-attempt")
            if j["history"]:
                j["history"][-1].update(end=t, rc=rc, seconds=ev.get("seconds"))
        elif kind == "gave_up":
            j["state"] = "gave_up"
    for name in done_markers:
        jobs.setdefault(name, {"job": name, "state": "done", "attempts": 0, "gpu": None, "start": None, "end": None,
                               "seconds": None, "rc": 0, "history": []})
    for j in jobs.values():
        if j["state"] == "running" and not alive:
            j["state"] = "interrupted"
        if j["state"] in ("running", "interrupted"):
            st = ts(j["start"])
            j["elapsed_s"] = now - st if st else None
        j["done_marker"] = j["job"] in done_markers
        m = JOB_RE.match(j["job"])
        j["stem"] = m.group("stem") if m else j["job"]
        stage = m.group("stage") if m else "other"
        j["stage"] = next((s for s in STAGES if stage.startswith(s)), "other")
        lp = qdir / "logs" / f"{j['job']}.log"
        j["log"] = rel(lp) if lp.exists() else None

    # shard count: command line, else queue_start job count, else observed indices
    shards = None
    try:
        shards = int(qinfo.get("shards")) if qinfo.get("shards") else None
    except Exception:
        shards = None
    arm_by_stem = {c["stem"]: c for c in arms()}
    stems_seen = list(dict.fromkeys(j["stem"] for j in jobs.values()))
    if shards is None and meta["jobs_declared"] and stems_seen:
        n_seeds = sum(len(arm_by_stem.get(s, {}).get("seeds") or [0] * 5) for s in stems_seen)
        guess = (int(meta["jobs_declared"]) - 2 * len(stems_seen) - n_seeds) / (2 * len(stems_seen))
        shards = int(guess) if guess >= 1 and guess == int(guess) else None
    if shards is None:
        idx = [int(j["job"].rsplit("_", 1)[1]) for j in jobs.values() if j["stage"] in ("node", "meta")]
        shards = max(idx) + 1 if idx else 8
    stems = {}
    for stem in list(dict.fromkeys(stems_seen + [c["stem"] for c in arms()])):
        cfg = arm_by_stem.get(stem, {})
        sj = [j for j in jobs.values() if j["stem"] == stem]
        expected = {"trim": 1, "node": shards, "meta": shards, "prompts": 1,
                    "train": len(cfg.get("seeds") or []) or sum(j["stage"] == "train" for j in sj) or 5}
        stages = {}
        for s in STAGES:
            sjs = [j for j in sj if j["stage"] == s]
            stages[s] = {"total": max(expected[s], len(sjs)),
                         "done": sum(j["state"] == "done" or j["done_marker"] for j in sjs),
                         "running": sum(j["state"] == "running" for j in sjs),
                         "failed_attempts": sum(sum(1 for h in j["history"] if h["rc"] not in (0, None)) for j in sjs),
                         "gave_up": sum(j["state"] == "gave_up" for j in sjs)}
        stems[stem] = {"stem": stem, "in_queue": bool(sj), "stages": stages,
                       "done": sum(v["done"] for v in stages.values()),
                       "total": sum(v["total"] for v in stages.values())}
    order = {"running": 0, "interrupted": 1, "failed-attempt": 2, "gave_up": 3, "pending": 4, "done": 5}
    jl = sorted(jobs.values(), key=lambda j: (order.get(j["state"], 9), ts(j["start"]) or 0, j["job"]))
    return {"meta": meta, "shards": shards, "jobs": jl, "stems": list(stems.values()),
            "done_markers": sorted(done_markers), "status_file": rel(sf)}


def cosine_trims() -> list:
    res = []
    try:
        for mp in sorted((ROOT / "runs" / "cache" / "trims").glob("*/metadata.json")):
            m = read_json(mp)
            if isinstance(m, dict) and m.get("selection") == "embedding_cosine":
                st = stat(mp)
                res.append({**{k: v for k, v in m.items() if scalar(v)}, "folder": mp.parent.name,
                            "created": iso(st.st_mtime) if st else None,
                            "files": sorted(p.name for p in mp.parent.iterdir() if p.name != "metadata.json")})
    except Exception:
        pass
    return res


def _store_progress(path: Path, required, created: float | None) -> dict:
    lines = count_lines(path)
    observe_growth(path, lines)
    st = stat(path)
    rate = growth_rate(path)
    rate_src = "live (last 15 min)" if rate else None
    if not rate and lines and st and created and st.st_mtime > created + 5:
        rate = lines / (st.st_mtime - created)
        rate_src = "average since store creation"
    req = int(required) if isinstance(required, (int, float)) else None
    remaining = max(req - (lines or 0), 0) if req is not None else None
    eta = remaining / rate if rate and remaining else (0 if remaining == 0 else None)
    with LOCK:
        h = list(_GROWTH.get(str(path), ()))
    return {"file": rel(path), "exists": st is not None, "lines": lines, "required": req, "remaining": remaining,
            "pct": (100.0 * min(lines or 0, req) / req) if req else None, "bytes": st.st_size if st else None,
            "mtime": iso(st.st_mtime) if st else None, "idle_s": time.time() - st.st_mtime if st else None,
            "rate_per_min": rate * 60 if rate else (0.0 if rate == 0 else None), "rate_source": rate_src,
            "eta_s": eta, "first_seen_lines": h[0][1] if h else None, "first_seen": iso(h[0][0]) if h else None,
            "new_since_dashboard_start": (lines - h[0][1]) if h and lines is not None else None}


def get_summary_stores() -> dict:
    est = get_estimates()
    est = est if isinstance(est, dict) else {}
    req_meta = est.get("required_metapath_summaries") or {}
    req_node = est.get("required_new_node_summaries") or {}
    cache = ROOT / "runs" / "cache"
    trims = cosine_trims()
    meta_stores = []
    try:
        all_meta = [(p, read_json(p)) for p in sorted((cache / "metapath_summaries").glob("*/metadata.json"))]
    except Exception:
        all_meta = []
    for t in trims:
        found = [(p, m) for p, m in all_meta if isinstance(m, dict) and m.get("trims") == t["folder"]]
        if not found:
            meta_stores.append({"trims": t["folder"], "dataset": t.get("dataset"), "folder": None,
                                "required": req_meta.get(t.get("dataset")), "lines": 0, "exists": False})
        for p, m in found:
            st = stat(p)
            prog = _store_progress(p.parent / "summaries.jsonl", req_meta.get(m.get("dataset")),
                                   st.st_mtime if st else None)
            meta_stores.append({"trims": t["folder"], "dataset": m.get("dataset"), "folder": p.parent.name,
                                "node_store": m.get("node_store"), "model": m.get("model"), **prog})
    node_stores = []
    datasets = list(dict.fromkeys([t.get("dataset") for t in trims] + [c.get("dataset") for c in arms()]))
    try:
        for p in sorted((cache / "node_summaries").glob("*/metadata.json")):
            m = read_json(p)
            if not isinstance(m, dict) or m.get("dataset") not in datasets or "gemma" not in str(m.get("model")).lower():
                continue
            prog = _store_progress(p.parent / "summaries.jsonl", None, None)
            prog["required_new"] = req_node.get(m.get("dataset"))
            node_stores.append({"dataset": m.get("dataset"), "folder": p.parent.name, "model": m.get("model"),
                                **prog})
    except Exception:
        pass
    return {"trims": trims, "metapath": meta_stores, "node": node_stores}


def baseline_stats(ds: str, bhash: str, bresults: str) -> dict:
    """Per-seed timing of the stored baseline runs (and its queue log), for ETA fallbacks."""
    raw = ROOT / bresults / "raw"
    mfs = sorted(raw.glob(f"{ds}_seed*_{bhash}/run_manifest.json"))
    key = tuple((str(p), getattr(stat(p), "st_mtime", 0)) for p in mfs) + (getattr(
        stat(ROOT / "runs" / "queue" / "status.jsonl"), "st_mtime", 0),)

    def compute():
        rt, tr, eps = [], [], []
        for p in mfs:
            d = read_json(p) or {}
            if num(d.get("runtime_seconds")):
                rt.append(float(d["runtime_seconds"]))
            if num(d.get("train_seconds")):
                tr.append(float(d["train_seconds"]))
                n = count_lines(p.parent / "training_log.jsonl")
                if n:
                    eps.append(float(d["train_seconds"]) / n)
        job_secs = []
        for ev in read_jsonl(ROOT / "runs" / "queue" / "status.jsonl"):
            if ev.get("event") == "end" and ev.get("rc") == 0 and str(ev.get("job", "")).startswith(f"{ds}_k1_train_s"):
                if num(ev.get("seconds")):
                    job_secs.append(float(ev["seconds"]))
        return {"runs": len(mfs), "runtime_s": statistics.mean(rt) if rt else None,
                "train_s": statistics.mean(tr) if tr else None, "s_per_epoch": statistics.mean(eps) if eps else None,
                "job_s": statistics.mean(job_secs) if job_secs else None}

    try:
        return memo(f"baseline:{ds}:{bhash}", key, compute)
    except Exception:
        return {}


def get_training(queue: dict | None) -> dict:
    raw = gnn_dir() / "runs" / "results" / "raw"
    jobs = {j["job"]: j for j in (queue or {}).get("jobs", [])}
    by_hash = {c["hash"]: c for c in arms()}
    est = get_estimates()
    est = est if isinstance(est, dict) else {}
    runs = []
    try:
        folders = sorted(p for p in raw.iterdir() if p.is_dir())
    except Exception:
        folders = []
    now = time.time()
    for f in folders:
        try:
            m = RUN_RE.match(f.name)
            ds, seed, h = (m.group("ds"), int(m.group("seed")), m.group("hash")) if m else (None, None, None)
            cfg = by_hash.get(h, {})
            tl = f / "training_log.jsonl"
            rows = read_jsonl(tl)
            n = len(rows)
            st = stat(tl)
            with LOCK:
                seen = _EPOCH_SEEN.setdefault(str(tl), {})
                if n and n not in seen:
                    seen[n] = st.st_mtime if st else now
                seen_items = sorted(seen.items())
            max_ep = int(cfg.get("max_epochs") or 10)
            metrics = read_json(f / "metrics.json")
            manifest = read_json(f / "run_manifest.json")
            job = jobs.get(f"{cfg.get('stem')}_train_s{seed}") if cfg else None
            finished = isinstance(metrics, dict)
            if finished:
                status = "finished"
            elif job and job["state"] == "running":
                status = "running"
            elif job and job["state"] in ("failed-attempt", "gave_up", "interrupted"):
                status = job["state"]
            else:
                status = "stalled" if st and now - st.st_mtime > 3 * 3600 else "incomplete"
            base = baseline_stats(ds, cfg.get("baseline_hash", ""), cfg.get("baseline_results", "runs/results")) \
                if cfg.get("baseline_hash") else {}
            spe, spe_src = None, None
            if isinstance(manifest, dict) and num(manifest.get("train_seconds")) and n:
                spe, spe_src = float(manifest["train_seconds"]) / n, "run_manifest train_seconds"
            elif len(seen_items) >= 2 and seen_items[-1][0] > seen_items[0][0]:
                spe = (seen_items[-1][1] - seen_items[0][1]) / (seen_items[-1][0] - seen_items[0][0])
                spe_src = "observed epoch times"
            elif n and job and ts(job.get("start")) and st:
                spe, spe_src = (st.st_mtime - ts(job["start"])) / n, "since job start (incl. loading)"
            elif base.get("s_per_epoch"):
                spe, spe_src = base["s_per_epoch"], "baseline runs"
            else:
                per_seed_h = num((est.get("per seed training (h)") or {}).get(ds))
                if per_seed_h:
                    spe, spe_src = per_seed_h * 3600 / max_ep, "estimates.json"
            overhead = (base["job_s"] - base["train_s"]) if base.get("job_s") and base.get("train_s") else 0
            eta_seed = 0 if finished else ((max_ep - n) * spe + overhead if spe else None)
            last = rows[-1] if rows else {}
            best_auroc = max((r for r in rows if num(r.get("val_auroc")) is not None),
                             key=lambda r: r["val_auroc"], default=None)
            bl_rows = []
            if cfg.get("baseline_hash") and seed is not None:
                bl_rows = read_jsonl(ROOT / cfg.get("baseline_results", "runs/results") / "raw"
                                     / f"{ds}_seed{seed}_{cfg['baseline_hash']}" / "training_log.jsonl")
            runs.append({
                "run": f.name, "dataset": ds, "seed": seed, "hash": h, "config": cfg.get("stem"), "status": status,
                "epochs": n, "max_epochs": max_ep, "rows": rows, "last": last,
                "best_val_auroc_epoch": best_auroc.get("epoch") if best_auroc else None,
                "log_mtime": iso(st.st_mtime) if st else None, "idle_s": now - st.st_mtime if st else None,
                "s_per_epoch": spe, "s_per_epoch_source": spe_src, "eta_seed_s": eta_seed,
                "job": job["job"] if job else None, "job_start": job.get("start") if job else None,
                "metrics_best_epoch": metrics.get("best_epoch") if finished else None,
                "selection": {k: v.get("epoch") for k, v in (metrics.get("selection") or {}).items()
                              if isinstance(v, dict)} if finished else None,
                "runtime_seconds": (manifest or {}).get("runtime_seconds"),
                "train_seconds": (manifest or {}).get("train_seconds"),
                "peak_memory": (manifest or {}).get("peak_memory"),
                "baseline_rows": bl_rows, "baseline": base,
                "files": sorted(p.name for p in f.iterdir()),
            })
        except Exception as exc:
            runs.append({"run": f.name, "error": repr(exc)})
    per_config = []
    for c in arms():
        cr = [r for r in runs if r.get("hash") == c["hash"]]
        seeds = c.get("seeds") or []
        done = [r for r in cr if r.get("status") == "finished"]
        cur = [r for r in cr if r.get("status") != "finished"]
        base = baseline_stats(c["dataset"], c.get("baseline_hash", ""), c.get("baseline_results", "runs/results"))
        per_seed = None
        if done and all(num(r.get("runtime_seconds")) for r in done):
            per_seed = statistics.mean(float(r["runtime_seconds"]) for r in done)
            if base.get("job_s") and base.get("runtime_s"):
                per_seed += base["job_s"] - base["runtime_s"]
        elif base.get("job_s"):
            per_seed = base["job_s"]
        elif cur and cur[0].get("s_per_epoch"):
            per_seed = cur[0]["s_per_epoch"] * (c.get("max_epochs") or 10)
        remaining_cur = sum(r["eta_seed_s"] or 0 for r in cur if r.get("eta_seed_s") is not None)
        not_started = max(len(seeds) - len(done) - len(cur), 0)
        eta_all = remaining_cur + not_started * per_seed if per_seed is not None else None
        per_config.append({"config": c["stem"], "dataset": c["dataset"], "hash": c["hash"], "seeds": seeds,
                           "finished": len(done), "in_progress": len(cur), "not_started": not_started,
                           "per_seed_s": per_seed, "eta_all_s": eta_all if len(done) < len(seeds) else 0,
                           "baseline": base})
    return {"runs": runs, "per_config": per_config, "folder": rel(raw)}


def get_results() -> dict:
    metrics_keys = ("macro_f1", "auroc", "auprc")
    out = []
    for c in arms():
        ds = c["dataset"]
        rows = []
        for s in c.get("seeds") or []:
            am = read_json(ROOT / c["results"] / "raw" / f"{ds}_seed{s}_{c['hash']}" / "metrics.json")
            bm = read_json(ROOT / c.get("baseline_results", "runs/results") / "raw"
                           / f"{ds}_seed{s}_{c.get('baseline_hash')}" / "metrics.json") if c.get("baseline_hash") \
                else None
            at = (am or {}).get("test") if isinstance(am, dict) else None
            bt = (bm or {}).get("test") if isinstance(bm, dict) else None
            row = {"seed": s, "arm": {k: num((at or {}).get(k)) for k in metrics_keys} if at else None,
                   "baseline": {k: num((bt or {}).get(k)) for k in metrics_keys} if bt else None,
                   "arm_best_epoch": (am or {}).get("best_epoch") if isinstance(am, dict) else None,
                   "baseline_best_epoch": (bm or {}).get("best_epoch") if isinstance(bm, dict) else None}
            if row["arm"] and row["baseline"]:
                row["delta"] = {k: (row["arm"][k] - row["baseline"][k]) if row["arm"][k] is not None and
                                row["baseline"][k] is not None else None for k in metrics_keys}
            rows.append(row)

        def agg(vals):
            vals = [v for v in vals if v is not None]
            return {"mean": statistics.mean(vals), "std": statistics.pstdev(vals), "n": len(vals)} if vals else None

        summary = {}
        for k in metrics_keys:
            summary[k] = {
                "arm": agg([r["arm"][k] for r in rows if r["arm"]]),
                "baseline_paired": agg([r["baseline"][k] for r in rows if r["arm"] and r["baseline"]]),
                "baseline_all": agg([r["baseline"][k] for r in rows if r["baseline"]]),
                "delta": agg([r["delta"][k] for r in rows if r.get("delta")]),
            }
        out.append({"config": c["stem"], "dataset": ds, "hash": c["hash"], "baseline_hash": c.get("baseline_hash"),
                    "rows": rows, "summary": summary, "n_arm": sum(1 for r in rows if r["arm"])})
    compare = []
    try:
        for p in sorted((gnn_dir() / "runs" / "compare").glob("*.md")):
            compare.append({"path": rel(p), "text": read_text(p)})
    except Exception:
        pass
    return {"arms": out, "compare": compare}


def sample_gpu() -> dict | None:
    out = run_cmd(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
                   "power.draw", "--format=csv,noheader,nounits"], timeout=10)
    if out is None:
        return None
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 6:
            gpus.append({"index": p[0], "name": p[1], "util": num(p[2]), "mem_used": num(p[3]),
                         "mem_total": num(p[4]), "temp": num(p[5]), "power_w": num(p[6]) if len(p) > 6 else None})
    apps = []
    out2 = run_cmd(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    for line in (out2 or "").strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            apps.append({"pid": p[0], "name": p[1], "mem_mib": num(p[2])})
    return {"t": time.time(), "gpus": gpus, "apps": apps}


def sampler() -> None:
    while True:
        try:
            s = sample_gpu()
            if s:
                with LOCK:
                    GPU_HISTORY.append(s)
            get_summary_stores()      # records line growth even when nobody is viewing
        except Exception:
            pass
        time.sleep(SAMPLE_SECONDS)


def get_system(procs: list) -> dict:
    with LOCK:
        hist = list(GPU_HISTORY)
    latest = hist[-1] if hist else None
    if latest is None or time.time() - latest["t"] > 3 * SAMPLE_SECONDS:
        latest = sample_gpu()
    load = None
    try:
        la = Path("/proc/loadavg").read_text().split()
        load = {"1m": float(la[0]), "5m": float(la[1]), "15m": float(la[2]), "procs": la[3], "cpus": os.cpu_count()}
    except Exception:
        pass
    mem = None
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            info[k] = int(v.split()[0]) * 1024
        mem = {"total": info.get("MemTotal"), "available": info.get("MemAvailable"),
               "used": info.get("MemTotal", 0) - info.get("MemAvailable", 0),
               "swap_used": info.get("SwapTotal", 0) - info.get("SwapFree", 0)}
    except Exception:
        pass
    disk = None
    try:
        u = shutil.disk_usage("/workspace")
        disk = {"total": u.total, "used": u.used, "free": u.free}
    except Exception:
        pass
    tmux = run_cmd(["tmux", "ls"])
    series = [{"t": s["t"], "util": s["gpus"][0]["util"] if s["gpus"] else None,
               "mem_pct": (100 * s["gpus"][0]["mem_used"] / s["gpus"][0]["mem_total"])
               if s["gpus"] and s["gpus"][0]["mem_total"] else None,
               "mem_used": s["gpus"][0]["mem_used"] if s["gpus"] else None} for s in hist]
    return {"gpu": latest, "gpu_history": series, "load": load, "mem": mem, "disk": disk,
            "tmux": tmux.strip().splitlines() if tmux else None, "processes": procs,
            "dashboard_uptime_s": time.time() - STARTED}


def get_logs(queue: dict | None) -> dict:
    running, other = [], []
    qlogs = gnn_dir() / "runs" / "queue" / "logs"
    run_names = {j["job"] for j in (queue or {}).get("jobs", []) if j["state"] == "running"}
    try:
        for p in sorted(qlogs.glob("*.log"), key=lambda x: -getattr(stat(x), "st_mtime", 0)):
            st = stat(p)
            entry = {"path": rel(p), "job": p.stem, "bytes": st.st_size if st else None,
                     "mtime": iso(st.st_mtime) if st else None, "tail": tail_lines(p, 40)}
            (running if p.stem in run_names else other).append(entry)
    except Exception:
        pass
    exp = []
    try:
        for p in sorted((gnn_dir() / "logs").iterdir()):
            if p.is_file():
                st = stat(p)
                exp.append({"path": rel(p), "bytes": st.st_size if st else None,
                            "mtime": iso(st.st_mtime) if st else None, "tail": tail_lines(p, 40)})
    except Exception:
        pass
    return {"running": running, "queue_other": other, "experiment": exp}


def get_files(stores: dict) -> dict:
    out, truncated = [], False
    roots = [gnn_dir() / "runs"]
    cache = ROOT / "runs" / "cache"
    roots += [cache / "trims" / t["folder"] for t in stores.get("trims", [])]
    roots += [cache / "metapath_summaries" / m["folder"] for m in stores.get("metapath", []) if m.get("folder")]
    roots += [cache / "node_summaries" / m["folder"] for m in stores.get("node", []) if m.get("folder")]
    limit = 600
    for r in roots:
        try:
            for dirpath, dirnames, filenames in os.walk(r):
                dirnames.sort()
                for fn in sorted(filenames):
                    if len(out) >= limit:
                        truncated = True
                        break
                    p = Path(dirpath) / fn
                    st = stat(p)
                    if st:
                        out.append({"path": rel(p), "bytes": st.st_size, "mtime": iso(st.st_mtime), "m": st.st_mtime})
        except Exception:
            continue
    return {"files": out, "truncated": truncated, "limit": limit,
            "total_bytes": sum(f["bytes"] for f in out)}


def stage_estimates(ds: str) -> dict:
    est = get_estimates()
    if not isinstance(est, dict):
        return {}
    block = next((v for k, v in est.items() if k.startswith(ds) and "stage" in k and isinstance(v, dict)), {})
    out = {}
    for k, v in block.items():
        h = num(v)
        if h is None:
            continue
        kl = k.lower()
        for s, kw in (("trim", "trim"), ("node", "node"), ("meta", "metapath"), ("prompts", "prompt"),
                      ("train", "train"), ("total", "total")):
            if kw in kl:
                out[s] = h * 3600
    return out


def header_info(st: dict) -> dict:
    p = st.get("progress") or {}
    phases = [ph for ph in (p.get("phases") or []) if isinstance(ph, dict)]
    cur = next((ph for ph in phases if ph.get("status") == "running"), None) or \
        next((ph for ph in phases if ph.get("status") not in ("done", "skipped")), None)
    q = st.get("queue") or {}
    running = [j for j in q.get("jobs", []) if j["state"] == "running"]
    eta_s, parts = 0.0, []
    for stem in q.get("stems", []):
        if not stem["in_queue"]:
            continue
        cfg = next((c for c in arms() if c["stem"] == stem["stem"]), {})
        ds = cfg.get("dataset", "")
        se = stage_estimates(ds)
        for s in ("trim", "node", "meta", "prompts"):
            sg = stem["stages"][s]
            if sg["done"] >= sg["total"]:
                continue
            live = None
            if s == "meta":
                live = next((m.get("eta_s") for m in st["summaries"]["metapath"] if m.get("dataset") == ds
                             and m.get("eta_s") is not None), None)
            val = live if live is not None else se.get(s, 0)
            eta_s += val
            parts.append(f"{s} {'live' if live is not None else 'est.'}")
        tc = next((c for c in st["training"]["per_config"] if c["config"] == stem["stem"]), None)
        if tc and tc["eta_all_s"] is not None:
            eta_s += tc["eta_all_s"]
            parts.append("train live" if tc["finished"] or tc["in_progress"] else "train est.")
        elif se.get("train"):
            eta_s += se["train"]
            parts.append("train est.")
    gpu = (st["system"].get("gpu") or {}).get("gpus") or []
    return {"phase": cur, "running": [j["job"] for j in running], "gpu": gpu[0] if gpu else None,
            "eta_s": eta_s if parts else None, "eta_basis": ", ".join(parts),
            "queue_alive": (q.get("meta") or {}).get("process_alive")}


def build_state() -> dict:
    t0 = time.time()
    st: dict = {"generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "root": str(ROOT),
                "imports": {"our_pipeline": _pkg_load_config is not None, "yaml": yaml is not None}}

    def guard(name, fn, default=None):
        try:
            st[name] = fn()
        except Exception as exc:
            st[name] = default
            st.setdefault("errors", {})[name] = repr(exc)

    procs = []
    try:
        procs = get_processes()
    except Exception:
        pass
    guard("progress", get_progress)
    guard("estimates", get_estimates)
    guard("decisions", get_decisions, {"files": []})
    guard("configs", get_configs, [])
    guard("embeddings", get_embeddings, [])
    guard("diagnostics", get_diagnostics, {"json": [], "md": []})
    guard("queue", lambda: get_queue(procs))
    guard("summaries", get_summary_stores, {"trims": [], "metapath": [], "node": []})
    guard("training", lambda: get_training(st.get("queue")), {"runs": [], "per_config": []})
    guard("results", get_results, {"arms": [], "compare": []})
    guard("system", lambda: get_system(procs), {})
    guard("logs", lambda: get_logs(st.get("queue")), {"running": [], "queue_other": [], "experiment": []})
    guard("files", lambda: get_files(st.get("summaries") or {}), {"files": []})
    guard("header", lambda: header_info(st), {})
    st["build_seconds"] = round(time.time() - t0, 3)
    return st


# ---------------------------------------------------------------- rendering
e = html.escape


def fmt(v, digits: int = 4) -> str:
    if isinstance(v, bool):
        return e(str(v))
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    if isinstance(v, (dict, list)):
        return f"<code>{e(json.dumps(v, default=str))}</code>"
    return e("" if v is None else str(v))


def f2(v) -> str:
    return "" if num(v) is None else f"{float(v):.2f}"


def pct(v, digits: int = 1) -> str:
    return "" if num(v) is None else f"{100 * float(v):.{digits}f}%"


def badge(status) -> str:
    s = str(status or "pending")
    cls = re.sub(r"[^a-z_-]", "", s.lower())
    return f'<span class="b b-{e(cls)}">{e(s)}</span>'


def table(headers: list, rows: list, row_cls: list | None = None, cls: str = "") -> str:
    h = "".join(f"<th>{e(str(x))}</th>" for x in headers)
    body = []
    for i, r in enumerate(rows):
        rc = f' class="{row_cls[i]}"' if row_cls and row_cls[i] else ""
        body.append(f"<tr{rc}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def kv_table(d: dict) -> str:
    return table(["key", "value"], [[e(str(k)), fmt(v)] for k, v in (d or {}).items()], cls="kv")


def bar(done, total, label: str = "", extra: str = "") -> str:
    try:
        frac = max(0.0, min(1.0, float(done) / float(total))) if total else 0.0
    except Exception:
        frac = 0.0
    full = " full" if frac >= 1 else ""
    return (f'<div class="bar"><div class="bar-l"><span>{e(label)}</span><span>{e(str(done))} / {e(str(total))}'
            f' ({frac * 100:.0f}%){e(extra)}</span></div><div class="bar-t"><div class="bar-f{full}" '
            f'style="width:{frac * 100:.1f}%"></div></div></div>')


def details(key: str, summary: str, body: str, open_: bool = False) -> str:
    return (f'<details data-k="{e(key)}"{" open" if open_ else ""}><summary>{summary}</summary>'
            f'<div class="dbody">{body}</div></details>')


NS = '<p class="ns">not started</p>'


def ns(msg: str = "not started") -> str:
    return f'<p class="ns">{e(msg)}</p>'


def sparkline(series: list, w: int = 300, h: int = 80, title: str = "", fmt_v=f2, x_labels: list | None = None,
              ymin=None, ymax=None) -> str:
    """series: [{"name", "values", "cls", "dash"}]. One y-axis; native tooltips on points."""
    vals = [v for s in series for v in s["values"] if num(v) is not None]
    if not vals:
        return ""
    lo = min(vals) if ymin is None else ymin
    hi = max(vals) if ymax is None else ymax
    if hi - lo < 1e-12:
        hi, lo = hi + 0.5, lo - 0.5
    n = max(len(s["values"]) for s in series)
    pl, pr, pt, pb = 34, 6, 8, 16
    iw, ih = w - pl - pr, h - pt - pb

    def xy(i, v):
        x = pl + (iw * i / (n - 1) if n > 1 else iw / 2)
        y = pt + ih * (1 - (float(v) - lo) / (hi - lo))
        return x, y

    parts = [f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
             f'aria-label="{e(title)}">']
    for frac in (0, 0.5, 1):
        y = pt + ih * (1 - frac)
        parts.append(f'<line x1="{pl}" x2="{w - pr}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="{pl - 4}" y="{y + 3:.1f}" class="ax" text-anchor="end">'
                     f'{e(fmt_v(lo + (hi - lo) * frac))}</text>')
    labels = x_labels or [str(i + 1) for i in range(n)]
    for i in sorted({0, n - 1}):
        x, _ = xy(i, lo)
        parts.append(f'<text x="{x:.1f}" y="{h - 3}" class="ax" text-anchor="middle">{e(labels[i])}</text>')
    for s in series:
        pts = [(i, v) for i, v in enumerate(s["values"]) if num(v) is not None]
        if not pts:
            continue
        d = " ".join(f"{xy(i, v)[0]:.1f},{xy(i, v)[1]:.1f}" for i, v in pts)
        dash = ' stroke-dasharray="4 3"' if s.get("dash") else ""
        parts.append(f'<polyline points="{d}" class="ln {s.get("cls", "s1")}"{dash}/>')
        step = max(1, len(pts) // 60)
        for i, v in pts[::step] + ([pts[-1]] if (len(pts) - 1) % step else []):
            x, y = xy(i, v)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" class="pt {s.get("cls", "s1")}">'
                         f'<title>{e(s["name"])} @ {e(labels[i])}: {e(fmt_v(v))}</title></circle>')
    parts.append("</svg>")
    legend = ""
    if len(series) > 1:
        legend = '<div class="lg">' + "".join(
            f'<span><i class="sw {s.get("cls", "s1")}{" dash" if s.get("dash") else ""}"></i>{e(s["name"])}</span>'
            for s in series) + "</div>"
    return f'<figure class="chart"><figcaption>{e(title)}</figcaption>{"".join(parts)}{legend}</figure>'


# ---- markdown-lite
def md_inline(s: str) -> str:
    out = []
    s = s.replace("\\|", "|")
    for part in re.split(r"(`[^`]*`)", s):
        if len(part) >= 2 and part.startswith("`") and part.endswith("`"):
            out.append(f"<code>{e(part[1:-1])}</code>")
        else:
            t = e(part)
            t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
            t = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"<i>\1</i>", t)
            out.append(t)
    return "".join(out)


def md_cells(line: str) -> list:
    line = line.strip().replace("\\|", "\x00")
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip().replace("\x00", "|") for c in line.split("|")]


def md_render(text: str | None) -> str:
    if not text:
        return ns("empty")
    lines = text.splitlines()
    out, para, i = [], [], 0

    def flush():
        if para:
            out.append("<p>" + md_inline(" ".join(para)) + "</p>")
            para.clear()

    sep = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            flush()
            i += 1
        elif s.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append(f"<pre>{e(chr(10).join(lines[i + 1:j]))}</pre>")
            i = j + 1
        elif re.match(r"^#{1,6}\s", s):
            flush()
            lvl = len(s) - len(s.lstrip("#"))
            tag = f"h{min(6, lvl + 2)}"
            out.append(f'<{tag} class="md-h">{md_inline(s[lvl:].strip())}</{tag}>')
            i += 1
        elif s.startswith("|") and i + 1 < len(lines) and sep.match(lines[i + 1]):
            flush()
            head = md_cells(s)
            j, body = i + 2, []
            while j < len(lines) and lines[j].strip().startswith("|"):
                body.append(md_cells(lines[j]))
                j += 1
            out.append(table(head, [[md_inline(c) for c in r] for r in body], cls="md"))
            i = j
        elif re.match(r"^([-*+]|\d+\.)\s", s):
            flush()
            ordered = bool(re.match(r"^\d+\.", s))
            items = []
            while i < len(lines) and re.match(r"^\s*([-*+]|\d+\.)\s", lines[i]):
                items.append(re.sub(r"^\s*([-*+]|\d+\.)\s+", "", lines[i]))
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() and \
                        not re.match(r"^\s*([-*+]|\d+\.)\s", lines[i]):
                    items[-1] += " " + lines[i].strip()
                    i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{md_inline(x)}</li>" for x in items) + f"</{tag}>")
        else:
            para.append(s)
            i += 1
    flush()
    return f'<div class="md">{"".join(out)}</div>'


# ---- sections
def sec_header(st: dict) -> str:
    h = st.get("header") or {}
    ph = h.get("phase")
    phase = (f'{e(str(ph.get("id", "")))} &middot; {e(str(ph.get("name", "")))} {badge(ph.get("status"))}'
             if ph else '<span class="ns">all phases done</span>')
    run = h.get("running") or []
    runs = ", ".join(f"<code>{e(r)}</code>" for r in run) if run else '<span class="ns">none</span>'
    q = st.get("queue") or {}
    qa = ""
    if q:
        qa = badge("queue alive" if h.get("queue_alive") else ("queue ended" if q["meta"].get("queue_end")
                                                                else "queue stopped"))
    g = h.get("gpu")
    if g:
        mem = f"{(g['mem_used'] or 0) / 1024:.1f} / {(g['mem_total'] or 0) / 1024:.0f} GiB"
        gpu = (f'<b>{fmt(g["util"], 0) or "?"}%</b> util &middot; {e(mem)} &middot; '
               f'{e(str(int(g["temp"])) if g["temp"] is not None else "?")}&deg;C')
    else:
        gpu = '<span class="ns">unavailable</span>'
    stopped = "" if h.get("queue_alive") else "; queue not running: remaining work, not a live countdown"
    eta = (f'<b>{e(fmt_dur(h["eta_s"]))}</b> <span class="muted">({e(h.get("eta_basis", "") + stopped)})</span>'
           if h.get("eta_s") is not None else '<span class="ns">n/a (queue not started)</span>')
    tiles = [("Current phase", phase), ("Running jobs", f"{runs} {qa}"), ("GPU", gpu),
             ("ETA current phase", eta)]
    return '<div class="strip">' + "".join(f'<div class="tile"><div class="tl">{e(a)}</div><div class="tv">{b}</div>'
                                           f"</div>" for a, b in tiles) + "</div>"


def sec_overview(st: dict) -> str:
    out = ['<h2 id="overview">1. Overview</h2>']
    p = st.get("progress")
    if p:
        phases = [ph for ph in (p.get("phases") or []) if isinstance(ph, dict)]
        out.append(f'<p class="muted">progress.json updated: {e(str(p.get("updated", "?")))}</p>')
        out.append('<div class="phases">' + "".join(
            f'<div class="ph ph-{e(re.sub(r"[^a-z_]", "", str(ph.get("status", "")).lower()))}">'
            f'<div><b>{e(str(ph.get("id", "")))}</b> {badge(ph.get("status"))}</div>'
            f'<div class="phn">{e(str(ph.get("name", "")))}</div><div class="muted sm">{e(str(ph.get("note", "")))}'
            f"</div></div>" for ph in phases) + "</div>")
        notes = p.get("notes") or []
        if notes:
            out.append("<h3>Notes (newest first)</h3><ul>" + "".join(f"<li>{md_inline(str(n))}</li>" for n in notes)
                       + "</ul>")
        extra = {k: v for k, v in p.items() if k not in ("phases", "notes", "updated")}
        if extra:
            out.append("<h3>Other progress fields</h3>" + kv_table(extra))
    else:
        out.append(ns("progress.json not found"))
    arm_list = [c for c in st.get("configs") or [] if c.get("hash")]
    if arm_list:
        out.append("<h3>Scope</h3>" + table(
            ["arm config", "dataset", "selection", "seeds", "epochs", "arm hash", "baseline", "baseline hash"],
            [[f"<code>{e(c['path'])}</code>", e(c["dataset"]), e(str(c.get("selection"))),
              e(", ".join(map(str, c.get("seeds") or []))), e(str(c.get("max_epochs"))), f"<code>{e(c['hash'])}</code>",
              f"<code>{e(c.get('baseline_path', ''))}</code>", f"<code>{e(c.get('baseline_hash', '?'))}</code>"]
             for c in arm_list]))
        out.append('<p class="muted sm">"Ours" = baseline pipeline <code>configs/ours/&lt;ds&gt;_k1.yaml</code>; the '
                   "arm replaces diffusion neighbour ranking with cosine on PMP GNN embeddings. Instagram runs first; "
                   "Reddit is decided after its results (G3).</p>")
    est = st.get("estimates")
    if isinstance(est, dict):
        rows = []
        for k, v in est.items():
            rows.append([e(k), "<br>".join(f"{e(str(a))}: <b>{fmt(b)}</b>" for a, b in v.items())
                         if isinstance(v, dict) else fmt(v)])
        out.append("<h3>Estimates (estimates.json)</h3>" + table(["key", "value"], rows, cls="kv"))
    for f in (st.get("decisions") or {}).get("files", []):
        out.append(details(f"dec:{f['path']}", f"Decisions log &middot; <code>{e(f['path'])}</code>",
                           md_render(f.get("text")), open_=True))
    return "".join(out)


def sec_configs(st: dict) -> str:
    out = ['<h2 id="configs">2. Configs</h2>']
    cfgs = st.get("configs") or []
    if not cfgs:
        return "".join(out) + ns("no experiment configs found")
    if not (st.get("imports") or {}).get("our_pipeline"):
        out.append('<p class="warn">our_pipeline not importable: hashes use the fallback loader.</p>')
    for c in cfgs:
        body = f"<pre>{e(c.get('text') or '')}</pre>"
        if c.get("overlay"):
            out.append(details(f"cfg:{c['stem']}", f"<code>{e(c['path'])}</code> (overlay)", body))
            continue
        if c.get("error"):
            out.append(details(f"cfg:{c['stem']}", f"<code>{e(c.get('path', ''))}</code>",
                               body + f'<p class="warn">{e(c["error"])}</p>', open_=True))
            continue
        info = (f'<p>experiment hash <code>{e(c["hash"])}</code> &middot; baseline '
                f'<code>{e(c.get("baseline_path", ""))}</code> hash <code>{e(str(c.get("baseline_hash")))}</code> '
                f'&middot; results <code>{e(str(c.get("results")))}/raw/{e(c["dataset"])}_seed&lt;s&gt;_'
                f'{e(c["hash"])}</code></p>')
        if c.get("baseline_error"):
            diff = f'<p class="warn">baseline: {e(c["baseline_error"])}</p>'
        else:
            diff = "<h4>Effective differences vs baseline (after load_config)</h4>" + table(
                ["dotted key", "baseline", "experiment"],
                [[f"<code>{e(d['key'])}</code>", fmt(d["baseline"]), fmt(d["experiment"])] for d in c.get("diff", [])])
        out.append(details(f"cfg:{c['stem']}", f"<code>{e(c['path'])}</code> &middot; {e(c['dataset'])} "
                           f"&middot; {len(c.get('diff', []))} differing keys", info + body + diff, open_=True))
    return "".join(out)


def sec_embeddings(st: dict) -> str:
    out = ['<h2 id="embeddings">3. GNN embeddings</h2>']
    embs = st.get("embeddings") or []
    if not embs:
        return "".join(out) + NS
    summary_rows = []
    for em in embs:
        if em.get("error"):
            continue
        ch = next((g for g in em["grid"] if g["name"] == em.get("chosen")), {})
        used = ", ".join(em.get("used_by_trims") or []) or "-"
        summary_rows.append([f"<b>{e(em['folder'])}</b>", e(str(em.get("chosen"))), f2(ch.get("val_auroc")),
                             f2(ch.get("val_macro_f1")), e(str(ch.get("best_epoch"))),
                             e(str((em.get("embedding") or {}).get("shape"))),
                             f"<code>{e(str((em.get('embedding') or {}).get('sha256_16')))}</code>",
                             e(fmt_dur(em.get("runtime_seconds"))), e(used)])
    out.append(table(["variant", "chosen", "val AUROC", "val Macro-F1", "best epoch", "shape", "sha256_16",
                      "runtime", "used by trims"], summary_rows))
    hp_cols = ("hid_dim", "dropout", "lr", "weighted_loss", "norm_feat", "epochs", "patience", "mlp_activation",
               "agg", "proj")
    for em in embs:
        if em.get("error"):
            out.append(f"<h3>{e(em['folder'])}</h3>" + ns(em["error"]))
            continue
        body = []
        rows, rcls = [], []
        for g in em["grid"]:
            hp = g["hp"]
            curve = sparkline([{"name": "val AUROC", "values": g["val_auroc_curve"] or [], "cls": "s1"}],
                              w=140, h=36, title="") if g.get("val_auroc_curve") else ""
            rows.append([f"<b>{e(str(g['name']))}</b>"] + [fmt(first(hp, k, k.replace("epochs", "max_epochs")))
                                                         for k in hp_cols]
                        + [f2(g["val_auroc"]), f2(g["val_macro_f1"]), fmt(g["val_threshold"], 3),
                           e(str(g["best_epoch"])), e(str(g["epochs_run"])), f2(g["seconds"]), curve])
            rcls.append("hl" if g["name"] == em.get("chosen") else "")
        body.append("<h4>Grid (chosen on validation AUROC; highlighted = chosen)</h4>" + table(
            ["config", *hp_cols, "val AUROC", "val Macro-F1", "threshold", "best epoch (0-based)", "epochs run",
             "seconds", "val AUROC curve"], rows, rcls))
        ch = next((g for g in em["grid"] if g["name"] == em.get("chosen")), None)
        if ch and (ch.get("val_auroc_curve") or ch.get("train_loss_curve")):
            charts = ""
            if ch.get("val_auroc_curve"):
                charts += sparkline([{"name": "val AUROC", "values": ch["val_auroc_curve"], "cls": "s1"}],
                                    w=420, h=110, title=f"chosen {ch['name']}: val AUROC per epoch",
                                    x_labels=[str(i) for i in range(len(ch["val_auroc_curve"]))])
            if ch.get("train_loss_curve"):
                charts += sparkline([{"name": "train loss", "values": ch["train_loss_curve"], "cls": "s2"}],
                                    w=420, h=110, title=f"chosen {ch['name']}: train loss per epoch",
                                    fmt_v=lambda v: f"{float(v):.3f}",
                                    x_labels=[str(i) for i in range(len(ch["train_loss_curve"]))])
            body.append(f'<div class="charts">{charts}</div>')
        lc = em.get("label_counts")
        if isinstance(lc, dict):
            body.append("<h4>Label counts</h4>" + table(
                ["split", "positives (fraud)", "negatives", "total", "fraud share"],
                [[e(k), fmt(first(v, "positives")), fmt(first(v, "negatives")), fmt(first(v, "total")),
                  pct((num(first(v, "positives")) or 0) / num(first(v, "total"))) if num(first(v, "total")) else ""]
                 for k, v in lc.items() if isinstance(v, dict)]))
        gc = em.get("graph_checks")
        if isinstance(gc, dict):
            grows = []
            for name, v in list((gc.get("relations") or {}).items()) + [("merged", gc.get("merged"))]:
                if isinstance(v, dict):
                    grows.append([e(name)] + [fmt(v.get(k)) for k in ("edges", "distinct_edges", "duplicate_edges",
                                                                      "symmetric", "self_loops", "isolated_nodes")])
            body.append(f"<h4>Graph checks (nodes: {fmt(gc.get('num_nodes'))})</h4>" + table(
                ["relation", "edges", "distinct", "duplicates", "symmetric", "self loops", "isolated nodes"], grows))
        info = {"dataset": em.get("dataset"), "model": em.get("model"), "layers": em.get("layers"),
                "seed": em.get("seed"), "device": em.get("device"), "relation_mode": em.get("relation_mode"),
                "config_file": em.get("config_file"), "selection rule": em.get("selection_rule"),
                "runtime": fmt_dur(em.get("runtime_seconds")), "git_commit": em.get("git_commit"),
                "test_metrics": em.get("test_metrics")}
        emb = em.get("embedding") or {}
        info.update({f"embedding.{k}": v for k, v in emb.items()})
        info["embedding.bytes"] = fmt_bytes(emb.get("bytes"))
        info["used by trims folders"] = em.get("used_by_trims")
        body.append("<h4>Embedding and run</h4>" + kv_table(info))
        if isinstance(em.get("split"), dict):
            body.append(details(f"emb:{em['folder']}:split", "split", kv_table(flatten(em["split"]))))
        if isinstance(em.get("features"), dict):
            body.append(details(f"emb:{em['folder']}:feat", "input features", kv_table(em["features"])))
        if isinstance(em.get("versions"), dict):
            body.append(details(f"emb:{em['folder']}:ver", "versions", kv_table(em["versions"])))
        out.append(details(f"emb:{em['folder']}", f"<b>{e(em['folder'])}</b> &middot; chosen {e(str(em.get('chosen')))}"
                           f" &middot; <code>{e(em.get('path', ''))}</code>", "".join(body),
                           open_=em["folder"].endswith("_l1")))
    return "".join(out)


def _agree_cells(block) -> list:
    if not isinstance(block, dict):
        return ["", "", "", ""]
    cells = []
    for grp in ("all", "fraud", "benign"):
        g = block.get(grp) if isinstance(block.get(grp), dict) else {}
        cells.append(f"{pct(g.get('agreement'))} <span class='muted'>n={fmt(g.get('neighbours_known'))}</span>"
                     if g else "")
    cov = " / ".join(pct((block.get(grp) or {}).get("coverage")) for grp in ("all", "fraud", "benign")
                     if isinstance(block.get(grp), dict))
    return cells + [cov]


def sec_diagnostics(st: dict) -> str:
    out = ['<h2 id="diagnostics">4. Diagnostics (Phase C)</h2>']
    dg = st.get("diagnostics") or {}
    if not dg.get("json") and not dg.get("md"):
        return "".join(out) + NS
    out.append('<p class="muted sm">Agreement = share of selected neighbours with the target\'s label, over neighbours '
               "with a train/val label, on validation targets with more than M candidates (n = neighbours_known). "
               "Coverage = share of selected neighbours with such a label (all / fraud / benign). Test labels are never "
               "read.</p>")
    for f in dg.get("json", []):
        data = f.get("data")
        out.append(f'<p class="muted">source <code>{e(f["path"])}</code></p>')
        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) else []
        if not data:
            out.append(ns("unreadable"))
        for d in data:
            if not isinstance(d, dict) or not isinstance(d.get("metapaths"), dict):
                out.append(f"<pre>{e(json.dumps(d, indent=1, default=str))}</pre>")
                continue
            out.append(f"<h3>{e(str(d.get('dataset')))} &middot; M = {fmt(d.get('M'))}, K = {fmt(d.get('K'))} "
                       f"&middot; baseline trims <code>{e(str(d.get('baseline_trims')))}</code></h3>")
            for mpn, mp in d["metapaths"].items():
                if not isinstance(mp, dict):
                    continue
                vt = mp.get("val_targets_with_headroom") or {}
                out.append(f"<h4>{e(mpn)}</h4>" + table(
                    ["targets", "headroom (> M candidates)", "no candidates", "median candidates",
                     "val targets with headroom (all / fraud / benign)"],
                    [[fmt(mp.get("targets")), pct(mp.get("headroom_share")), pct(mp.get("zero_candidates_share")),
                      fmt(mp.get("median_candidates")),
                      " / ".join(fmt(vt.get(k)) for k in ("all", "fraud", "benign")) if isinstance(vt, dict) else
                      fmt(vt)]]))
                rows = []
                for label, key in (("Diffusion (Ours K=1)", "diffusion"), ("Random M", "random")):
                    if key in mp:
                        rows.append([e(label), "", "", "", "", ""] + _agree_cells(mp[key]))
                for vn, v in (mp.get("variants") or {}).items():
                    if not isinstance(v, dict):
                        continue
                    rows.append([f"<b>{e(vn)}</b>", f2(v.get("val_auroc")), f2(v.get("val_macro_f1")),
                                 e(str(v.get("chosen_config", ""))), fmt(v.get("overlap_jaccard_mean"), 3),
                                 pct(v.get("identical_set_share"))] + _agree_cells(v.get("agreement")))
                out.append(table(["selection", "GNN val AUROC", "GNN val Macro-F1", "chosen config",
                                  "Jaccard vs diffusion", "identical sets", "agreement all", "agreement fraud",
                                  "agreement benign", "coverage all / fraud / benign"], rows))
    for f in dg.get("md", []):
        out.append(details(f"diag:{f['path']}", f"Report <code>{e(f['path'])}</code>", md_render(f.get("text"))))
    return "".join(out)


def sec_pipeline(st: dict) -> str:
    out = ['<h2 id="pipeline">5. Pipeline progress (Phase D)</h2>']
    q = st.get("queue")
    if not q:
        out.append(ns("queue not started (experiments/gnn/runs/queue/status.jsonl absent)"))
    else:
        m = q["meta"]
        cmd = m.get("cmdline") or {}
        out.append(f'<p class="muted">queue_start {fmt(m.get("queue_start"))} &middot; queue_end '
                   f'{fmt(m.get("queue_end")) or "-"} &middot; declared jobs {fmt(m.get("jobs_declared"))} &middot; '
                   f'shards {fmt(q.get("shards"))} &middot; run_queue.py '
                   f'{"running (pid " + e(str(cmd.get("pid"))) + ")" if m.get("process_alive") else "not running"}'
                   f'</p>')
        if cmd.get("args"):
            out.append(f"<p class='sm'><code>{e(cmd['args'])}</code></p>")
        if m.get("queue_end"):
            out.append(f"<p>queue ended: unfinished {fmt(m.get('unfinished'))}, failed {fmt(m.get('failed'))}</p>")
        for s in q["stems"]:
            bars = "".join(bar(s["stages"][k]["done"], s["stages"][k]["total"], STAGE_LABEL[k],
                               (f" · {s['stages'][k]['running']} running" if s["stages"][k]["running"] else "")
                               + (f" · {s['stages'][k]['failed_attempts']} failed attempt(s)"
                                  if s["stages"][k]["failed_attempts"] else "")
                               + (f" · {s['stages'][k]['gave_up']} gave up" if s["stages"][k]["gave_up"] else ""))
                           for k in STAGES)
            note = "" if s["in_queue"] else ' <span class="ns">(not in queue)</span>'
            out.append(f'<div class="card2"><h3>{e(s["stem"])}{note} &middot; {s["done"]} / {s["total"]} jobs</h3>'
                       f'<div class="bars">{bars}</div></div>')
        rows, rcls = [], []
        for j in q["jobs"]:
            hist = j.get("history") or []
            att = "; ".join(f"#{h['attempt']} rc={h['rc']} {fmt_dur(h['seconds']) if h['seconds'] is not None else ''}"
                            for h in hist if h.get("end"))
            dur = fmt_dur(j["seconds"]) if j.get("seconds") is not None else (
                f"{fmt_dur(j['elapsed_s'])} elapsed" if j.get("elapsed_s") is not None else "")
            rows.append([f"<code>{e(j['job'])}</code>", badge(j["state"]), e(str(j["attempts"])), fmt(j.get("gpu")),
                         fmt(j.get("start")), e(dur), fmt(j.get("rc")), e(att),
                         f"<code>{e(j['log'])}</code>" if j.get("log") else ""])
            rcls.append("hl" if j["state"] == "running" else "")
        out.append("<h3>Jobs (status.jsonl)</h3>" + table(
            ["job", "state", "attempts", "gpu", "last start", "duration / elapsed", "rc", "finished attempts", "log"],
            rows, rcls))
    sm = st.get("summaries") or {}
    out.append("<h3>Summary generation</h3>")
    if not sm.get("trims"):
        out.append(ns("no embedding_cosine trims folder yet in runs/cache/trims"))
    else:
        out.append(table(["trims folder", "dataset", "K", "M", "selection_embeddings_sha", "created", "files"],
                         [[f"<code>{e(t['folder'])}</code>", e(str(t.get("dataset"))), fmt(t.get("K")),
                           fmt(t.get("M")), f"<code>{e(str(t.get('selection_embeddings_sha')))}</code>",
                           fmt(t.get("created")), e(", ".join(t.get("files") or []))] for t in sm["trims"]]))
    for ms in sm.get("metapath", []):
        if not ms.get("folder"):
            out.append(f'<div class="card2"><h4>metapath summaries &middot; {e(str(ms.get("dataset")))} &middot; trims '
                       f'<code>{e(ms["trims"])}</code></h4>' + bar(0, ms.get("required") or "?", "summaries.jsonl lines")
                       + ns("metapath store not created yet") + "</div>")
            continue
        rate = (f" · {ms['rate_per_min']:.1f}/min ({ms.get('rate_source')})" if ms.get("rate_per_min") else "")
        eta = f" · ETA {fmt_dur(ms['eta_s'])}" if ms.get("eta_s") else ""
        idle = (f'<span class="muted"> last write {fmt_dur(ms["idle_s"])} ago</span>'
                if ms.get("idle_s") is not None else "")
        out.append(f'<div class="card2"><h4>metapath summaries &middot; {e(str(ms.get("dataset")))} &middot; store '
                   f'<code>{e(ms["folder"])}</code> (trims <code>{e(ms["trims"])}</code>, node store '
                   f'<code>{e(str(ms.get("node_store")))}</code>)</h4>'
                   + bar(ms.get("lines") or 0, ms.get("required") or "?", "summaries.jsonl lines", rate + eta)
                   + f'<p class="sm">{fmt_bytes(ms.get("bytes"))} &middot; mtime {fmt(ms.get("mtime"))}{idle}</p>'
                   + "</div>")
    for ns_ in sm.get("node", []):
        req = ns_.get("required_new")
        new = ns_.get("new_since_dashboard_start")
        rate = f" · {ns_['rate_per_min']:.1f}/min" if ns_.get("rate_per_min") else ""
        body = (f'<p>total lines <b>{fmt(ns_.get("lines"))}</b> &middot; required new (estimates.json) '
                f'<b>{fmt(req)}</b> &middot; new since dashboard start {fmt(new)}{e(rate)} &middot; '
                f'{fmt_bytes(ns_.get("bytes"))} &middot; mtime {fmt(ns_.get("mtime"))}</p>')
        if req and new is not None:
            body += bar(min(new, req), req, "new lines seen by this dashboard (lower bound)")
        out.append(f'<div class="card2"><h4>node summaries &middot; {e(str(ns_.get("dataset")))} &middot; '
                   f'<code>{e(ns_["folder"])}</code> ({e(str(ns_.get("model")))})</h4>{body}</div>')
    return "".join(out)


def sec_training(st: dict) -> str:
    out = ['<h2 id="training">6. Training</h2>']
    tr = st.get("training") or {}
    for pc in tr.get("per_config", []):
        b = pc.get("baseline") or {}
        out.append(f'<div class="card2"><h3>{e(pc["config"])} &middot; {e(pc["dataset"])}</h3>'
                   + bar(pc["finished"], len(pc["seeds"]) or 5, "seeds finished",
                         f" · {pc['in_progress']} in progress")
                   + f'<p class="sm">per-seed time {e(fmt_dur(pc.get("per_seed_s")) or "?")} &middot; ETA all seeds '
                     f'<b>{e(fmt_dur(pc.get("eta_all_s")) if pc.get("eta_all_s") is not None else "?")}</b> &middot; '
                     f'baseline: {fmt(b.get("runs"))} runs, {e(f2(b.get("s_per_epoch")))} s/epoch, job '
                     f'{e(fmt_dur(b.get("job_s")))}</p></div>')
    runs = tr.get("runs") or []
    if not runs:
        return "".join(out) + ns(f"no run folders under {tr.get('folder', 'experiments/gnn/runs/results/raw')} yet")
    for r in runs:
        if r.get("error"):
            out.append(f"<h3>{e(r['run'])}</h3>" + ns(r["error"]))
            continue
        rows = r["rows"]
        epochs = [str(x.get("epoch", i + 1)) for i, x in enumerate(rows)]
        bl = r.get("baseline_rows") or []
        n = max(len(rows), len(bl), 1)
        xl = [str(i + 1) for i in range(n)]
        charts = ""
        if rows:
            def col(rs, k):
                return [x.get(k) for x in rs]
            charts += sparkline([{"name": "arm val AUROC", "values": col(rows, "val_auroc"), "cls": "s1"}]
                                + ([{"name": "baseline val AUROC (same seed)", "values": col(bl, "val_auroc"),
                                     "cls": "s2", "dash": True}] if bl else []),
                                w=340, h=120, title="val AUROC", x_labels=xl)
            charts += sparkline([{"name": "arm val loss", "values": col(rows, "val_loss"), "cls": "s1"}]
                                + ([{"name": "baseline val loss", "values": col(bl, "val_loss"), "cls": "s2",
                                     "dash": True}] if bl else []),
                                w=340, h=120, title="val loss", x_labels=xl, fmt_v=lambda v: f"{float(v):.3f}")
            if any(num(x.get("train_loss")) is not None for x in rows):
                charts += sparkline([{"name": "arm train loss", "values": col(rows, "train_loss"), "cls": "s1"}]
                                    + ([{"name": "baseline train loss", "values": col(bl, "train_loss"), "cls": "s2",
                                         "dash": True}] if bl else []),
                                    w=340, h=120, title="train loss", x_labels=xl, fmt_v=lambda v: f"{float(v):.3f}")
        keys = list(dict.fromkeys(k for x in rows for k in x if k != "epoch"))
        bl_by_ep = {x.get("epoch"): x for x in bl}
        trows = [[e(ep)] + [fmt(x.get(k)) for k in keys] + [f2(bl_by_ep.get(x.get("epoch"), {}).get("val_auroc"))]
                 for ep, x in zip(epochs, rows, strict=False)]
        rcls = ["hl" if r.get("metrics_best_epoch") is not None and str(x.get("epoch")) == str(r["metrics_best_epoch"])
                else "" for x in rows]
        spe = r.get("s_per_epoch")
        info = (f'<p class="sm">epochs <b>{r["epochs"]} / {r["max_epochs"]}</b> &middot; '
                f'{e(f2(spe)) or "?"} s/epoch ({e(str(r.get("s_per_epoch_source")))}) &middot; ETA seed '
                f'<b>{e(fmt_dur(r["eta_seed_s"]) if r.get("eta_seed_s") is not None else "?")}</b> &middot; last write '
                f'{e(fmt_dur(r.get("idle_s")))} ago &middot; best val AUROC at epoch {fmt(r.get("best_val_auroc_epoch"))}'
                + (f' &middot; kept epoch (primary rule) {fmt(r.get("metrics_best_epoch"))} &middot; rules '
                   f'{fmt(r.get("selection"))} &middot; runtime {e(fmt_dur(r.get("runtime_seconds")))} &middot; peak '
                   f'{fmt(r.get("peak_memory"))}' if r["status"] == "finished" else "")
                + f' &middot; files: {e(", ".join(r.get("files") or []))}</p>')
        body = (bar(r["epochs"], r["max_epochs"], "epochs") + info + f'<div class="charts">{charts}</div>'
                + table(["epoch", *keys, "baseline val_auroc"], trows, rcls))
        out.append(details(f"run:{r['run']}", f"<b>{e(r['run'])}</b> {badge(r['status'])} &middot; seed "
                           f"{fmt(r.get('seed'))} &middot; {e(str(r.get('config')))} &middot; {r['epochs']} / "
                           f"{r['max_epochs']} epochs", body, open_=r["status"] != "finished"))
    return "".join(out)


def sec_results(st: dict) -> str:
    out = ['<h2 id="results">7. Results</h2>',
           '<p class="warn"><b>Test set.</b> Per-seed numbers for monitoring only; final statistics (CIs, Holm) come '
           "from <code>scripts/compare_arms.py</code>.</p>"]
    res = st.get("results") or {}
    any_rows = False
    for a in res.get("arms", []):
        if not a["n_arm"]:
            out.append(f"<h3>{e(a['config'])}</h3>" + ns("no finished seeds yet (metrics.json absent)"))
            continue
        any_rows = True
        rows = []
        for r in a["rows"]:
            cells = [e(str(r["seed"]))]
            for k in ("macro_f1", "auroc", "auprc"):
                av = (r["arm"] or {}).get(k)
                bv = (r["baseline"] or {}).get(k)
                dv = (r.get("delta") or {}).get(k)
                cls = "pos" if (dv or 0) > 0 else ("neg" if (dv or 0) < 0 else "")
                cells += [f2(av), f2(bv), f'<span class="{cls}">{"+" if (dv or 0) > 0 else ""}{f2(dv)}</span>']
            cells.append(f"{fmt(r.get('arm_best_epoch'))} / {fmt(r.get('baseline_best_epoch'))}")
            rows.append(cells)

        def ms(x):
            return f"{x['mean']:.2f} ± {x['std']:.2f} (n={x['n']})" if x else ""

        srow = ["<b>mean ± std (pop.)</b>"]
        brow = ["baseline, all seeds"]
        for k in ("macro_f1", "auroc", "auprc"):
            s = a["summary"][k]
            d = s["delta"]
            srow += [ms(s["arm"]), ms(s["baseline_paired"]),
                     f'<b class="{"pos" if d and d["mean"] > 0 else "neg"}">{("+" if d and d["mean"] > 0 else "")}'
                     f'{d["mean"]:.2f}</b>' if d else ""]
            brow += ["", ms(s["baseline_all"]), ""]
        rows += [srow + [""], brow + [""]]
        out.append(f"<h3>{e(a['config'])} vs Ours K=1 (<code>{e(str(a['baseline_hash']))}</code>) &middot; test set"
                   f"</h3>" + table(["seed", "arm Macro-F1", "Ours Macro-F1", "Δ F1", "arm AUROC", "Ours AUROC",
                                     "Δ AUROC", "arm AUPRC", "Ours AUPRC", "Δ AUPRC", "best epoch arm / Ours"], rows))
    if not res.get("arms") or not any_rows:
        out.append(ns("no results yet") if res.get("arms") else NS)
    for f in res.get("compare", []):
        out.append(details(f"cmp:{f['path']}", f"compare_arms report <code>{e(f['path'])}</code>",
                           md_render(f.get("text")), open_=True))
    if not res.get("compare"):
        out.append(ns("no experiments/gnn/runs/compare/*.md yet"))
    return "".join(out)


def sec_system(st: dict) -> str:
    out = ['<h2 id="system">8. System</h2>']
    sy = st.get("system") or {}
    g = sy.get("gpu") or {}
    if g.get("gpus"):
        out.append(table(["#", "GPU", "util %", "memory", "temp °C", "power W"],
                         [[e(str(x["index"])), e(x["name"]), fmt(x["util"], 0),
                           e(f"{(x['mem_used'] or 0):.0f} / {(x['mem_total'] or 0):.0f} MiB"), fmt(x["temp"], 0),
                           fmt(x.get("power_w"), 0)] for x in g["gpus"]]))
        if g.get("apps"):
            out.append(table(["pid", "process", "GPU MiB"], [[e(a["pid"]), e(a["name"]), fmt(a["mem_mib"], 0)]
                                                              for a in g["apps"]]))
    else:
        out.append(ns("nvidia-smi unavailable"))
    hist = sy.get("gpu_history") or []
    if len(hist) >= 2:
        def hm(t):
            return datetime.fromtimestamp(t, UTC).strftime("%H:%M")
        out.append('<div class="charts wide">' + sparkline(
            [{"name": "GPU util %", "values": [h["util"] for h in hist], "cls": "s1"},
             {"name": "GPU memory %", "values": [h["mem_pct"] for h in hist], "cls": "s2"}],
            w=760, h=160, title=f"GPU 0, last {fmt_dur(hist[-1]['t'] - hist[0]['t'])} (UTC, {SAMPLE_SECONDS} s "
                                f"samples, max 6 h, in memory)",
            x_labels=[hm(h["t"]) for h in hist], ymin=0, ymax=100, fmt_v=lambda v: f"{float(v):.0f}") + "</div>")
    else:
        out.append(ns("GPU history: collecting (one sample every 15 s)"))
    la, mem, dk = sy.get("load"), sy.get("mem"), sy.get("disk")
    kv = {}
    if la:
        kv["load average 1/5/15 min"] = f"{la['1m']:.2f} / {la['5m']:.2f} / {la['15m']:.2f} ({la['cpus']} CPUs)"
    if mem:
        kv["RAM used / total"] = f"{fmt_bytes(mem['used'])} / {fmt_bytes(mem['total'])} " \
                                 f"(available {fmt_bytes(mem['available'])}, swap used {fmt_bytes(mem['swap_used'])})"
    if dk:
        kv["disk /workspace"] = f"{fmt_bytes(dk['used'])} / {fmt_bytes(dk['total'])} used, {fmt_bytes(dk['free'])} free " \
                                f"({100 * dk['used'] / dk['total']:.0f}%)"
    kv["dashboard uptime"] = fmt_dur(sy.get("dashboard_uptime_s"))
    kv["page build"] = f"{st.get('build_seconds', 0) * 1000:.0f} ms"
    out.append(kv_table(kv))
    out.append("<h3>tmux sessions</h3>" + (f"<pre>{e(chr(10).join(sy['tmux']))}</pre>" if sy.get("tmux")
                                           else ns("tmux ls: none / unavailable")))
    procs = sy.get("processes") or []
    out.append("<h3>Relevant processes</h3>" + (table(
        ["pid", "ppid", "elapsed", "RSS MB", "CPU %", "command"],
        [[e(str(p["pid"])), e(str(p["ppid"])), e(p["elapsed"]), fmt(p["rss_mb"], 1), fmt(p["cpu_pct"], 1),
          f"<code>{e(p['args'])}</code>"] for p in procs]) if procs else ns("none running")))
    return "".join(out)


def sec_logs(st: dict) -> str:
    out = ['<h2 id="logs">9. Logs</h2>']
    lg = st.get("logs") or {}

    def one(x, key, open_):
        return details(key, f"<code>{e(x['path'])}</code> <span class='muted'>{fmt_bytes(x.get('bytes'))} &middot; "
                            f"{fmt(x.get('mtime'))}</span>", f"<pre>{e(x.get('tail') or '(empty)')}</pre>", open_)

    out.append("<h3>Running jobs (last 40 lines)</h3>")
    out.append("".join(one(x, f"log:{x['path']}", True) for x in lg.get("running", [])) or ns("no running job"))
    if lg.get("queue_other"):
        out.append("<h3>Other queue job logs</h3>" + "".join(one(x, f"log:{x['path']}", False)
                                                             for x in lg["queue_other"]))
    out.append("<h3>experiments/gnn/logs/</h3>")
    out.append("".join(one(x, f"log:{x['path']}", False) for x in lg.get("experiment", [])) or ns("no files"))
    return "".join(out)


def sec_files(st: dict) -> str:
    out = ['<h2 id="files">10. Files</h2>']
    fl = st.get("files") or {}
    files = fl.get("files") or []
    if not files:
        return "".join(out) + ns("no files")
    now = time.time()
    rows = [[f"<code>{e(f['path'])}</code>", e(fmt_bytes(f["bytes"])), e(f["mtime"] or ""),
             e(fmt_dur(now - f["m"]) + " ago")] for f in files]
    recent = sorted(files, key=lambda f: -f["m"])[:12]
    out.append("<h3>Most recently modified</h3>" + table(
        ["path", "size", "mtime (UTC)", "age"],
        [[f"<code>{e(f['path'])}</code>", e(fmt_bytes(f["bytes"])), e(f["mtime"] or ""),
          e(fmt_dur(now - f["m"]) + " ago")] for f in recent]))
    note = f" (truncated at {fl.get('limit')})" if fl.get("truncated") else ""
    out.append(details("files:all", f"All {len(files)} files{e(note)} &middot; {fmt_bytes(fl.get('total_bytes'))} "
                                    "(experiments/gnn/runs/** and the new cache folders)",
                       table(["path", "size", "mtime (UTC)", "age"], rows)))
    return "".join(out)


NAV = [("overview", "Overview"), ("configs", "Configs"), ("embeddings", "Embeddings"), ("diagnostics", "Diagnostics"),
       ("pipeline", "Pipeline"), ("training", "Training"), ("results", "Results"), ("system", "System"),
       ("logs", "Logs"), ("files", "Files")]


def render_main(st: dict) -> str:
    secs = []
    for fn in (sec_overview, sec_configs, sec_embeddings, sec_diagnostics, sec_pipeline, sec_training, sec_results,
               sec_system, sec_logs, sec_files):
        try:
            secs.append(fn(st))
        except Exception as exc:
            secs.append(f"<h2>{e(fn.__name__)}</h2><p class='warn'>render error: {e(repr(exc))}</p>")
    errs = st.get("errors")
    err = f'<p class="warn">collection errors: {e(json.dumps(errs))}</p>' if errs else ""
    head = (f'<p class="muted sm" id="gen">Generated {e(st["generated"])} UTC &middot; built in '
            f'{st.get("build_seconds", 0) * 1000:.0f} ms &middot; auto-refresh 15 s &middot; '
            f'<a href="/api/state">JSON</a> &middot; root <code>{e(st["root"])}</code></p>')
    return head + sec_header(st) + err + "".join(f"<section>{s}</section>" for s in secs)


def render(st: dict) -> str:
    nav = "".join(f'<a href="#{k}">{e(v)}</a>' for k, v in NAV)
    return PAGE.replace("{{NAV}}", nav).replace("{{MAIN}}", render_main(st))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<noscript><meta http-equiv="refresh" content="15"></noscript>
<title>GNN Exp1 Progress</title><style>
:root{--bg:#f6f6f4;--fg:#0b0b0b;--card:#fcfcfb;--mut:#52514e;--bd:#e2e1dc;--pre:#f0efec;--s1:#2a78d6;--s2:#eb6834;
--hl:#2a78d61a;--good:#0ca30c;--warn:#b67800;--crit:#d03b3b;--nav:#fcfcfbee;--track:#e6e5e0}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--fg:#f0efec;--card:#1a1a19;--mut:#c3c2b7;--bd:#33332f;
--pre:#111110;--s1:#3987e5;--s2:#d95926;--hl:#3987e526;--good:#0ca30c;--warn:#fab219;--crit:#e66767;--nav:#1a1a19ee;
--track:#2c2c29}}
*{box-sizing:border-box}html{scroll-padding-top:52px}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
overflow-x:hidden}
nav{position:sticky;top:0;z-index:10;background:var(--nav);backdrop-filter:blur(6px);border-bottom:1px solid var(--bd);
display:flex;gap:4px;overflow-x:auto;padding:8px 12px;white-space:nowrap}
nav b{margin-right:8px}nav a{color:var(--fg);text-decoration:none;padding:3px 9px;border-radius:999px;font-size:13px}
nav a:hover{background:var(--hl)}
main{max-width:1280px;margin:0 auto;padding:12px 16px}
h2{font-size:17px;margin:0 0 10px}h3{font-size:14.5px;margin:14px 0 6px}h4{font-size:13.5px;margin:12px 0 6px}
section{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:14px;margin-bottom:14px;min-width:0}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-bottom:14px}
.tile{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:10px 12px;min-width:0}
.tl{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}.tv{margin-top:4px;word-break:break-word}
.phases{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px}
.ph{border:1px solid var(--bd);border-radius:10px;padding:8px 10px}.ph-running,.ph-waiting_gate{border-color:var(--s1)}
.phn{font-weight:600;margin:2px 0}
.card2{border:1px solid var(--bd);border-radius:10px;padding:8px 12px;margin:8px 0;min-width:0}
.card2 h3,.card2 h4{margin:4px 0 8px}
.tw{overflow-x:auto;max-width:100%;margin:4px 0}table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--bd);vertical-align:top}
th{color:var(--mut);font-weight:600;white-space:nowrap}td{font-variant-numeric:tabular-nums}
table.kv td:first-child{white-space:nowrap;color:var(--mut)}tr.hl td{background:var(--hl);font-weight:600}
pre{background:var(--pre);border:1px solid var(--bd);border-radius:8px;padding:8px;overflow:auto;font-size:12px;
max-height:420px;margin:6px 0;white-space:pre}
code{font-size:12px;word-break:break-all;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.muted{color:var(--mut)}.ns{color:var(--mut);font-style:italic}.sm{font-size:12.5px}
.warn{border-left:3px solid var(--warn);padding:4px 10px;background:var(--pre);border-radius:4px}
.pos{color:var(--good)}.neg{color:var(--crit)}
details{border:1px solid var(--bd);border-radius:10px;margin:8px 0;min-width:0}
summary{cursor:pointer;padding:7px 10px;font-size:13px}details[open]>summary{border-bottom:1px solid var(--bd)}
.dbody{padding:6px 10px;min-width:0}
.b{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;font-weight:600;white-space:nowrap;
background:var(--track);color:var(--fg)}
.b-running,.b-queuealive{background:var(--s1);color:#fff}.b-done,.b-finished{background:#0a7f0a;color:#fff}
.b-failed,.b-gave_up,.b-stalled{background:var(--crit);color:#fff}
.b-failed-attempt,.b-interrupted,.b-queuestopped{background:#c4561f;color:#fff}
.b-waiting_gate,.b-incomplete{background:#a26b00;color:#fff}
.bars{display:grid;gap:6px}.bar-l{display:flex;justify-content:space-between;gap:8px;font-size:12.5px;flex-wrap:wrap}
.bar-t{height:8px;background:var(--track);border-radius:4px;overflow:hidden}
.bar-f{height:100%;background:var(--s1);border-radius:4px}.bar-f.full{background:var(--good)}.bar{margin:4px 0}
.charts{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0}.charts.wide figure{flex:1 1 100%}
figure.chart{margin:0;min-width:0;max-width:100%}figcaption{font-size:12px;color:var(--mut);margin-bottom:2px}
svg.spark{max-width:100%;height:auto;display:block;overflow:visible}
.charts.wide svg.spark{width:100%}
svg .grid{stroke:var(--bd);stroke-width:1}svg .ax{fill:var(--mut);font-size:9.5px}
svg .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}svg .ln.s1{stroke:var(--s1)}
svg .ln.s2{stroke:var(--s2)}svg .pt{stroke:var(--card);stroke-width:1}svg .pt.s1{fill:var(--s1)}
svg .pt.s2{fill:var(--s2)}svg .pt:hover{r:4}
.lg{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--mut)}
.sw{display:inline-block;width:14px;height:0;border-top:2px solid;vertical-align:middle;margin-right:4px}
.sw.s1{border-color:var(--s1)}.sw.s2{border-color:var(--s2)}.sw.dash{border-top-style:dashed}
.md p{margin:6px 0}.md table td{min-width:90px}.md table td:first-child{min-width:0}
ul{padding-left:20px}li{margin:2px 0}a{color:var(--s1)}
</style></head><body><nav><b>GNN Exp1</b>{{NAV}}</nav><main id="main">{{MAIN}}</main>
<script>
(function(){
  var KEY_S="gnnScroll", KEY_D="gnnDetails";
  function load(k){try{return JSON.parse(sessionStorage.getItem(k)||"null");}catch(x){return null;}}
  function save(k,v){try{sessionStorage.setItem(k,JSON.stringify(v));}catch(x){}}
  function applyDetails(){
    var m=load(KEY_D)||{};
    document.querySelectorAll("details[data-k]").forEach(function(d){
      var k=d.getAttribute("data-k"); if(k in m){d.open=!!m[k];}
    });
  }
  document.addEventListener("toggle",function(ev){
    var d=ev.target; if(!d||!d.getAttribute||!d.getAttribute("data-k"))return;
    var m=load(KEY_D)||{}; m[d.getAttribute("data-k")]=d.open; save(KEY_D,m);
  },true);
  applyDetails();
  var y=load(KEY_S); if(typeof y==="number"&&!location.hash){window.scrollTo(0,y);}
  window.addEventListener("scroll",function(){save(KEY_S,window.scrollY);},{passive:true});
  window.addEventListener("beforeunload",function(){save(KEY_S,window.scrollY);});
  function refresh(){
    fetch("/?partial=1",{cache:"no-store"}).then(function(r){
      if(!r.ok)throw new Error(r.status); return r.text();
    }).then(function(t){
      var sy=window.scrollY; document.getElementById("main").innerHTML=t; applyDetails(); window.scrollTo(0,sy);
    }).catch(function(){
      var g=document.getElementById("gen"); if(g&&g.textContent.indexOf("[stale")<0){g.textContent+=" [stale: refresh failed]";}
    });
  }
  setInterval(refresh,15000);
})();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, ctype: str, body: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path, _, query = self.path.partition("?")
        try:
            if path == "/api/state":
                self._send(200, "application/json", json.dumps(build_state(), indent=1, default=str))
            elif path in ("/", "/index.html"):
                st = build_state()
                body = render_main(st) if "partial=1" in query else render(st)
                self._send(200, "text/html; charset=utf-8", body)
            else:
                self._send(404, "text/plain", "not found")
        except Exception as exc:  # never crash the server
            try:
                self._send(500, "text/plain", f"error: {exc!r}")
            except Exception:
                pass

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    ROOT = Path(a.root).resolve()
    threading.Thread(target=sampler, daemon=True, name="sampler").start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"GNN dashboard on http://{a.host}:{a.port} (root={ROOT})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
