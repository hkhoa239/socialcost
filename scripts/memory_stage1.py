#!/usr/bin/env python3
"""Temporal interaction memory over stored stage-1 predictions: tune (validation only), build, evaluate.

  tune   grid over hold / decay / EMA / geometry on the validation split; objective = mean NLL over the unobserved
         time-since-seen buckets (0-1, 1-2, 2-5, > 5 s) of pair relations + raised hand, equally weighted.
  build  writes stage1/<src>mem<mode>/preds/<eid>.npz for every episode (same keys as the source + edge_u, node_u,
         stale_edge, stale_node, geo_reset) so TypeMix trains and evaluates on them as on any stage-1 source
  eval   per mode: relation / gesture metrics by time-since-last-seen bucket on Test A+B+C (+ per split)

Visibility = the cascade cache's per-person sensor visibility fraction (depth-consistent body samples, no ground truth).
Ground-truth relations (cascade cache `inter` / `itype`, the stage-1 training targets) are used only for scoring.

    $PY memory_stage1.py tune  --src cffullraw --out <json>
    $PY memory_stage1.py build --src cffullraw --config <tuned json>
    $PY memory_stage1.py eval  --src cffullraw --config <tuned json> --out <json> --report <md>
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.bench.data import STAGE1_ROOT, split_ids  # noqa: E402
from socialcost.bench.interaction_net import type_class  # noqa: E402
from socialcost.bench.memory import MemoryConfig, run_episode  # noqa: E402

CACHE = os.environ.get("SC_CASCADE_CACHE", os.path.join(os.path.dirname(STAGE1_ROOT), "cascade_cache"))
BUCKETS = [("visible", 0, 0), ("0-1 s", 1e-6, 1), ("1-2 s", 1, 2), ("2-5 s", 2, 5), ("> 5 s", 5, 1e9), ("never seen", np.inf, np.inf)]
MODES = ("none", "hold", "decay", "decay_geo")
DT = 0.2


def load(eid: str, src: str):
    f = os.path.join(STAGE1_ROOT, src, "preds", f"{eid}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    c = np.load(os.path.join(CACHE, f"{eid}.npz"))
    return {k: z[k] for k in z.files}, c


def gt_edges(c) -> np.ndarray:
    gid, ity = c["inter"], c["itype"]
    T, P = gid.shape
    out = np.zeros((T, P, P), np.int64)
    for t in range(T):
        same = (gid[t][:, None] >= 0) & (gid[t][:, None] == gid[t][None, :])
        cls = np.array([type_class(x) for x in ity[t]])
        out[t] = np.where(same, cls[:, None], 0)
    return out


def priors(src: str) -> tuple[tuple, tuple]:
    """Class priors from the TRAIN split ground truth (pair classes, raised hand, wheelchair)."""
    ce, cn, n_e, n_n = np.zeros(5), np.zeros(2), 0, 0
    for eid in split_ids("train"):
        f = os.path.join(CACHE, f"{eid}.npz")
        if not os.path.exists(f):
            continue
        c = np.load(f)
        g = gt_edges(c)
        T, P = g.shape[:2]
        iu = np.triu_indices(P, 1)
        v = g[:, iu[0], iu[1]].ravel()
        ce += np.bincount(v, minlength=5)[:5]
        n_e += v.size
        cn += [c["halt"].sum(), c["wheel"].sum() * T]
        n_n += T * P
    pe = ce / max(n_e, 1)
    pe = np.clip(pe, 1e-3, None)
    return tuple((pe / pe.sum()).round(4)), tuple((np.clip(cn / max(n_n, 1), 1e-3, None)).round(4))


def episode_rows(eid: str, src: str, cfg: MemoryConfig) -> dict | None:
    d = load(eid, src)
    if d is None:
        return None
    z, c = d
    names = [str(n) for n in c["names"]]
    T, P = c["vis"].shape
    if P < 1:
        return None
    m = run_episode(cfg, names, c["track"], c["vis"], z["edge_p"].astype(np.float32), z["node_p"].astype(np.float32), DT)
    g = gt_edges(c)
    iu = np.triu_indices(P, 1)
    rows = {"ep": m["edge_p"][:, iu[0], iu[1]].reshape(-1, 5), "gt": g[:, iu[0], iu[1]].ravel(), "st": m["stale_edge"][:, iu[0], iu[1]].ravel(),
            "geo": m["geo_reset"][:, iu[0], iu[1]].ravel()}
    # 'ended while unobserved': GT relation at the last joint observation was positive and is none now
    last_pos = np.zeros((T, len(iu[0])), bool)
    seen_pos = np.zeros(len(iu[0]), bool)
    st = m["stale_edge"][:, iu[0], iu[1]]
    gg = g[:, iu[0], iu[1]]
    for t in range(T):
        obs = st[t] == 0
        seen_pos = np.where(obs, gg[t] > 0, seen_pos)
        last_pos[t] = seen_pos
    rows["ended"] = (last_pos & (gg == 0) & (st > 0)).ravel()
    # re-entry: observed now after a gap >= 1 s (first 2 s after re-observation)
    re = np.zeros((T, len(iu[0])), bool)
    gap_end = np.full(len(iu[0]), -1e9)
    prev = np.full(len(iu[0]), np.inf)
    for t in range(T):
        now = st[t]
        back = (now == 0) & (prev >= 1.0) & np.isfinite(prev)
        gap_end = np.where(back, t * DT, gap_end)
        re[t] = (now == 0) & (t * DT - gap_end <= 2.0)
        prev = now
    rows["reentry"] = re.ravel()
    rows["np"] = m["node_p"].reshape(-1, 2)
    rows["ngt"] = np.stack([c["halt"].astype(np.float32), np.repeat(c["wheel"][None].astype(np.float32), T, 0)], -1).reshape(-1, 2)
    rows["nst"] = m["stale_node"].ravel()
    return rows


def collect(ids, src, cfg, workers=8) -> dict:
    with Pool(workers) as pool:
        parts = [r for r in pool.starmap(episode_rows, [(e, src, cfg) for e in ids]) if r is not None]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def bucket_mask(st, lo, hi):
    if lo == 0 and hi == 0:
        return st == 0
    if np.isinf(lo):
        return np.isinf(st)
    return (st > lo - (1e-6 if lo > 0 else 0)) & (st <= hi) & (st > 0) & np.isfinite(st)


def f1(y, yhat):
    tp = float((y & yhat).sum()); fp = float((~y & yhat).sum()); fn = float((y & ~yhat).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None


def ece(y, p, bins=10):
    e, n = 0.0, len(y)
    if n == 0:
        return None
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins + (1e-9 if b == bins - 1 else 0))
        if m.any():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics(R: dict) -> dict:
    out = {}
    for name, lo, hi in BUCKETS:
        m = bucket_mask(R["st"], lo, hi)
        if not m.any():
            continue
        p, g = R["ep"][m].astype(np.float64), R["gt"][m]
        pint = 1 - p[:, 0]
        y = g > 0
        nll = float(-np.log(np.clip(p[np.arange(len(g)), g], 1e-6, 1)).mean())
        out[name] = {"pairs": int(m.sum()), "positives": int(y.sum()), "inter_f1": f1(y, pint > 0.5), "talk_f1": f1(g == 1, p.argmax(1) == 1),
                     "queue_f1": f1(g == 2, p.argmax(1) == 2), "nll": nll, "ece": ece(y.astype(float), pint),
                     "pos_recall": float((pint[y] > 0.5).mean()) if y.any() else None,
                     "fp_rate": float((pint[~y] > 0.5).mean()) if (~y).any() else None}
        nm = bucket_mask(R["nst"], lo, hi)
        if nm.any():
            hp, hg = R["np"][nm, 0].astype(np.float64), R["ngt"][nm, 0] > 0.5
            wp, wg = R["np"][nm, 1].astype(np.float64), R["ngt"][nm, 1] > 0.5
            bce = lambda pp, yy: float(-(yy * np.log(np.clip(pp, 1e-6, 1)) + (~yy) * np.log(np.clip(1 - pp, 1e-6, 1))).mean())  # noqa: E731
            out[name].update({"persons": int(nm.sum()), "halt_f1": f1(hg, hp > 0.5), "halt_nll": bce(hp, hg), "wheel_f1": f1(wg, wp > 0.5),
                              "wheel_nll": bce(wp, wg)})
    stale = (R["st"] > 0) & np.isfinite(R["st"])
    e = R["ended"]
    out["ended_while_unobserved"] = {"pairs": int(e.sum()), "still_predicted_interacting": float(((1 - R["ep"][e, 0]) > 0.5).mean()) if e.any() else None}
    out["stale_false_positive_persistence"] = float(((1 - R["ep"][stale & (R["gt"] == 0), 0]) > 0.5).mean()) if (stale & (R["gt"] == 0)).any() else None
    re = R["reentry"]
    out["after_reentry"] = {"pairs": int(re.sum()), "inter_f1": f1(R["gt"][re] > 0, (1 - R["ep"][re, 0]) > 0.5)} if re.any() else {}
    out["geo_resets"] = int(R["geo"].sum())
    stale_b = [b for b in ("0-1 s", "1-2 s", "2-5 s", "> 5 s") if b in out]
    out["objective_stale_nll"] = float(np.mean([out[b]["nll"] + out[b].get("halt_nll", 0) for b in stale_b])) if stale_b else None
    return out


def tune(src: str, out: str) -> None:
    ids = split_ids("val")
    pe, pn = priors(src)
    res = {"priors": {"edge": pe, "node": pn}, "grid": []}
    base = dict(edge_prior=pe, node_prior=pn)
    cands = [MemoryConfig(mode="none", **base)]
    for tgt, h, d, e in itertools.product(("prior", "stage1"), (0.0, 1.0, 2.0, 3.0, 5.0), (1.0, 3.0, 5.0, 10.0), (1.0, 0.7)):
        cands.append(MemoryConfig(mode="decay", decay_target=tgt, hold_s=h, decay_s=d, ema=e, **base))
    for e in (1.0, 0.7):
        cands.append(MemoryConfig(mode="hold", ema=e, **base))
    for c in cands:
        m = metrics(collect(ids, src, c))
        res["grid"].append({"cfg": dataclasses.asdict(c), "objective": m["objective_stale_nll"], "metrics": m})
        print(c.tag(), round(m["objective_stale_nll"], 4), flush=True)
    best = min((g for g in res["grid"] if g["cfg"]["mode"] == "decay"), key=lambda g: g["objective"])
    # geometry thresholds on top of the best decay schedule
    for gs, mt in itertools.product((1.0, 1.5, 2.5), (2.5, 3.0, 4.0)):
        c = MemoryConfig(**{**best["cfg"], "mode": "decay_geo", "geo_sep_growth": gs, "geo_max_sep_talk": mt})
        m = metrics(collect(ids, src, c))
        res["grid"].append({"cfg": dataclasses.asdict(c), "objective": m["objective_stale_nll"], "metrics": m})
        print(c.tag(), gs, mt, round(m["objective_stale_nll"], 4), flush=True)
    bh = min((g for g in res["grid"] if g["cfg"]["mode"] == "hold"), key=lambda g: g["objective"])
    bg = min((g for g in res["grid"] if g["cfg"]["mode"] == "decay_geo"), key=lambda g: g["objective"])
    res["selected"] = {"none": cands[0].__dict__ | {"edge_prior": pe, "node_prior": pn}, "hold": bh["cfg"], "decay": best["cfg"], "decay_geo": bg["cfg"]}
    res["selected"]["none"] = dataclasses.asdict(cands[0])
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("selected", json.dumps(res["selected"], default=float))


def build_one(job) -> None:
    eid, src, mode, cfgd = job
    d = load(eid, src)
    dst = os.path.join(STAGE1_ROOT, f"{src}mem{mode}", "preds", f"{eid}.npz")
    if d is None or os.path.exists(dst):
        return
    z, c = d
    names = [str(n) for n in c["names"]]
    m = run_episode(MemoryConfig(**cfgd), names, c["track"], c["vis"], z["edge_p"].astype(np.float32), z["node_p"].astype(np.float32), DT)
    out = dict(z)
    out["edge_p"], out["node_p"] = m["edge_p"].astype(np.float16), m["node_p"].astype(np.float16)
    for k in ("edge_u", "node_u", "stale_edge", "stale_node"):
        out[k] = np.minimum(m[k], 1e4).astype(np.float16)
    out["geo_reset"] = m["geo_reset"]
    np.savez_compressed(dst, **out)


def build(src: str, config: str, modes) -> None:
    sel = json.load(open(config))["selected"]
    eps = sorted(f[:-4] for f in os.listdir(os.path.join(STAGE1_ROOT, src, "preds")) if f.endswith(".npz"))
    for mode in modes:
        root = os.path.join(STAGE1_ROOT, f"{src}mem{mode}")
        os.makedirs(os.path.join(root, "preds"), exist_ok=True)
        for f in ("metrics.json",):
            if os.path.exists(os.path.join(STAGE1_ROOT, src, f)) and not os.path.exists(os.path.join(root, f)):
                shutil.copy(os.path.join(STAGE1_ROOT, src, f), os.path.join(root, f))
        json.dump({"source": src, "memory": sel[mode]}, open(os.path.join(root, "memory_config.json"), "w"), indent=1, default=float)
        with Pool(8) as pool:
            pool.map(build_one, [(e, src, mode, sel[mode]) for e in eps])
        print(mode, len(os.listdir(os.path.join(root, "preds"))), "episodes", flush=True)


def evaluate(src: str, config: str, out: str, report: str) -> None:
    sel = json.load(open(config))["selected"]
    res = {}
    for mode in MODES:
        cfg = MemoryConfig(**sel[mode])
        res[mode] = {"config": sel[mode], "test": metrics(collect(sum((split_ids(s) for s in ("test_a", "test_b", "test_c")), []), src, cfg))}
        for s in ("test_a", "test_b", "test_c"):
            res[mode][s] = metrics(collect(split_ids(s), src, cfg))
        print(mode, round(res[mode]["test"]["objective_stale_nll"], 4), flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
    write_report(res, report)


def write_report(res: dict, path: str) -> None:
    f = lambda v, p=3: "–" if v is None else f"{v:.{p}f}"  # noqa: E731
    names = {"none": "1. no memory", "hold": "2. hard hold", "decay": "3. hold + decay", "decay_geo": "4. hold + decay + geometry"}
    L = ["# Temporal interaction memory: stage-1 beliefs by time since last seen", "",
         "Test A+B+C. Pair relations (5 classes) and person cues; `visible` = jointly observed now. Settings selected on validation.", ""]
    for metric, lab in (("inter_f1", "interaction F1"), ("nll", "relation NLL"), ("pos_recall", "recall of true interactions"),
                        ("fp_rate", "false-positive rate"), ("ece", "interaction ECE"), ("halt_f1", "raised-hand F1"), ("wheel_f1", "wheelchair F1")):
        L += [f"## {lab}", "", "| Memory | " + " | ".join(b[0] for b in BUCKETS) + " |", "|---|" + "---|" * len(BUCKETS)]
        for mode in MODES:
            t = res[mode]["test"]
            L.append(f"| {names[mode]} | " + " | ".join(f(t.get(b[0], {}).get(metric)) for b in BUCKETS) + " |")
        L.append("")
    L += ["## Persistence and re-entry", "", "| Memory | ended while unobserved, still predicted | stale false-positive rate | F1 after re-entry | geometry resets |",
          "|---|---|---|---|---|"]
    for mode in MODES:
        t = res[mode]["test"]
        L.append(f"| {names[mode]} | {f(t['ended_while_unobserved']['still_predicted_interacting'])} ({t['ended_while_unobserved']['pairs']}) | "
                 f"{f(t['stale_false_positive_persistence'])} | {f(t['after_reentry'].get('inter_f1'))} | {t['geo_resets']} |")
    L += ["", "Pair counts per bucket (no memory): " + ", ".join(f"{b[0]} {res['none']['test'].get(b[0], {}).get('pairs', 0)}" for b in BUCKETS), ""]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("tune", "build", "eval"))
    ap.add_argument("--src", default="cffullraw")
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--modes", nargs="+", default=list(MODES[1:]))
    a = ap.parse_args()
    if a.cmd == "tune":
        tune(a.src, a.out)
    elif a.cmd == "build":
        build(a.src, a.config, a.modes)
    else:
        evaluate(a.src, a.config, a.out, a.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
