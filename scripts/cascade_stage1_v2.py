#!/usr/bin/env python3
"""Stage 1 v2: explicit social groups (GroupNet, socialcost/bench/group_net.py) with ensembles and out-of-fold export.

    $PY cascade_stage1_v2.py --variant {tracks2,vis2} --out data/costmap_bench/stage1/<variant> [--hist 40] [--no-oof]

Same protocol as v1 (cascade_stage1.py): M members on all train episodes, temperatures (same-group, type, node) fitted on
validation, K out-of-fold ensembles for the train episodes, export of every sample of all 600 episodes. The export
decodes explicit groups (perturb-and-MAP correlation clustering) and writes
  <out>/preds/<episode>.npz       grouped marginal in the v1 format (+ MAP partition, its types, P(same))
  <out>raw/preds/<episode>.npz    factorised marginal without grouping (ablation input for stage 2)
Metrics (metrics.json, both exports): v1 pair metrics (comparable with v1), group detection (F1 at member IoU >= 0.5
and exact, typed), halt F1 on visible people, ECE, Test D.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from cascade_stage1 import f1, test_d  # noqa: E402
from socialcost.bench.data import ROOT, split_ids  # noqa: E402
from socialcost.bench.group_net import (N_TYPES, GroupNet, Stage1Samples2, collate2, decode_groups,  # noqa: E402
                                        group_loss)
from socialcost.bench.interaction_net import CLASSES, type_class  # noqa: E402
from socialcost.bench import counterfactual as cfm  # noqa: E402


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


REG = {"dropout": 0.0, "wd": 1e-4, "d": 96, "layers": 3}
CF = {"mode": "none", "batch": 16, "margin": 2.0, "w_rank": 0.5, "w_inv": 1.0, "cost": 0.0}
_CRITIC = {}


def cost_term(b, out, dev):
    """Cost-aware loss through the frozen TypeMix renderer (socialcost/bench/cost_aware.py), weighted by CF['cost']."""
    if CF["cost"] <= 0:
        return None
    if "m" not in _CRITIC:
        from socialcost.bench.cost_aware import CostCritic
        _CRITIC["m"] = CostCritic().to(dev)
    return _CRITIC["m"](b, *out)


def train_member(ds, idx, crops, seed, steps, batch, dev, w, workers, log, cf_ds=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GroupNet(crops, d=REG["d"], layers=REG["layers"], dropout=REG["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=REG["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    dl = DataLoader(Subset(ds, idx), batch_size=batch, shuffle=True, drop_last=True, num_workers=workers, collate_fn=collate2,
                    persistent_workers=True, pin_memory=True)
    use_cf = cf_ds is not None and CF["mode"] != "none" and len(cf_ds)
    if use_cf:
        cf_dl = DataLoader(cf_ds, batch_size=CF["batch"], shuffle=True, drop_last=True, num_workers=max(2, workers // 2),
                           collate_fn=cfm.collate_cf, persistent_workers=True, pin_memory=True)
        cf_it = iter(cf_dl)
    step, t0, run = 0, time.time(), defaultdict(float)
    model.train()
    while step < steps:
        for b in dl:
            b = to_dev(b, dev)
            if use_cf:
                try:
                    on, off, obs = next(cf_it)
                except StopIteration:
                    cf_it = iter(cf_dl)
                    on, off, obs = next(cf_it)
                on, off, obs = to_dev(on, dev), to_dev(off, dev), obs.to(dev)
                plan = cfm.cf_plan(on, off, obs, crops)
                if CF["mode"] == "nohidden":
                    for k in ("pair", "node"):
                        plan[f"{k}_inv"] = plan[f"{k}_inv"] & ~plan[f"{k}_soft"]
                else:
                    cfm.apply_soft_targets(on, off, plan)
                nb, nc = b["mask"].shape[0], on["mask"].shape[0]
                full = cfm.concat(b, on, off)
                with torch.autocast("cuda", dtype=torch.float16):
                    out_all = model(full)
                loss, parts = group_loss(*out_all, full, w)
                o_on = tuple(x[nb:nb + nc] for x in out_all)
                o_off = tuple(x[nb + nc:] for x in out_all)
                l_cf, p_cf = cfm.cf_loss(o_on, o_off, on, off, plan, CF["margin"], 0.0 if CF["mode"] == "norank" else CF["w_rank"],
                                         0.0 if CF["mode"] == "noinv" else CF["w_inv"])
                loss = loss + l_cf
                parts |= p_cf
                lc = cost_term(b, tuple(x[:nb] for x in out_all), dev)
                if lc is not None:
                    loss = loss + CF["cost"] * lc
                    parts["cost"] = float(lc)
            else:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(b)
                loss, parts = group_loss(*out, b, w)
                lc = cost_term(b, out, dev)
                if lc is not None:
                    loss = loss + CF["cost"] * lc
                    parts["cost"] = float(lc)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            for k, v in parts.items():
                run[k] += v
            if step % 500 == 0:
                log(f"  seed {seed} step {step} " + " ".join(f"{k} {v / 500:.3f}" for k, v in run.items()) + f" ({time.time() - t0:.0f} s)")
                run.clear()
            if step >= steps:
                break
    model.eval()
    return model


@torch.no_grad()
def logits_of(models, ds, idx, batch, dev, workers) -> dict:
    """{(episode, t): (same [M,P,P], type [M,P,P,4], node [M,P,2])} cropped to the real number of people."""
    dl = DataLoader(Subset(ds, idx), batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate2)
    out = {}
    for b in dl:
        keys = b["key"]
        n_people = b["mask"].sum(1).tolist()
        b = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.float16):
            res = [m(b) for m in models]
        S = torch.stack([r[0].float() for r in res], 1).cpu().numpy()
        Ty = torch.stack([r[1].float() for r in res], 1).cpu().numpy()
        N = torch.stack([r[2].float() for r in res], 1).cpu().numpy()
        for i, k in enumerate(keys):
            P = int(n_people[i])
            out[k] = (S[i, :, :P, :P].astype(np.float16), Ty[i, :, :P, :P].astype(np.float16), N[i, :, :P].astype(np.float16))
    return out


def fit_temperatures(lg: dict, ds) -> dict:
    """Grid search of the three temperatures on validation (NLL): same-group (binary), type (on true group pairs), node."""
    by = {"same": [], "type": [], "node": []}
    for (e, t), (S, Ty, N) in lg.items():
        c = ds.cache[e]
        P = c.P
        gid = c.inter[t]
        sm = (gid[:, None] >= 0) & (gid[:, None] == gid[None, :])
        iu = np.triu_indices(P, 1)
        by["same"].append((S[:, iu[0], iu[1]].astype(np.float32), sm[iu]))
        tm = sm[iu]
        if tm.any():
            y = np.array([type_class(x) - 1 for x in c.itype[t]])[iu[0]][tm]
            by["type"].append((Ty[:, iu[0], iu[1]][:, tm].astype(np.float32), y))
        by["node"].append((N.astype(np.float32), np.stack([c.halt[t], c.wheel], -1)))
    grid = np.arange(0.5, 3.01, 0.1)
    best = {}
    for kind, rows in by.items():
        sc = {}
        for T in grid:
            nll, n = 0.0, 0
            for z, y in rows:
                if kind == "type":
                    zz = z / T
                    zz = zz - zz.max(-1, keepdims=True)
                    lp = zz - np.log(np.exp(zz).sum(-1, keepdims=True))
                    p = np.exp(lp).mean(0)
                    nll -= np.log(p[np.arange(len(y)), y].clip(1e-8)).sum()
                    n += len(y)
                else:
                    p = (1 / (1 + np.exp(-z / T))).mean(0)
                    y = y.astype(np.float32)
                    nll -= (y * np.log(p.clip(1e-8)) + (1 - y) * np.log((1 - p).clip(1e-8))).sum()
                    n += y.size
            sc[round(float(T), 2)] = nll / max(n, 1)
        best[kind] = min(sc, key=sc.get)
    return best


_DEC = {}


def _decode_episode(args):
    e, rows = args
    T = _DEC["T"]
    rng = np.random.default_rng(int(hashlib.sha1(e.encode()).hexdigest(), 16) % 2 ** 32)
    out = {}
    for t, (S, Ty, N) in rows.items():
        d = decode_groups(list(S), list(Ty), T["same"], T["type"], _DEC["spm"], rng)
        pn = 1 / (1 + np.exp(-N.astype(np.float64) / T["node"]))  # M, P, 2
        pnm = pn.mean(0)

        def bh(p):
            p = p.clip(1e-6, 1 - 1e-6)
            return -(p * np.log(p) + (1 - p) * np.log(1 - p))

        d["node_p"] = pnm
        d["node_mi"] = np.clip(bh(pnm) - bh(pn).mean(0), 0, None)
        out[t] = d
    return e, out


def decode_all(lg: dict, temps: dict, spm: int, procs: int) -> dict:
    by_ep = defaultdict(dict)
    for (e, t), v in lg.items():
        by_ep[e][t] = v
    _DEC.update({"T": temps, "spm": spm})
    sink = {}
    with Pool(procs) as pool:
        for e, out in pool.imap_unordered(_decode_episode, by_ep.items()):
            sink[e] = out
    return sink


def write_preds(sink: dict, out_dir: str, raw: bool) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pre = "raw_" if raw else ""
    for e, rows in sink.items():
        T = max(rows) + 1
        P = rows[min(rows)]["edge_p"].shape[0]
        arr = {"edge_p": np.zeros((T, P, P, 5), np.float16), "edge_mi": np.zeros((T, P, P), np.float16),
               "edge_entropy": np.zeros((T, P, P), np.float16), "node_p": np.zeros((T, P, 2), np.float16), "node_mi": np.zeros((T, P, 2), np.float16)}
        if not raw:
            arr |= {"map_group": -np.ones((T, P), np.int8), "map_type": -np.ones((T, P), np.int8), "p_same": np.zeros((T, P, P), np.float16)}
        for t, r in rows.items():
            arr["edge_p"][t] = r[pre + "edge_p"]
            arr["edge_mi"][t] = r[pre + "edge_mi"]
            arr["edge_entropy"][t] = r[pre + "edge_entropy"]
            arr["node_p"][t] = r["node_p"]
            arr["node_mi"][t] = r["node_mi"]
            if not raw:
                arr["map_group"][t] = r["map_group"]
                arr["map_type"][t] = r["map_type"]
                arr["p_same"][t] = r["p_same"]
        np.savez_compressed(os.path.join(out_dir, f"{e}.npz"), **arr)


def groups_of(lab) -> list[set]:
    g = defaultdict(set)
    for i, x in enumerate(lab):
        if x >= 0:
            g[int(x)].add(i)
    return [s for s in g.values() if len(s) >= 2]


def evaluate(sink: dict, ds, split_of: dict, raw: bool) -> dict:
    pre = "raw_" if raw else ""
    agg = defaultdict(lambda: defaultdict(float))
    bins = defaultdict(lambda: np.zeros((10, 3)))
    for e, rows in sink.items():
        g = split_of[e]
        c = ds.cache[e]
        P = c.P
        iu = np.triu_indices(P, 1)
        for t, r in rows.items():
            gid = c.inter[t]
            tcls = np.array([type_class(x) for x in c.itype[t]])
            sm = (gid[:, None] >= 0) & (gid[:, None] == gid[None, :])
            y = np.where(sm, tcls[:, None], 0)[iu]
            ep = r[pre + "edge_p"]
            yp = ep.argmax(-1)[iu]
            for k in range(1, len(CLASSES)):
                agg[g][f"tp{k}"] += ((yp == k) & (y == k)).sum()
                agg[g][f"fp{k}"] += ((yp == k) & (y != k)).sum()
                agg[g][f"fn{k}"] += ((yp != k) & (y == k)).sum()
            pint = 1 - ep[..., 0][iu]
            ys = (y > 0).astype(float)
            agg[g]["inter_tp"] += ((pint > 0.5) & (ys > 0)).sum()
            agg[g]["inter_fp"] += ((pint > 0.5) & (ys == 0)).sum()
            agg[g]["inter_fn"] += ((pint <= 0.5) & (ys > 0)).sum()
            b = np.clip((pint * 10).astype(int), 0, 9)
            np.add.at(bins[g], b, np.stack([np.ones_like(pint), pint, ys], 1))
            for k, lab in ((0, c.halt[t]), (1, np.asarray(c.wheel))):
                pp, yy = r["node_p"][:, k] > 0.5, lab > 0.5
                agg[g][f"n{k}_tp"] += (pp & yy).sum()
                agg[g][f"n{k}_fp"] += (pp & ~yy).sum()
                agg[g][f"n{k}_fn"] += (~pp & yy).sum()
                if k == 0:
                    v = c.vis[t] > 0.3
                    agg[g]["hv_tp"] += (pp & yy & v).sum()
                    agg[g]["hv_fp"] += (pp & ~yy & v).sum()
                    agg[g]["hv_fn"] += (~pp & yy & v).sum()
            if not raw:  # explicit group detection (MAP partition)
                gt = groups_of(gid)
                gtt = {frozenset(s): int(tcls[min(s)]) - 1 for s in gt}
                pr = groups_of(r["map_group"])
                prt = {frozenset(s): int(r["map_type"][min(s)]) for s in pr}
                for thr, tag in ((0.5, "g50"), (1.0, "gex")):
                    used = set()
                    for s in pr:
                        best, bi = 0.0, None
                        for j, q in enumerate(gt):
                            if j in used:
                                continue
                            iou = len(s & q) / len(s | q)
                            if iou > best:
                                best, bi = iou, j
                        if bi is not None and best >= thr - 1e-9:
                            used.add(bi)
                            agg[g][f"{tag}_tp"] += 1
                            agg[g][f"{tag}t_tp"] += prt[frozenset(s)] == gtt[frozenset(gt[bi])]
                        else:
                            agg[g][f"{tag}_fp"] += 1
                    agg[g][f"{tag}_fn"] += len(gt) - len(used)
                    agg[g][f"{tag}_np"] += len(pr)
                    agg[g][f"{tag}_ng"] += len(gt)
    res = {}
    for g, a in agg.items():
        r = {f"f1_{CLASSES[k]}": f1(a[f"tp{k}"], a[f"fp{k}"], a[f"fn{k}"]) for k in range(1, len(CLASSES))}
        present = [k for k in range(1, len(CLASSES)) if a[f"tp{k}"] + a[f"fn{k}"] > 0]
        r["macro_f1"] = float(np.mean([r[f"f1_{CLASSES[k]}"] for k in present])) if present else None
        r["f1_interacting"] = f1(a["inter_tp"], a["inter_fp"], a["inter_fn"])
        r["f1_halt"] = f1(a["n0_tp"], a["n0_fp"], a["n0_fn"])
        r["f1_halt_visible"] = f1(a["hv_tp"], a["hv_fp"], a["hv_fn"])
        r["f1_wheelchair"] = f1(a["n1_tp"], a["n1_fp"], a["n1_fn"])
        B = bins[g]
        r["ece_interacting"] = float(np.abs(B[:, 1] - B[:, 2]).sum() / max(B[:, 0].sum(), 1))
        if not raw:
            for tag in ("g50", "gex"):
                r[f"group_f1_{tag}"] = f1(a[f"{tag}_tp"], a[f"{tag}_fp"], a[f"{tag}_fn"])
                r[f"group_typed_f1_{tag}"] = f1(a[f"{tag}t_tp"], a[f"{tag}_np"] - a[f"{tag}t_tp"], a[f"{tag}_ng"] - a[f"{tag}t_tp"])
        res[g] = {k: (float(v) if v is not None else None) for k, v in r.items()}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("tracks2", "vis2"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hist", type=int, default=40, help="history samples at 5 Hz (40 = 8 s)")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--samples-per-member", type=int, default=8)
    ap.add_argument("--decode-procs", type=int, default=8)
    ap.add_argument("--no-oof", action="store_true", help="stage-1 metrics only (no train-episode export)")
    ap.add_argument("--dropout", type=float, default=0.0, help="v3: 0.2")
    ap.add_argument("--wd", type=float, default=1e-4, help="v3: 0.01")
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--cf", choices=("none", "full", "noinv", "norank", "nohidden"), default="none",
                    help="counterfactual supervision on train ON/OFF pairs (socialcost/bench/counterfactual.py)")
    ap.add_argument("--cf-batch", type=int, default=16)
    ap.add_argument("--cost-weight", type=float, default=0.0, help="cost-aware loss through the frozen TypeMix renderer")
    ap.add_argument("--augment", action="store_true", help="v3: mirror, position / velocity noise, history dropout (training only)")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "train.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    dev = torch.device("cuda")
    crops = args.variant == "vis2"
    man = {e["episode_id"]: e for e in json.load(open(os.path.join(ROOT, "manifests", "manifest_v1.json")))["episodes"]}
    splits = {s: split_ids(s) for s in ("train", "val", "test_a", "test_b", "test_c")}
    split_of = {e: s for s, v in splits.items() for e in v}
    t0 = time.time()
    ds = Stage1Samples2([e for v in splits.values() for e in v], args.hist, crops)
    REG.update({"dropout": args.dropout, "wd": args.wd, "d": args.d, "layers": args.layers})
    CF.update({"mode": args.cf, "batch": args.cf_batch, "cost": args.cost_weight})

    def cf_of(ids: set):
        if args.cf == "none":
            return None
        c = cfm.CFPairs(cfm.train_pairs(ids), ds.cache, args.hist, crops)
        log(f"counterfactual pairs: {len(c)} matched samples, {c.skipped} units skipped")
        return c
    ds_tr = Stage1Samples2([e for v in splits.values() for e in v], args.hist, crops, augment=True, cache=ds.cache) if args.augment else ds
    assert ds_tr.items == ds.items
    log(f"dataset {len(ds)} samples, load {time.time() - t0:.0f} s")
    tr_idx = [i for i, (e, t) in enumerate(ds.items) if split_of[e] == "train" and t % 2 == 0]
    # loss weights from train frequencies
    same_pos = same_n = halt_pos = wheel_pos = nodes = 0
    tcnt = np.zeros(N_TYPES)
    for e in splits["train"]:
        c = ds.cache[e]
        if not c.P:
            continue
        for t in range(0, c.T, 2):
            gid = c.inter[t]
            sm = (gid[:, None] >= 0) & (gid[:, None] == gid[None, :]) & ~np.eye(c.P, dtype=bool)
            same_pos += sm.sum()
            same_n += c.P * (c.P - 1)
            for a, b in zip(*np.nonzero(np.triu(sm, 1))):
                tcnt[type_class(c.itype[t][a]) - 1] += 1
            halt_pos += c.halt[t].sum()
            wheel_pos += c.wheel.sum()
            nodes += c.P
    tw = (tcnt.sum() / np.maximum(tcnt, 1)) ** 0.5
    w = {"same_pos": torch.tensor(((same_n - same_pos) / max(same_pos, 1)) ** 0.5, device=dev),
         "type": torch.tensor(tw / tw.mean(), dtype=torch.float32, device=dev),
         "node_pos": torch.tensor([((nodes - halt_pos) / max(halt_pos, 1)) ** 0.5, ((nodes - wheel_pos) / max(wheel_pos, 1)) ** 0.5], dtype=torch.float32, device=dev)}
    info = {"variant": args.variant, "args": vars(args), "type_counts": tcnt.tolist(), "weights": {k: v.tolist() for k, v in w.items()}}
    log(json.dumps(info))

    def idx_of(ids: set, stride: int) -> list[int]:
        return [i for i, (e, t) in enumerate(ds.items) if e in ids and t % stride == 0]

    cf_full = cf_of(set(splits["train"]))
    full = [train_member(ds_tr, tr_idx, crops, 100 + m, args.steps, args.batch, dev, w, args.workers, log, cf_full) for m in range(args.members)]
    for m, model in enumerate(full):
        torch.save(model.state_dict(), os.path.join(out, f"full_m{m}.pt"))
    lg = logits_of(full, ds, idx_of(set(splits["val"]), 1), 128, dev, args.workers)
    temps = fit_temperatures(lg, ds)
    info.update({"temperatures": temps, "full_train_s": time.time() - t0})
    log(f"temperatures {temps}")
    for s in ("test_a", "test_b", "test_c"):
        lg |= logits_of(full, ds, idx_of(set(splits[s]), 1), 128, dev, args.workers)
    if not args.no_oof:
        fold_of = {e: int(hashlib.sha1(man[e]["unit_id"].encode()).hexdigest(), 16) % args.folds for e in splits["train"]}
        for f in range(args.folds):
            trn = {e for e in splits["train"] if fold_of[e] != f}
            ho = {e for e in splits["train"] if fold_of[e] == f}
            tri = [i for i, (e, t) in enumerate(ds.items) if e in trn and t % 2 == 0]
            cf_f = cf_of(trn)
            models = [train_member(ds_tr, tri, crops, 200 + 10 * f + m, args.steps, args.batch, dev, w, args.workers, log, cf_f)
                      for m in range(args.members)]
            lg |= logits_of(models, ds, idx_of(ho, 1), 128, dev, args.workers)
            log(f"fold {f} done ({len(ho)} episodes)")
            del models
            torch.cuda.empty_cache()
    t1 = time.time()
    sink = decode_all(lg, temps, args.samples_per_member, args.decode_procs)
    log(f"decoded {len(lg)} samples in {time.time() - t1:.0f} s")
    raw_out = out + "raw"
    if not args.no_oof:
        write_preds(sink, os.path.join(out, "preds"), raw=False)
        write_preds(sink, os.path.join(raw_out, "preds"), raw=True)
    ev = {e: r for e, r in sink.items() if split_of[e] != "train"}
    held = cfm.train_pairs({e for s_ in ("val", "test_a", "test_b", "test_c") for e in splits[s_]})
    metrics = {"stage1": evaluate(ev, ds, split_of, raw=False), "test_d": test_d(sink, ds.cache), "cf_test_d": cfm.cf_test_d(sink, ds.cache, held, raw=False),
               "info": {**info, "total_s": time.time() - t0}}
    raw_sink = {e: {t: {**r, "edge_p": r["raw_edge_p"]} for t, r in rows.items()} for e, rows in sink.items()}
    metrics_raw = {"stage1": evaluate(ev, ds, split_of, raw=True), "test_d": test_d(raw_sink, ds.cache), "cf_test_d": cfm.cf_test_d(sink, ds.cache, held, raw=True),
                   "info": {**info, "export": "raw (no grouping)"}}
    os.makedirs(raw_out, exist_ok=True)
    json.dump(metrics_raw, open(os.path.join(raw_out, "metrics.json"), "w"), indent=1)
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    log(json.dumps({s: metrics["stage1"].get(s) for s in ("test_a", "test_b", "test_c")}, indent=1))
    log(json.dumps(metrics["test_d"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
