#!/usr/bin/env python3
"""Stage 1 of the cascade: interaction / gesture / mobility-aid predictor ensembles with out-of-fold predictions.

    $PY cascade_stage1.py --variant {tracks,vis} --out data/costmap_bench/stage1/<variant>

1. full ensemble: M members trained on all train episodes; temperatures (edge, node) fitted on validation (grid search, NLL)
2. out-of-fold: train episodes split into K folds by unit (both conditions of a pair stay together); per fold M members
   trained on the other folds predict the held-out fold, so stage 2 is trained on predictions from models that never
   saw those episodes
3. export: every sample of all 600 episodes -> preds/<episode>.npz (edge_p, edge_mi, edge_entropy, node_p, node_mi);
   train episodes from their fold ensemble, val / test from the full ensemble
4. stage-1 metrics on Test A/B/C and Test D (metrics.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.bench.data import ROOT, split_ids, test_d_pairs  # noqa: E402
from socialcost.bench.interaction_net import (CLASSES, InteractionNet, Stage1Samples, collate1, ensemble_stats,  # noqa: E402
                                              stage1_loss)


def train_member(ds, idx, with_crops, seed, steps, batch, dev, class_w, pos_w, workers):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = InteractionNet(with_crops).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    dl = DataLoader(Subset(ds, idx), batch_size=batch, shuffle=True, drop_last=True, num_workers=workers, collate_fn=collate1, persistent_workers=True)
    step = 0
    model.train()
    while step < steps:
        for b in dl:
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            el, nl = model(b)
            loss = stage1_loss(el, nl, b, class_w, pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step >= steps:
                break
    model.eval()
    return model


@torch.no_grad()
def logits_of(models, ds, idx, batch, dev, workers):
    dl = DataLoader(Subset(ds, idx), batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate1)
    out = []
    for b in dl:
        keys = b["key"]
        b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        res = [m(b) for m in models]
        out.append((keys, [r[0].float().cpu() for r in res], [r[1].float().cpu() for r in res], b["ecls"].cpu(), b["halt"].cpu(), b["wheel"].cpu(), b["mask"].cpu()))
    return out


def fit_temperature(chunks) -> tuple[float, float]:
    best = {}
    for kind in ("edge", "node"):
        scores = {}
        for T in np.arange(0.6, 3.01, 0.1):
            nll, n = 0.0, 0
            for keys, el, nl, ecls, halt, wheel, mask in chunks:
                if kind == "edge":
                    p = torch.stack([F.softmax(z / T, -1) for z in el]).mean(0)
                    m = ecls >= 0
                    nll += float(-(p[m].gather(1, ecls[m].unsqueeze(1)).clamp(min=1e-8).log()).sum())
                    n += int(m.sum())
                else:
                    p = torch.stack([torch.sigmoid(z / T) for z in nl]).mean(0)
                    y = torch.stack([halt, wheel], -1)
                    mm = mask.unsqueeze(-1).expand_as(y)
                    nll += float(-(y * p.clamp(min=1e-8).log() + (1 - y) * (1 - p).clamp(min=1e-8).log())[mm].sum())
                    n += int(mm.sum())
            scores[round(float(T), 2)] = nll / max(n, 1)
        best[kind] = min(scores, key=scores.get)
    return best["edge"], best["node"]


def export(chunks, t_edge, t_node, sink: dict) -> None:
    for keys, el, nl, *_ in chunks:
        st = ensemble_stats(el, nl, t_edge, t_node)
        for i, (e, t) in enumerate(keys):
            sink.setdefault(e, {})[t] = {k: v[i].numpy() for k, v in st.items()}


def write_preds(sink: dict, out_dir: str, ds_lookup) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for e, rows in sink.items():
        P = ds_lookup[e].P
        T = max(rows) + 1
        arr = {"edge_p": np.zeros((T, P, P, len(CLASSES)), np.float16), "edge_mi": np.zeros((T, P, P), np.float16),
               "edge_entropy": np.zeros((T, P, P), np.float16), "node_p": np.zeros((T, P, 2), np.float16), "node_mi": np.zeros((T, P, 2), np.float16)}
        for t, r in rows.items():
            arr["edge_p"][t] = r["edge_p"][:P, :P]
            arr["edge_mi"][t] = r["edge_mi"][:P, :P]
            arr["edge_entropy"][t] = r["edge_entropy"][:P, :P]
            arr["node_p"][t] = r["node_p"][:P]
            arr["node_mi"][t] = r["node_mi"][:P]
        np.savez_compressed(os.path.join(out_dir, f"{e}.npz"), **arr)


def f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def evaluate(chunks, t_edge, t_node, split_of) -> dict:
    """Pair classification (each unordered pair once), node detection, calibration of p(interacting)."""
    agg = defaultdict(lambda: defaultdict(float))
    bins = defaultdict(lambda: np.zeros((10, 3)))
    for keys, el, nl, ecls, halt, wheel, mask in chunks:
        st = ensemble_stats(el, nl, t_edge, t_node)
        pe, pn = st["edge_p"], st["node_p"]
        pred = pe.argmax(-1)
        P = pe.shape[1]
        iu = torch.triu_indices(P, P, 1)
        for i, (e, t) in enumerate(keys):
            g = split_of[e]
            y = ecls[i][iu[0], iu[1]]
            yp = pred[i][iu[0], iu[1]]
            ok = y >= 0
            y, yp = y[ok], yp[ok]
            pint = 1 - pe[i][iu[0], iu[1], 0][ok]
            for c in range(1, len(CLASSES)):
                agg[g][f"tp{c}"] += float(((yp == c) & (y == c)).sum())
                agg[g][f"fp{c}"] += float(((yp == c) & (y != c)).sum())
                agg[g][f"fn{c}"] += float(((yp != c) & (y == c)).sum())
            agg[g]["acc_n"] += float(len(y))
            agg[g]["acc_ok"] += float((yp == y).sum())
            b = np.clip((pint.numpy() * 10).astype(int), 0, 9)
            for bi, pi, yi in zip(b, pint.numpy(), (y > 0).numpy()):
                bins[g][bi] += (1, pi, yi)
            m = mask[i]
            for k, lab in ((0, halt[i]), (1, wheel[i])):
                pp = pn[i, :, k][m] > 0.5
                yy = lab[m] > 0.5
                agg[g][f"n{k}_tp"] += float((pp & yy).sum())
                agg[g][f"n{k}_fp"] += float((pp & ~yy).sum())
                agg[g][f"n{k}_fn"] += float((~pp & yy).sum())
    res = {}
    for g, a in agg.items():
        r = {f"f1_{CLASSES[c]}": f1(a[f"tp{c}"], a[f"fp{c}"], a[f"fn{c}"]) for c in range(1, len(CLASSES))}
        present = [c for c in range(1, len(CLASSES)) if a[f"tp{c}"] + a[f"fn{c}"] > 0]
        r["macro_f1"] = float(np.mean([r[f"f1_{CLASSES[c]}"] for c in present])) if present else None
        r["pair_accuracy"] = a["acc_ok"] / max(a["acc_n"], 1)
        r["f1_halt"] = f1(a["n0_tp"], a["n0_fp"], a["n0_fn"])
        r["f1_wheelchair"] = f1(a["n1_tp"], a["n1_fp"], a["n1_fn"])
        B = bins[g]
        r["ece_interacting"] = float(sum(abs(B[k, 1] - B[k, 2]) for k in range(10)) / max(B[:, 0].sum(), 1))
        res[g] = r
    return res


def test_d(sink: dict, cache) -> dict:
    """Does the predicted social state differ between ON and OFF of a counterfactual pair? Mean over samples of the
    relevant probability for the controlled people (participants: talk / queue between them; primary: halt / wheelchair)."""
    man = {e["episode_id"]: e for e in json.load(open(os.path.join(ROOT, "manifests", "manifest_v1.json")))["episodes"]}
    out = defaultdict(lambda: {"on": [], "off": []})
    for pr in test_d_pairs():
        fam = pr["family"]
        for cond in ("on", "off"):
            eid = pr[cond]
            if eid not in sink:
                continue
            names = list(cache[eid].names) if hasattr(cache[eid], "names") else None
            hum = man[eid]["humans"]
            idx = {h["name"]: i for i, h in enumerate(hum)}
            ctrl = [idx[h["name"]] for h in hum if h["role"] in ("primary", "participant", "member")]
            vals = []
            for t, r in sink[eid].items():
                if fam in ("conversation", "queue"):
                    c = 1 if fam == "conversation" else 2
                    ps = [r["edge_p"][a, b, c] for a in ctrl for b in ctrl if a < b]
                    vals.append(float(np.mean(ps)) if ps else 0.0)
                elif fam == "stop_gesture":
                    vals.append(float(np.mean([r["node_p"][a, 0] for a in ctrl])))
                elif fam == "wheelchair":
                    vals.append(float(np.mean([r["node_p"][a, 1] for a in ctrl])))
            out[f"{fam}|{pr['observability']}"][cond].append(float(np.mean(vals)) if vals else None)
    return {k: {"p_on": float(np.mean([x for x in v["on"] if x is not None])), "p_off": float(np.mean([x for x in v["off"] if x is not None])),
                "pairs": len(v["on"])} for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("tracks", "vis"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dev = torch.device("cuda")
    crops = args.variant == "vis"
    os.makedirs(args.out, exist_ok=True)
    man = {e["episode_id"]: e for e in json.load(open(os.path.join(ROOT, "manifests", "manifest_v1.json")))["episodes"]}
    splits = {s: split_ids(s) for s in ("train", "val", "test_a", "test_b", "test_c")}
    split_of = {e: s for s, v in splits.items() for e in v}
    all_ids = [e for v in splits.values() for e in v]
    t0 = time.time()
    ds = Stage1Samples(all_ids, stride=1, with_crops=crops)
    ds_train2 = [i for i, (e, t) in enumerate(ds.items) if split_of[e] == "train" and t % 2 == 0]
    # class weights ~ frequency^-0.5 (train), positive weights for the node heads
    cnt = np.zeros(len(CLASSES))
    halt_pos = wheel_pos = nodes = 0
    for i in ds_train2[::10]:
        e, t = ds.items[i]
        b = ds[i]
        y = b["ecls"][b["ecls"] >= 0].numpy()
        cnt += np.bincount(y, minlength=len(CLASSES))
        m = b["mask"].numpy()
        halt_pos += b["halt"].numpy()[m].sum()
        wheel_pos += b["wheel"].numpy()[m].sum()
        nodes += m.sum()
    cw = (cnt.sum() / np.maximum(cnt, 1)) ** 0.5
    class_w = torch.tensor(cw / cw.mean(), dtype=torch.float32, device=dev)
    pos_w = torch.tensor([((nodes - halt_pos) / max(halt_pos, 1)) ** 0.5, ((nodes - wheel_pos) / max(wheel_pos, 1)) ** 0.5], dtype=torch.float32, device=dev)
    info = {"variant": args.variant, "class_counts": cnt.tolist(), "class_weights": class_w.tolist(), "pos_weights": pos_w.tolist(), "args": vars(args)}
    print("class counts", cnt, "weights", class_w.tolist(), "pos_w", pos_w.tolist(), flush=True)

    def idx_of(ids: set, stride: int = 2) -> list[int]:
        return [i for i, (e, t) in enumerate(ds.items) if e in ids and t % stride == 0]

    train_ids = set(splits["train"])
    full = [train_member(ds, idx_of(train_ids), crops, 100 + m, args.steps, args.batch, dev, class_w, pos_w, args.workers) for m in range(args.members)]
    for m, model in enumerate(full):
        torch.save(model.state_dict(), os.path.join(args.out, f"full_m{m}.pt"))
    val_chunks = logits_of(full, ds, idx_of(set(splits["val"]), 1), 128, dev, args.workers)
    t_edge, t_node = fit_temperature(val_chunks)
    info.update({"t_edge": t_edge, "t_node": t_node, "full_train_s": time.time() - t0})
    print("temperatures", t_edge, t_node, flush=True)
    sink: dict = {}
    eval_chunks = []
    for s in ("val", "test_a", "test_b", "test_c"):
        ch = val_chunks if s == "val" else logits_of(full, ds, idx_of(set(splits[s]), 1), 128, dev, args.workers)
        export(ch, t_edge, t_node, sink)
        eval_chunks += ch
    # out-of-fold predictions for the train episodes (folds by unit)
    fold_of = {e: int(hashlib.sha1(man[e]["unit_id"].encode()).hexdigest(), 16) % args.folds for e in splits["train"]}
    for f in range(args.folds):
        tr = {e for e in splits["train"] if fold_of[e] != f}
        ho = {e for e in splits["train"] if fold_of[e] == f}
        models = [train_member(ds, idx_of(tr), crops, 200 + 10 * f + m, args.steps, args.batch, dev, class_w, pos_w, args.workers)
                  for m in range(args.members)]
        export(logits_of(models, ds, idx_of(ho, 1), 128, dev, args.workers), t_edge, t_node, sink)
        print(f"fold {f} done ({len(ho)} episodes)", flush=True)
    write_preds(sink, os.path.join(args.out, "preds"), ds.cache)
    metrics = {"stage1": evaluate(eval_chunks, t_edge, t_node, split_of), "test_d": test_d(sink, ds.cache), "info": {**info, "total_s": time.time() - t0}}
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    print(json.dumps(metrics["stage1"], indent=1)[:3000])
    print(json.dumps(metrics["test_d"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
