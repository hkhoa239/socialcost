#!/usr/bin/env python3
"""Costmap prediction benchmark: train one model on dataset_v1, select on validation, evaluate on Test A/B/C/D.

    python bench_train.py --model {rgb,depth,rgbd,pos,pos_int,sngnn_priv,sngnn_sensor} --label {native,membership}
                          --seed S --out DIR [--lr LR] [--samples N] [--batch B] [--overfit N] [--tune]

SOCIALCOST models reuse socialcost.learn.models (ImageToGrid / PositionToGrid) and the original weighted smooth-L1
loss on the full grid. SNGNN2D-v2 uses the official SELECT_GNN through socialcost.bench.sngnn_adapter with its
original MSE loss on [-1, 1] targets and Adam settings. Every model sees the same number of training samples
(--samples). Checkpoints are selected on validation only (score = MAE on nonzero cells + MAE on the social part
above the personal-space rule). Writes DIR/{config.json, log.jsonl, best.pt, metrics.json, predictions.npz}.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.bench.data import CASCADE_CHANNELS, POS_INT_CHANNELS, Dv1Samples, collate, split_ids, test_d_pairs  # noqa: E402
from socialcost.bench.metrics import Acc, counterfactual  # noqa: E402
from socialcost.learn.models import ImageToGrid, PositionToGrid, n_params  # noqa: E402

SOCIALCOST = ("rgb", "depth", "rgbd", "pos", "pos_int")
# probabilistic heads (mean + per-cell log-variance, beta-NLL): bounds and cascades with identical stage-2 architecture
PROB = ("pos_u", "pos_int_u", "cascade_tracks", "cascade_vis")
# TypeMix: typed parametric renderer marginalised over predicted types + residual U-Net (socialcost/bench/typemix.py)
TYPEMIX = ("typemix_tracks", "typemix_vis", "typemix_gt", "typemix_vis_nosem",
           "typemix_tracks2", "typemix_vis2", "typemix_vis2raw", "typemix_tracks3raw", "typemix_vis3raw", "typemix_cfnoneraw", "typemix_cffullraw", "typemix_cfcostraw",
           "typemix_cffullrawmemhold", "typemix_cffullrawmemdecay", "typemix_cffullrawmemdecay_geo")  # *mem<mode>: temporal memory (memory_stage1.py)  # *2: stage-1 v2 explicit groups (group_net.py), *2raw: v2 without grouping
PROB = PROB + TYPEMIX
BETA = 0.5  # beta-NLL (Seitzer et al. 2022)
LOGVAR_MIN, LOGVAR_MAX = -12.0, 0.0
SNGNN = ("sngnn_priv", "sngnn_sensor")
DEFAULT_LR = {"rgb": 1e-3, "depth": 1e-3, "rgbd": 1e-3, "pos": 1e-3, "pos_int": 1e-3, "sngnn_priv": 5e-5, "sngnn_sensor": 5e-5} | {m: 1e-3 for m in PROB}
DEFAULT_BATCH = {m: 16 for m in SOCIALCOST + PROB} | {m: 8 for m in SNGNN}
TEST = ("test_a", "test_b", "test_c")


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def variant(model: str) -> str:
    if model in TYPEMIX:
        return "_".join(model.split("_")[:2])  # typemix_vis_nosem uses the typemix_vis inputs
    return "graph" if model in SNGNN else ("pos" if model == "pos_u" else model)


def build_model(model: str, sample_graph=None):
    if model in ("rgb", "depth", "rgbd"):
        return ImageToGrid({"rgbd": 5, "depth": 2, "rgb": 3}[model])
    if model == "pos":
        return PositionToGrid()
    if model == "pos_int":
        return PositionToGrid(cin=POS_INT_CHANNELS)
    if model == "pos_u":
        return PositionToGrid(cin=3, out_ch=2)
    if model in ("pos_int_u", "cascade_tracks", "cascade_vis"):
        return PositionToGrid(cin=CASCADE_CHANNELS, out_ch=2)
    if model in TYPEMIX:
        from socialcost.bench.typemix import TypeMixCost
        return TypeMixCost(use_semantic_variance=not model.endswith("_nosem"))
    from socialcost.bench import sngnn_adapter as sa
    return sa.SNGNNCostmap(sample_graph.ndata["h"].shape[1], sa.num_rels())


def socialcost_loss(pred, target, rule):
    # original SOCIALCOST estimator loss (scripts/train_estimator.py), full grid
    w = 1.0 + 4.0 * (target > 0).float() + 10.0 * (target > rule + 0.5 / 252).float()
    return (w * torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=0.02)).mean()


def mean_logvar(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Probabilistic head: mean in [0, 1] (sigmoid) and log-variance (normalised cost units), clamped."""
    return torch.sigmoid(out[:, 0]), out[:, 1].float().clamp(LOGVAR_MIN, LOGVAR_MAX)


def beta_nll(mu, logvar, target, rule, beta: float = BETA):
    """beta-NLL with the SOCIALCOST cell weights: w * sg(sigma^(2 beta)) * [ (y - mu)^2 / (2 sigma^2) + log(sigma^2) / 2 ]
    (Kendall & Gal 2017 heteroscedastic Gaussian; Seitzer et al. 2022 beta weighting, beta = 0.5)."""
    w = 1.0 + 4.0 * (target > 0).float() + 10.0 * (target > rule + 0.5 / 252).float()
    var = logvar.exp()
    nll = 0.5 * (target - mu.float()) ** 2 / var + 0.5 * logvar
    return (w * var.detach() ** beta * nll).mean()


class Runner:
    def __init__(self, model_name: str, model, dev):
        self.name, self.model, self.dev = model_name, model, dev
        self.sngnn = model_name in SNGNN
        self.prob = model_name in PROB

    def to_dev(self, x: dict) -> dict:
        if self.sngnn:
            from socialcost.bench.sngnn_adapter import batch_graphs
            return {"graph": batch_graphs(x["graph"], self.dev)}
        return {k: v.to(self.dev, non_blocking=True) for k, v in x.items()}

    def predict(self, x: dict) -> torch.Tensor:
        """Prediction in [0, 1] on the 200 x 200 label grid."""
        return self.predict_dist(x)[0]

    def predict_dist(self, x: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(mean in [0, 1], sigma in [0, 1] or None for deterministic models)."""
        out = self.model(self.to_dev(x))
        if self.prob:
            mu, lv = mean_logvar(out)
            return mu, (0.5 * lv).exp()
        return (((out + 1) / 2).clamp(0, 1) if self.sngnn else out), None

    def loss(self, b: dict) -> torch.Tensor:
        out = self.model(self.to_dev(b["x"]))
        t = b["target"].to(self.dev)
        if self.sngnn:
            return torch.nn.functional.mse_loss(out, t * 2 - 1)  # official SNGNN2D-v2 loss and label scaling
        if self.prob:
            mu, lv = mean_logvar(out)
            return beta_nll(mu, lv, t, b["rule"].to(self.dev))
        return socialcost_loss(out, t, b["rule"].to(self.dev))


def groups_of(split: str, meta: dict) -> list[str]:
    return [split, f"{split}|family:{meta['family']}"]


@torch.no_grad()
def evaluate(runner: Runner, loader, split: str, keep: dict | None = None, tgt: dict | None = None, sig: dict | None = None) -> dict:
    runner.model.eval()
    acc = Acc()
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            p, sg = runner.predict_dist(b["x"])
        p = p.float().cpu()
        sg = sg.float().cpu() if sg is not None else None
        acc.add([groups_of(split, m) for m in b["meta"]], p, b["target"], b["rule"], b["fov"], b["inter"], sg)
        if keep is not None:
            pu = (p * 252).round().clamp(0, 252).to(torch.uint8).numpy()
            tu = (b["target"] * 252).round().to(torch.uint8).numpy()
            su = (sg * 252).round().clamp(0, 255).to(torch.uint8).numpy() if sg is not None else None
            for i, m in enumerate(b["meta"]):
                keep[(m["episode"], m["sample"])] = pu[i]
                tgt[(m["episode"], m["sample"])] = tu[i]
                if su is not None and sig is not None:
                    sig[(m["episode"], m["sample"])] = su[i]
    runner.model.train()
    return acc.result()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=SOCIALCOST + SNGNN + PROB)
    ap.add_argument("--score-fp-weight", type=float, default=0.0,
                    help="checkpoint selection adds w x whole-grid MAE (penalises painting cost everywhere; v2 protocol uses 2)")
    ap.add_argument("--label", required=True, choices=("native", "membership", "extended", "native_v2", "membership_v2", "extended_v2"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--samples", type=int, default=96000, help="training samples seen (budget, identical across models)")
    ap.add_argument("--train-stride", type=int, default=2)
    ap.add_argument("--val-stride", type=int, default=5)
    ap.add_argument("--test-stride", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=12000, help="validate every N training samples")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--tune", action="store_true", help="validation only, no test evaluation (hyperparameter selection)")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision (fp16 autocast + GradScaler, used for every model)")
    args = ap.parse_args()
    lr = args.lr or DEFAULT_LR[args.model]
    batch = args.batch or DEFAULT_BATCH[args.model]
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "metrics.json")):
        print("already done", args.out)
        return 0
    seed_all(args.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    amp = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    gb = None
    if args.model in SNGNN:
        from socialcost.bench.sngnn_adapter import GraphBuilder
        gb = GraphBuilder("privileged" if args.model == "sngnn_priv" else "sensor")
    v = variant(args.model)
    train = Dv1Samples(split_ids("train"), args.label, v, stride=args.train_stride, graph_builder=gb)
    val = Dv1Samples(split_ids("val"), args.label, v, stride=args.val_stride, eval_maps=True, graph_builder=gb)
    if args.overfit:
        idx = list(range(0, len(train), max(1, len(train) // args.overfit)))[: args.overfit]
        train = Subset(train, idx)
        val = Subset(Dv1Samples(split_ids("train"), args.label, v, stride=args.train_stride, eval_maps=True, graph_builder=gb), idx)
    model = build_model(args.model, train[0]["x"].get("graph") if gb else None).to(dev)
    runner = Runner(args.model, model, dev)
    if args.model in SNGNN:
        from socialcost.bench.sngnn_adapter import official_params
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=official_params()["weight_decay"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, args.samples // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if args.model in SOCIALCOST + PROB else None
    cfg = {**vars(args), "lr": lr, "batch": batch, "steps": steps, "params": n_params(model), "train_samples": len(train), "val_samples": len(val),
           "loss": "MSE on [-1,1] (official)" if args.model in SNGNN else ("beta-NLL (beta 0.5) x SOCIALCOST cell weights" if args.model in PROB else "weighted smooth-L1 (SOCIALCOST)"), "amp": not args.no_amp, "label_version":
           {"native": "arena-native/1.1", "membership": "extended-membership/1.1", "extended": "extended/2.0",
            "native_v2": "socialcost_v2/2.0 native", "membership_v2": "socialcost_v2/2.0 membership", "extended_v2": "socialcost_v2/2.0 extended"}[args.label]}
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    kw = dict(num_workers=args.workers, collate_fn=collate, pin_memory=args.model not in SNGNN, persistent_workers=args.workers > 0)
    tl = DataLoader(train, batch_size=batch, shuffle=True, drop_last=True, **kw)
    vl = DataLoader(val, batch_size=batch, shuffle=False, **kw)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    best, step, seen, t0, run_loss = float("inf"), 0, 0, time.time(), []
    model.train()
    while step < steps:
        for b in tl:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = runner.loss(b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched:
                sched.step()
            step += 1
            seen += batch
            run_loss.append(float(loss))
            if seen % args.val_every < batch or step == steps:
                vm = evaluate(runner, vl, "val")["val"]
                # social part: nonzero error is complemented by interaction IoU (higher is better)
                sc = (vm["mae_nonzero"] or 0) + 50.0 * (1.0 - (vm["inter_iou"] or 0.0)) + args.score_fp_weight * (vm["mae"] or 0)
                rec = {"step": step, "seen": seen, "train_loss": float(np.mean(run_loss)), "val": vm, "score": sc, "wall_s": round(time.time() - t0, 1)}
                run_loss = []
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[{os.path.basename(args.out)}] step {step}/{steps} loss {rec['train_loss']:.4f} val mae {vm['mae']:.2f} nz {vm['mae_nonzero']:.1f} "
                      f"hc_iou {vm['hc_iou']} inter_iou {vm['inter_iou']} score {sc:.2f} ({rec['wall_s']}s)", flush=True)
                if sc < best:
                    best = sc
                    torch.save({"model": model.state_dict(), "step": step, "config": cfg}, os.path.join(args.out, "best.pt"))
            if step >= steps:
                break
    train_wall = time.time() - t0
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=dev)["model"])
    metrics = {"best_val_score": best, "train_wall_s": train_wall, "params": n_params(model)}
    if args.overfit or args.tune:
        metrics["val"] = evaluate(runner, vl, "val")
    else:
        preds, tgts, sigs, poses = {}, {}, {}, {}
        for sp in TEST:
            ds = Dv1Samples(split_ids(sp), args.label, v, stride=args.test_stride, eval_maps=True, graph_builder=gb)
            metrics[sp] = evaluate(runner, DataLoader(ds, batch_size=batch, shuffle=False, **{**kw, "persistent_workers": False}), sp, preds, tgts, sigs)
            for e in ds.eps:
                for k in range(0, e.n, args.test_stride):
                    poses[(e.id, k)] = e.robots[k]["pose_map_base"]
        metrics["test_d"] = counterfactual(test_d_pairs(), preds, tgts, poses)
        metrics["test_d_unaligned"] = counterfactual(test_d_pairs(), preds, tgts)
        np.savez_compressed(os.path.join(args.out, "predictions.npz"), **{f"{e}|{k}": p for (e, k), p in preds.items()})
        if sigs:
            np.savez_compressed(os.path.join(args.out, "sigma.npz"), **{f"{e}|{k}": p for (e, k), p in sigs.items()})
        # inference latency (batch 1, model forward only, inputs prepared) and peak memory
        ds = Dv1Samples(split_ids("test_a")[:1], args.label, v, stride=50, graph_builder=gb)
        b = collate([ds[0]])
        model.eval()
        with torch.no_grad():
            x = runner.to_dev(b["x"])
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(30):
                model(x)
            torch.cuda.synchronize()
        metrics["latency_ms_batch1"] = (time.time() - t) / 30 * 1000
        if gb is not None:
            t = time.time()
            for _ in range(10):
                gb(ds.eps[0], 20)
            metrics["graph_build_ms_cpu"] = (time.time() - t) / 10 * 1000
    if hasattr(model, "interpretable"):
        metrics["typemix_parameters"] = model.interpretable()
    json.dump(metrics, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: (v.get(k) if isinstance(v, dict) and k in v else v) for k, v in metrics.items() if k in ("best_val_score", "latency_ms_batch1")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
