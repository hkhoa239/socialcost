#!/usr/bin/env python3
"""Train / evaluate one social cost estimator.

    python train_estimator.py --config configs/experiments/<name>.yaml [--overfit 32] [--eval-only ckpt.pt]

Config keys: name, inputs (rgbd|depth|rgb|pos|pos_fov), label (native|membership|extended), datasets
(roots), split_file (json: {train: [episode keys], val: [...], test_*: [...]}), epochs, batch, lr, seed,
out_dir. Writes <out_dir>/<name>/{config.yaml, run.json, best.pt, metrics.json, log.jsonl}.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost.learn.data import EpisodeIndex, SocialCostSamples, episode_dirs  # noqa: E402
from socialcost.learn.metrics import Accumulator  # noqa: E402
from socialcost.learn.models import build, n_params  # noqa: E402


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def collate(batch: list[dict]) -> dict:
    x = {k: torch.stack([b["x"][k] for b in batch]) for k in batch[0]["x"]}
    return {"x": x, "target": torch.stack([b["target"] for b in batch]), "rule": torch.stack([b["rule"] for b in batch]),
            "fov": torch.stack([b["fov"] for b in batch]), "meta": [b["meta"] for b in batch]}


def loss_fn(pred: torch.Tensor, target: torch.Tensor, rule: torch.Tensor, region: torch.Tensor | None = None) -> torch.Tensor:
    # mostly-zero targets: weight cells with cost x5 and the social part above the rule x10;
    # region (camera FOV) restricts the loss to what the camera can observe
    w = 1.0 + 4.0 * (target > 0).float() + 10.0 * (target > rule + 0.5 / 252).float()
    if region is not None:
        w = w * region.float()
        return (w * torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=0.02)).sum() / region.float().sum().clamp(min=1)
    return (w * torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=0.02)).mean()


def to_dev(x: dict, dev: torch.device) -> dict:
    return {k: v.to(dev, non_blocking=True) for k, v in x.items()}


@torch.no_grad()
def evaluate(model, loader, dev, per_sample: list | None = None) -> dict:
    model.eval()
    acc = Accumulator()
    for b in loader:
        pred = model(to_dev(b["x"], dev)).float().cpu()
        groups = ["overall"] + [f"family:{m['family']}" for m in b["meta"]]
        for gi, g in enumerate(groups):
            if gi == 0:
                acc.add(g, pred, b["target"], b["rule"], b["fov"])
            else:
                i = gi - 1
                acc.add(g, pred[i:i + 1], b["target"][i:i + 1], b["rule"][i:i + 1], b["fov"][i:i + 1])
        if per_sample is not None:
            for i, m in enumerate(b["meta"]):
                per_sample.append({**m, "pred": (pred[i] * 252).round().clamp(0, 252).to(torch.uint8).numpy()})
    return acc.result()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overfit", type=int, default=0, help="train and evaluate on N training samples only")
    ap.add_argument("--eval-only", default=None)
    ap.add_argument("--save-predictions", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    name = cfg["name"] + (f"_overfit{args.overfit}" if args.overfit else "")
    out = os.path.join(cfg["out_dir"], name)
    os.makedirs(out, exist_ok=True)
    seed_all(cfg.get("seed", 0))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eps = {e.key: e for e in (EpisodeIndex(d) for d in episode_dirs(cfg["datasets"]))}
    split = json.load(open(cfg["split_file"]))
    sets = {k: [eps[e] for e in v if e in eps] for k, v in split.items() if not k.startswith("_")}
    ds = {k: SocialCostSamples(v, cfg["label"], cfg["inputs"]) for k, v in sets.items() if v}
    train = ds["train"]
    if args.overfit:
        train = torch.utils.data.Subset(train, list(range(0, len(train), max(1, len(train) // args.overfit)))[: args.overfit])
    kw = dict(batch_size=cfg.get("batch", 8), num_workers=cfg.get("workers", 6), collate_fn=collate, pin_memory=True)
    model = build(cfg["inputs"]).to(dev)
    run = {"name": name, "config": cfg, "params": n_params(model), "device": str(dev), "split_sizes": {k: len(v) for k, v in ds.items()},
           "git": {p: subprocess.run(["git", "-C", p, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
                   for p in ("/opt/arena_ws/src/Arena",)},
           "label_versions": {k: json.load(open(os.path.join(r, "labels.json")))[{"native": "social_cost", "membership": "social_cost_membership", "extended": "social_cost_extended"}[cfg["label"]]].get("version")
                              for k, r in enumerate(cfg["datasets"]) if os.path.exists(os.path.join(r, "labels.json"))},
           "overfit": args.overfit}
    yaml.safe_dump(cfg, open(os.path.join(out, "config.yaml"), "w"))

    if not args.eval_only:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=1e-4)
        epochs = cfg.get("epochs", 20) if not args.overfit else cfg.get("overfit_epochs", 200)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        tl = DataLoader(train, shuffle=True, drop_last=False, **kw)
        vl = DataLoader(ds["val"] if not args.overfit else train, shuffle=False, **kw)
        best, t0 = float("inf"), time.time()
        log = open(os.path.join(out, "log.jsonl"), "a")
        for ep in range(epochs):
            model.train()
            tot, n = 0.0, 0
            for b in tl:
                pred = model(to_dev(b["x"], dev))
                region = b["fov"].to(dev) if cfg.get("loss_region", "all") == "fov" else None
                loss = loss_fn(pred, b["target"].to(dev), b["rule"].to(dev), region)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss) * len(b["meta"])
                n += len(b["meta"])
            sched.step()
            vm = evaluate(model, vl, dev)["overall"]
            # select on the social part and the nonzero region, not global MAE
            fov = cfg.get("loss_region", "all") == "fov"
            score = (vm["mae_nonzero_fov" if fov else "mae_nonzero"] or 0) + (vm["mae_social_extra_fov" if fov else "mae_social_extra"] or 0)
            rec = {"epoch": ep, "train_loss": tot / max(n, 1), "val": vm, "score": score, "wall_s": time.time() - t0}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{name}] epoch {ep} loss {rec['train_loss']:.4f} val mae_all {vm['mae_all']:.2f} nonzero {vm['mae_nonzero']} social_extra {vm['mae_social_extra']} iou {vm['iou_nonzero_all']}", flush=True)
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "epoch": ep, "config": cfg}, os.path.join(out, "best.pt"))
        run["train_wall_s"] = time.time() - t0
    ckpt = args.eval_only or os.path.join(out, "best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=dev)["model"])
    run["checkpoint"] = ckpt
    metrics = {}
    preds_all = {}
    eval_sets = {"train_subset": train} if args.overfit else {k: v for k, v in ds.items() if k != "train"}
    for k, d in eval_sets.items():
        per = [] if args.save_predictions else None
        metrics[k] = evaluate(model, DataLoader(d, shuffle=False, **kw), dev, per)
        if per is not None:
            preds_all[k] = per
    # inference cost
    b = next(iter(DataLoader(list(eval_sets.values())[0], batch_size=1, collate_fn=collate)))
    x = to_dev(b["x"], dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            model(x)
        torch.cuda.synchronize()
    run["inference_ms_batch1"] = (time.time() - t) / 50 * 1000
    run["inference_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    json.dump(run, open(os.path.join(out, "run.json"), "w"), indent=2, default=str)
    if preds_all:
        np.savez_compressed(os.path.join(out, "predictions.npz"), **{f"{k}|{p['episode']}|{p['sample']}": p["pred"] for k, v in preds_all.items() for p in v})
    print(json.dumps({k: v.get("overall") for k, v in metrics.items()}, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
