#!/usr/bin/env python3
"""Post-hoc evaluation from saved predictions (no retraining).

    $PY bench_post.py realign --root data/costmap_bench          # Test D with OFF maps warped into the ON robot frame
    $PY bench_post.py ensemble --root data/costmap_bench          # seed ensembles of the probabilistic models

realign: writes runs/main/<run>/test_d_aligned.json for every run with predictions.npz.
ensemble: for each probabilistic model and label, combines the 3 seeds (deep-ensemble view, Lakshminarayanan et al. 2017):
  mu = mean_s mu_s,  sigma^2 = mean_s sigma_s^2 (aleatoric) + var_s mu_s (epistemic)
and evaluates accuracy and uncertainty metrics on Test A/B/C -> runs/ensemble/<model>_<label>.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from socialcost.bench.data import LABELS, ROOT, Episode, split_ids, test_d_pairs  # noqa: E402
from socialcost.bench.metrics import Acc, counterfactual  # noqa: E402
from socialcost.learn.data import fov_mask  # noqa: E402


def load_npz(path: str) -> dict:
    z = np.load(path)
    return {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files}


def poses_and_targets(keys, label: str) -> tuple[dict, dict]:
    eps, poses, tg = {}, {}, {}
    for e, k in keys:
        if e not in eps:
            eps[e] = Episode(e)
        poses[(e, k)] = eps[e].robots[k]["pose_map_base"]
        tg[(e, k)] = np.load(os.path.join(ROOT, "episodes", e, LABELS[label], f"{k:06d}.npy"))
    return poses, tg


def realign(root: str) -> None:
    pairs = test_d_pairs()
    want = {p[c] for p in pairs for c in ("on", "off")}
    for d in sorted(glob.glob(os.path.join(root, "runs", "main", "*"))):
        f = os.path.join(d, "predictions.npz")
        if not os.path.exists(f) or os.path.exists(os.path.join(d, "test_d_aligned.json")):
            continue
        label = os.path.basename(d).rsplit("_", 2)[1]
        P = {k: v for k, v in load_npz(f).items() if k[0] in want}
        poses, tg = poses_and_targets(P.keys(), label)
        res = {"aligned": counterfactual(pairs, P, tg, poses), "unaligned": counterfactual(pairs, P, tg)}
        json.dump(res, open(os.path.join(d, "test_d_aligned.json"), "w"), indent=1)
        print(os.path.basename(d), round(res["aligned"]["all"]["cf_err"], 2), round(res["aligned"]["all"]["cf_response"], 2), flush=True)


def ensemble(root: str) -> None:
    os.makedirs(os.path.join(root, "runs", "ensemble"), exist_ok=True)
    man = {e["episode_id"]: e for e in json.load(open(os.path.join(ROOT, "manifests", "manifest_v1.json")))["episodes"]}
    e0 = Episode(split_ids("test_a")[0])
    fov = torch.from_numpy(fov_mask(e0.grid, e0.K, e0.T_bc))
    for model in ("pos_u", "pos_int_u", "cascade_tracks", "cascade_vis", "typemix_tracks", "typemix_vis", "typemix_gt", "typemix_vis_nosem",
                  "typemix_tracks2", "typemix_vis2", "typemix_vis2raw", "typemix_tracks3raw", "typemix_vis3raw",
                  "typemix_cfnoneraw", "typemix_cffullraw", "typemix_cfcostraw"):
        for label in ("native", "membership", "extended"):
            dirs = sorted(glob.glob(os.path.join(root, "runs", "main", f"{model}_{label}_s*")))
            dirs = [d for d in dirs if os.path.exists(os.path.join(d, "sigma.npz"))]
            if len(dirs) < 2:
                continue
            outj = os.path.join(root, "runs", "ensemble", f"{model}_{label}.json")
            if os.path.exists(outj) and json.load(open(outj)).get("seeds") == [os.path.basename(d) for d in dirs] \
                    and os.path.getmtime(outj) > max(os.path.getmtime(os.path.join(d, "sigma.npz")) for d in dirs):
                continue  # up to date
            mus = [load_npz(os.path.join(d, "predictions.npz")) for d in dirs]
            sgs = [load_npz(os.path.join(d, "sigma.npz")) for d in dirs]
            keys = sorted(set(mus[0]) & set.intersection(*[set(m) for m in mus[1:]]))
            acc = {"ensemble": Acc(), "aleatoric_only": Acc(), "epistemic_only": Acc()}
            ens_mu = {}
            poses, tg = poses_and_targets(keys, label)
            by_split = defaultdict(list)
            for k in keys:
                by_split[man[k[0]]["split"]].append(k)
            for sp, ks in by_split.items():
                for i in range(0, len(ks), 32):
                    chunk = ks[i:i + 32]
                    M = np.stack([[m[k] for m in mus] for k in chunk]).astype(np.float32) / 252  # B,S,H,W
                    Sg = np.stack([[s[k] for s in sgs] for k in chunk]).astype(np.float32) / 252
                    mu = M.mean(1)
                    alea = (Sg ** 2).mean(1)
                    epi = M.var(1)
                    T = torch.from_numpy(np.stack([tg[k] for k in chunk]).astype(np.float32) / 252)
                    comps = [np.load(os.path.join(ROOT, "episodes", k[0], "components", f"{k[1]:06d}.npz")) for k in chunk]
                    rule = torch.from_numpy(np.stack([c["personal_space"] for c in comps]).astype(np.float32) / 252)
                    inter = torch.from_numpy(np.stack([(c["conversation"] if label == "native" else np.maximum(np.maximum(c["conversation"], c["queue"]),
                                                                                                c["gesture"] if label == "extended" else 0)) > 0 for c in comps]))
                    F = fov.unsqueeze(0).expand(len(chunk), -1, -1)
                    groups = [[sp, f"{sp}|family:{man[k[0]]['family']}"] for k in chunk]
                    m_t = torch.from_numpy(mu)
                    acc["ensemble"].add(groups, m_t, T, rule, F, inter, torch.from_numpy(np.sqrt(alea + epi)))
                    acc["aleatoric_only"].add(groups, m_t, T, rule, F, inter, torch.from_numpy(np.sqrt(alea)))
                    acc["epistemic_only"].add(groups, m_t, T, rule, F, inter, torch.from_numpy(np.sqrt(epi)))
                    for j, k in enumerate(chunk):
                        ens_mu[k] = (mu[j] * 252).round().astype(np.uint8)
            res = {name: a.result() for name, a in acc.items()}
            res["test_d"] = counterfactual(test_d_pairs(), ens_mu, tg, poses)
            res["seeds"] = [os.path.basename(d) for d in dirs]
            json.dump(res, open(os.path.join(root, "runs", "ensemble", f"{model}_{label}.json"), "w"), indent=1)
            print(model, label, {sp: round(res["ensemble"][sp]["mae_nonzero"], 2) for sp in ("test_a", "test_b", "test_c") if sp in res["ensemble"]}, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("realign", "ensemble"))
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    (realign if args.what == "realign" else ensemble)(os.path.abspath(args.root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
