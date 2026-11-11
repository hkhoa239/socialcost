# SOCIALCOST: reproduce everything

From a clean machine to every result: code, Arena + Isaac Sim, dataset and checkpoints, the costmap benchmark, the temporal
interaction memory, and closed-loop Nav2 navigation.

| What | Where |
|---|---|
| Code | https://github.com/hkhoa239/socialcost |
| Dataset (dataset_v1, socialcost_v2 labels, navigation challenge) | https://huggingface.co/datasets/hk239/socialcost |
| Trained models (v2 costmap models, stage 1, memory) | https://huggingface.co/hk239/socialcost |
| Simulator | Arena (https://github.com/voshch/Arena) + Isaac Sim 6.1 |

## 0. Requirements and layout

- Ubuntu 22.04 / 24.04, NVIDIA GPU with ≥ 12 GB (RTX 3060 used), driver with CUDA 12+, Docker with nvidia-container-toolkit
  (user in group `docker`), `git`, `rsync`, `ffmpeg`, `uv` (https://docs.astral.sh/uv/).
- Disk: ~150 GB (Isaac image 32 GB, dataset 50 GB packed / 47 GB unpacked, checkpoints 1 GB, outputs).
- **Paths.** The scripts default to the workspace at `/home/robotics/arena_ws` on the host, mounted at `/opt/arena_ws` in
  the Arena container. The simplest setup is the same path (or a symlink `sudo ln -s $HOME/arena_ws /home/robotics/arena_ws`).
  Otherwise override with the environment variables in Section 13.

```
arena_ws/
  src/Arena/                    Arena (simulator, Nav2, humansim, Isaac bridge) + SOCIALCOST patches
  src/deps/socialcost/          this repository
  external/SNGNN2D-v2/          official SNGNN2D-v2 (baseline, untouched)
  data/socialcost/              dataset_v1/, dataset_v1_labels_v2/, navigation_challenge_v1/, experiments/, reports/
  data/costmap_bench/           .venv (benchmark env), runs/v2/, stage1/, cascade_cache/, memory/
```

```bash
export WS=/home/robotics/arena_ws
```

## 1. Code

```bash
# Arena: follow its installer (creates the ROS 2 workspace at $WS and the arena container)
cd ~ && curl https://raw.githubusercontent.com/voshch/Arena/jazzy/install.sh > install.sh && bash install.sh
cd $WS/src/Arena
git checkout c9de66d156442e9b086f8ff3e6e2d1b842e6bd8b && git submodule update --init --recursive
# exact submodule revisions used: socialcost/arena_patches/arena_base.txt

# SOCIALCOST
mkdir -p $WS/src/deps && git clone https://github.com/hkhoa239/socialcost.git $WS/src/deps/socialcost

# SNGNN2D-v2 baseline (left untouched; the adapter lives in socialcost/bench/sngnn_adapter.py)
mkdir -p $WS/external && git clone https://github.com/gnns4hri/SNGNN2D-v2.git $WS/external/SNGNN2D-v2
git -C $WS/external/SNGNN2D-v2 checkout 6a34e0306787c31378261aebc7a0effbdbdcec98

# SOCIALCOST additions to Arena: run_seed launch argument, halt-gesture republish on reset (humansim), Isaac render-every
# flag, socialcost_estimate* Nav2 overlays, the socialcost_office world, all scv1_* / socialcost_* scenarios
bash $WS/src/deps/socialcost/arena_patches/apply.sh $WS/src/Arena
```

## 2. Arena + Isaac Sim

```bash
cd $WS && source arena
arena feature docker gpu on
arena feature isaac install          # pulls nvcr.io/nvidia/isaac-sim:6.1.0 (~32 GB) and builds the bridge image
```

Build the packages SOCIALCOST needs, inside the Arena container (`--packages-select`: `arena build <pkg>` would rebuild every
dependent package):

```bash
docker exec -it -u arena -w /opt/arena_ws arena-arena_ws-arena-1 bash
source /opt/arena_ws/source
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/opt/venv/bin/python \
  --base-paths /opt/arena_ws/src --build-base /opt/arena_ws/build --install-base /opt/arena_ws/install \
  --packages-select isaacsim_msgs arena_runtime arena_simulation_setup arena_robots task_generator arena_isaac arena_humansim socialcost
exit
```

`arena_simulation_setup` must be rebuilt whenever a scenario directory is added.

### The `socialcost-arena` container

SOCIALCOST jobs run in a sibling container (same contents and mounts, own PID namespace, `ROS_DOMAIN_ID=7`), so cleanup of other
jobs in the main container cannot kill them. It drives the same compose `isaac` service; only one Isaac instance can run at a time.

```bash
docker commit --pause=false arena-arena_ws-arena-1 socialcost-arena:snapshot-20261029   # once (carries the apt packages)
bash $WS/src/deps/socialcost/scripts/launch/start_sibling.sh                            # (re)creates socialcost-arena
```

If nothing else uses the main container you can skip this and use `arena-arena_ws-arena-1` with `export ROS_DOMAIN_ID=7`
(replace `socialcost-arena` below). Enter it:

```bash
docker exec -it -u arena -w /opt/arena_ws socialcost-arena bash
cd /opt/arena_ws && source /opt/arena_ws/source        # must be sourced from /opt/arena_ws
```

Copy the helper scripts the pipelines expect to `data/socialcost/scripts/`:

```bash
mkdir -p $WS/data/socialcost/scripts
cp $WS/src/deps/socialcost/scripts/launch/{launch_isaac.sh,stop_isaac.sh,start_sibling.sh} $WS/data/socialcost/scripts/
cp $WS/src/deps/socialcost/tools/*.sh $WS/data/socialcost/scripts/ 2>/dev/null || true
```

## 3. Benchmark environment (host)

The costmap models train on the host in a Python 3.11 venv pinned to torch 2.1.2 + DGL 2.0 (SNGNN2D-v2's requirement; the Arena
container's torch cannot load DGL).

```bash
mkdir -p $WS/data/costmap_bench && cd $WS/data/costmap_bench
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python dgl==2.0.0 -f https://data.dgl.ai/wheels/cu121/repo.html
uv pip install --python .venv/bin/python -r $WS/src/deps/socialcost/env/requirements-bench.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121 -f https://data.dgl.ai/wheels/cu121/repo.html
cp $WS/src/deps/socialcost/env/env.sh .
source env.sh && $PY -c "import torch, dgl; print(torch.__version__, dgl.__version__, torch.cuda.is_available())"
```

Every host command below starts with `cd $WS/data/costmap_bench && source env.sh` (defines `$PY`, `PYTHONPATH`,
`LD_LIBRARY_PATH`, `DGLBACKEND`).

## 4. Dataset

```bash
uv tool install "huggingface_hub[cli]"        # or: python3 -m venv ~/hfenv && ~/hfenv/bin/pip install -U "huggingface_hub[cli]"
hf download hk239/socialcost --repo-type dataset --local-dir $WS/data/hf_dataset

D=$WS/data/socialcost; mkdir -p $D/dataset_v1/episodes $D/dataset_v1_labels_v2/episodes
cd $WS/data/hf_dataset
cp -r manifests splits metadata $D/dataset_v1/
ls v1_episodes/*.tar | xargs -P 8 -I{} tar -xf {} -C $D/dataset_v1/episodes
ls v2_labels/*.tar   | xargs -P 8 -I{} tar -xf {} -C $D/dataset_v1_labels_v2/episodes
cp v2_labels_manifest.json $D/dataset_v1_labels_v2/manifest.json
tar -xf navigation_challenge_v1.tar -C $D
ls $D/dataset_v1/episodes | wc -l          # 600
```

Contents: `docs/DATASET_FORMAT.md` and `docs/LABELS.md`. Per episode: RGB, depth, robot / pedestrian states, interaction
frames, v1 labels (`social_cost` native, `social_cost_membership`, `social_cost_extended`), component maps, visibility. The v2 tree
has the same three label directories plus `components/`. Splits (`splits/`, incl. the counterfactual Test D pairs) are shared.

Rebuild the per-person stage-1 cache (tracks, visibility, RGB-D crops; derived, ~2.5 GB, ~20 min):

```bash
cd $WS/data/costmap_bench && source env.sh
$PY -m socialcost.bench.cascade_data 12            # cascade_cache/<episode>.npz
$PY -m socialcost.bench.cascade_data --crops2 12   # cascade_cache/crops2/ (128 x 64 crops)
```

## 5. Trained models

```bash
hf download hk239/socialcost --local-dir $WS/data/hf_models
B=$WS/data/costmap_bench; mkdir -p $B/runs $B/stage1
rsync -a $WS/data/hf_models/costmap_v2/ $B/runs/v2/        # <model>_<label>_s<seed>/{best.pt, config.json, metrics.json, log.jsonl}
rsync -a $WS/data/hf_models/stage1/ $B/stage1/             # GroupNet ensembles, stage-1 predictions, memory configs
cp $WS/data/hf_models/{selected_lr.json,v2_results_summary.json} $B/
```

| Checkpoint folder | Model |
|---|---|
| `typemix_cffullraw_<label>` | **ours**: counterfactual stage 1 (tracks + RGB-D crops) → TypeMix |
| `typemix_cffullrawmem{hold,decay,decay_geo}_<label>` | ours + temporal memory (ablations; `decay_geo` is the selected one) |
| `typemix_gt_<label>` | TypeMix fed the true social types (privileged reference) |
| `typemix_{vis,tracks}`, `cascade_{vis,tracks}` | stage-1 variants |
| `pos`, `pos_u` | position U-Net (deterministic / probabilistic) |
| `rgb`, `depth`, `rgbd` | image U-Nets |
| `sngnn_sensor`, `sngnn_priv` | SNGNN2D-v2 (sensor graph / privileged interaction links) |

Labels: `native_v2`, `membership_v2`, `extended_v2` (navigation uses `extended_v2`). Seeds 0-2. Not included (out of memory during
training, retrain with Section 8): `sngnn_priv` membership s0/s2 and native s0/s1, `sngnn_sensor` extended s0, membership s2, native s1.

## 6. Run the simulator

Inside `socialcost-arena` (sourced):

```bash
# a dataset scenario
arena launch sim:=isaac headless:=true world:=reception robot:=turtlebot human:=arena \
  task.robots:=scenario task.obstacles:=scenario task.scenario.file:=socialcost_pair_conversation_bA_on run_seed:=1 \
  robot.mobile.params_overlay:=social_cost_layer

# a navigation-challenge configuration (N01..N36: world and scenario in navigation_challenge_v1.json)
arena launch sim:=isaac headless:=true world:=airport robot:=turtlebot human:=arena \
  task.robots:=scenario task.obstacles:=scenario task.scenario.file:=scv1_nav_N09 run_seed:=5009 \
  robot.mobile.local_planner:=dwb robot.mobile.params_overlay:=socialcost_estimate_global
```

Detached with a log: `docker exec -d socialcost-arena bash -l /opt/arena_ws/data/socialcost/scripts/launch_isaac.sh <log> <args...>`;
stop: `docker exec socialcost-arena bash /opt/arena_ws/data/socialcost/scripts/stop_isaac.sh`. The first launch after a container
start compiles shaders (~3 min).

Nav2 overlays for the social layer (`arena_robots/config/nav2/overlays/`):

| Overlay | Social cost goes to | Use |
|---|---|---|
| `socialcost_estimate_global` | global costmap only; local controller = Nav2 default | **recommended** (no freezing on faint cost) |
| `socialcost_estimate` | both costmaps, DWB obstacle critic retuned (scale 0.3, sum_scores) | v1 navigation runs (freezes) |
| `socialcost_estimate_obs01[_patient]` | both, obstacle scale 0.1 (+ 60 s progress checker) | controller diagnosis |

## 7. Tests

```bash
# inside socialcost-arena (needs the compiled label painter)
cd /opt/arena_ws/src/deps/socialcost
PYTHONPATH=. /opt/venv/bin/python -m pytest tests/test_labels.py tests/test_labels_v2.py tests/test_memory.py -q
ROS_DOMAIN_ID=9 PYTHONPATH=. /opt/venv/bin/python -m pytest tests/test_live_agreement.py -q     # real Nav2 plugin, ~75 s
```

## 8. Labels (optional: the dataset already contains them)

```bash
# inside socialcost-arena
cd /opt/arena_ws/src/deps/socialcost
PYTHONPATH=. /opt/venv/bin/python scripts/relabel_v2.py --spec v2 --out /opt/arena_ws/data/socialcost/dataset_v1_labels_v2 --workers 8
PYTHONPATH=. /opt/venv/bin/python scripts/labels_v2_validate.py --out /opt/arena_ws/data/socialcost/reports/labels_v2
```

`socialcost/labels_v2.py` holds the specs (`v2` final; `v2_fixedsigma`, `v2_allpairs` ablations; `v1` reproduction). Every
label is painted by the same C code as the Nav2 social layer (`arena_social_cost_layer/cost_math.hpp`).

## 9. Train costmap models

One run (host):

```bash
cd $WS/data/costmap_bench && source env.sh
$PY $WS/src/deps/socialcost/scripts/bench_train.py --out runs/v2/typemix_cffullraw_extended_v2_s0 \
  --model typemix_cffullraw --label extended_v2 --seed 0 --lr 0.001 --samples 48000 --val-every 6000 --batch 16 --workers 6 \
  --score-fp-weight 2.0
```

Learning rates: `pos, pos_u, cascade_*, typemix_*` 1e-3; `rgb, depth, rgbd` 3e-4; `sngnn_sensor` 5e-5; `sngnn_priv` 2e-4.
`--score-fp-weight 2` (checkpoint score + 2 × whole-grid error) is the v2 protocol; without it image models can win by painting
cost everywhere. Each run selects its checkpoint on validation and writes test metrics (A, B, C, counterfactual D) and test
predictions.

All 108 runs (resumable, GPU-memory gated, retries out-of-memory failures):

```bash
cp $WS/src/deps/socialcost/tools/run_v2_train.py $WS/data/costmap_bench/
cd $WS/data/costmap_bench && setsid nohup .venv/bin/python run_v2_train.py > logs/v2_train.log 2>&1 &
tail -f STATUS_v2_train.txt
```

Stage 1 (GroupNet, cross-fitted predictions for TypeMix training) is included in the models download. To retrain it:
`$PY $WS/src/deps/socialcost/scripts/groups_pipeline.py --root $WS/data/costmap_bench --jobs 3 --workers 6 --cf`.

## 10. Temporal interaction memory

```bash
cd $WS/src/deps/socialcost && source $WS/data/costmap_bench/env.sh
M=$WS/data/costmap_bench/memory; mkdir -p $M
$PY scripts/memory_stage1.py tune  --src cffullraw --out $M/tune_cffullraw.json                    # validation only
$PY scripts/memory_stage1.py build --src cffullraw --config $M/tune_cffullraw.json                 # stage1/cffullrawmem*/
$PY scripts/memory_stage1.py eval  --src cffullraw --config $M/tune_cffullraw.json \
  --out $M/eval_cffullraw.json --report $WS/data/socialcost/reports/memory_stage1_results.md
```

The memory sources are in the models download; `build` is needed only after re-tuning. TypeMix memory models train with
`--model typemix_cffullrawmem{hold,decay,decay_geo}`.

## 11. Evaluate costmaps (offline)

```bash
cd $WS/src/deps/socialcost && source $WS/data/costmap_bench/env.sh
# summary table of runs/v2 -> data/costmap_bench/v2_results_summary.json (also shipped with the models)
# error around each person by time since last seen
$PY scripts/costmap_staleness_eval.py --label extended_v2 \
  --models typemix_cffullraw typemix_cffullrawmemdecay_geo typemix_gt pos_u --out $WS/data/costmap_bench/v2_eval/staleness_extended_v2.json
# inside vs outside the camera view
$PY scripts/fov_split_eval.py --out $WS/data/costmap_bench/fov_split/fov_split.json --report $WS/data/socialcost/reports/costmap_fov_split.md
# figures for the report
$PY scripts/v2_figures.py --out $WS/data/socialcost/reports/v2_results
```

Test metrics per run are in `runs/v2/<run>/metrics.json` (`test_a|b|c` → `mae`, `mae_nonzero`, `hc_iou`, `inter_iou`,
`fp_area_m2`; `test_d` → counterfactual `cf_err`, `cf_response`).

## 12. Closed-loop navigation

36 configurations (`navigation_challenge_v1.json`: conversation, queue, raised hand, group walking, occlusion), one Isaac launch per
world, an estimator per arm that publishes its predicted costmap to Nav2, and a ground-truth cost meter.

```bash
# inside socialcost-arena: v2 labels, native + extended, position U-Net vs ours (+ memory), ground truth, no social cost
docker exec -d -u arena socialcost-arena bash /opt/arena_ws/data/socialcost/scripts/run_labels_v2_nav.sh
cat $WS/data/socialcost/experiments/navigation/STATUS_labels_v2.txt     # BEGIN ... DONE; ~16 h for 288 runs
```

The script runs `scripts/run_nav_challenge.py --config configs/navigation/challenge_labels_v2.yaml`, then scores native arms against
the v2 native and extended arms against the v2 extended ground truth:

```bash
cd /opt/arena_ws/src/deps/socialcost
python scripts/nav_challenge_metrics.py $N/challenge_labels_v2 --challenge $CH --meter-label v2e --oracle gt_ext \
  --methods bodies gt_ext ours_ext oursmem_ext posu_ext --out $N/challenge_labels_v2_extended.json \
  --report /opt/arena_ws/data/socialcost/reports/navigation_labels_v2_extended.md
```

(`N=/opt/arena_ws/data/socialcost/experiments/navigation`, `CH=/opt/arena_ws/data/socialcost/navigation_challenge_v1/navigation_challenge_v1.json`.)

One configuration / one arm by hand:

```bash
python scripts/run_nav_challenge.py --config configs/navigation/challenge_labels_v2.yaml --only N09
python -m socialcost.estimator_node --method typemix --bench-runs v2 --bench-label extended_v2 \
  --bench-memory decay_geo --bench-threshold 12 --log /tmp/est.jsonl     # with a scenario running
```

Estimator methods: `zero` (no social cost), `oracle_ext` (`--label-spec v2 --label-channel native|extended`), `pos_u`, `typemix`
(`--bench-memory` for the memory), `typemix_gt`, `rgb` / `rgbd` (`--checkpoint a.pt,b.pt,c.pt`), `sngnn` (needs the host service:
`$PY scripts/sngnn_server.py --label extended_v2 --runs v2` and `--config-id {config_id}` in the arm's args).

The runner skips finished runs (resumable), waits while the machine load is above `max_load`, and redoes a run whose simulator ran
too slowly to reach the 180 s simulated time limit.

### Videos

```bash
# four arms side by side (inside socialcost-arena), then H.264 on the host
R=/opt/arena_ws/data/socialcost/experiments/navigation
python scripts/nav_compare_video.py --runs $R/challenge_labels_v2 --config N09 --arms gt_ext ours_ext oursmem_ext posu_ext \
  --metrics $R/challenge_labels_v2_extended.json --out $R/videos/labels_v2/N09.mp4
ffmpeg -i $WS/data/socialcost/experiments/navigation/videos/labels_v2/N09.mp4.raw.mp4 -c:v libx264 -pix_fmt yuv420p \
  $WS/data/socialcost/experiments/navigation/videos/labels_v2/N09.mp4
# every saved run of one arm: bash $WS/data/socialcost/scripts/make_nav_videos.sh challenge_labels_v2 ours_ext
```

## 13. Paths, environment variables, practical notes

| Variable | Default |
|---|---|
| `SC_DV1_ROOT` | `/home/robotics/arena_ws/data/socialcost/dataset_v1` (container: `/opt/arena_ws/...`) |
| `SC_LABELS_V2_ROOT` | `/home/robotics/arena_ws/data/socialcost/dataset_v1_labels_v2` |
| `SC_BENCH_ROOT` | `/opt/arena_ws/data/costmap_bench` (online models), `/home/robotics/arena_ws/data/costmap_bench` (scripts) |
| `SC_STAGE1_ROOT` | `<bench>/stage1` |
| `SC_CASCADE_CACHE` | `<bench>/cascade_cache` |
| `SNGNN2D_DIR` | `/home/robotics/arena_ws/external/SNGNN2D-v2` |

- One Isaac at a time; jobs run detached (`docker exec -d`, `setsid nohup`) and write a `STATUS*.txt`.
- Run times on one RTX 3060: TypeMix ~25 min per run, image U-Nets ~9 min, SNGNN2D-v2 ~45 min, full navigation batch (288 runs)
  ~16 h. TypeMix and SNGNN take 3-4.5 GB of GPU memory each.
- Navigation scenario folders and the `socialcost_office` world come from `arena_patches/`; after adding scenarios rebuild
  `arena_simulation_setup`.
- Results and their caveats: `reports/socialcost_v2_results_2026-11-09.docx`, `reports/navigation_labels_v2_{extended,native}.md`.
