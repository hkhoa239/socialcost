# Dataset format (`socialcost-dataset/1`)

```text
<root>/
  metadata.json                 dataset config, sensors, label params and grid, software versions, manifest
  manifest.json                 "<scenario>:<seed>:<rep>" -> episode directory
  splits/{train,val,test}.txt   episode directory names, one per line
  logs/                         live-costmap logs per job
  collect.log                   collection log
  validation/                   validation.json, validation.md
  vis/                          sample sheets
  episodes/episode_NNNNNN/
    rgb/NNNNNN.png              320x240 RGB (stored BGR by OpenCV, load_sample returns RGB)
    depth/NNNNNN.npy            float32 [240, 320], metres, planar depth (distance to image plane), NaN = invalid
    social_cost/NNNNNN.npy      uint8 [200, 200], exact Arena social layer, robot frame
    social_cost_membership/NNNNNN.npy   same with queue membership completed (COSTMAP_PIPELINE.md)
    live_social_cost/NNNNNN.npy uint8, live plugin grid (map-aligned rolling window), QA only
    camera_info.json            K, D, R, P, size, frames, T_base_camera
    robot_states.jsonl          one line per sample
    pedestrian_states.jsonl     one line per sample (privileged)
    peds_stream.jsonl           every arena_peds message during the recording (privileged)
    interactions.jsonl          humansim InteractionFrame changes (privileged)
    sync.jsonl                  per-sample stamps, encodings, live costmap metadata
    episode_metadata.json       scenario, seeds, timing, stats, label params, QA inputs
```

## Sample

`socialcost.dataset.load_sample(ep_dir, k)` and `SocialCostDataset(root, split)` return:

```python
sample = {
    "rgb": uint8[240, 320, 3],                 # RGB
    "depth": float32[240, 320],                # m, NaN invalid
    "social_cost": uint8[200, 200],
    "social_cost_membership": uint8[200, 200],
    "robot_pose": {x, y, z, qx, qy, qz, qw, yaw},           # base_link in map
    "camera_intrinsics": float[3, 3],
    "camera_extrinsics": {"T_base_camera": {...}, "T_map_camera": {...}},
    "timestamp": float,                        # simulator seconds (image stamp)
}
```

`pedestrian_states` is added only with `with_privileged=True`. It is for labels, debugging and position-based baselines, and must not be an input of the image estimator.

## Frames and conventions

| Item | Convention |
|---|---|
| World frame | `map` (Arena/Isaac). Scenario yaml coordinates are offset from `map` by a constant (≈ +5.3 m, +5.3 m in map_empty); every stored pose and pedestrian is in `map`. |
| Robot frame | `env_0/turtlebot/base_link`, x forward, y left, z up (REP-103) |
| Camera frame | `env_0/turtlebot/oakd_rgb_camera_optical_frame`, REP-103 optical: x right, y down, z forward. Mounted 0.17 m above the floor, optical axis along base +x. |
| Intrinsics | fx = fy = 221.765, cx = 160, cy = 120, no distortion (horizontal FOV 1.25 rad = 71.6°, vertical 56.7°) |
| Depth | Isaac `DistanceToImagePlane`: planar z in the optical frame. Pixels with no surface (above the walls; the room has no ceiling) are NaN. |
| Label grid | 10 m × 10 m, 0.05 m, 200 × 200, centred on the robot, axis-aligned with base_link. `label[j, i]`: i along +x (forward), j along +y (left); cell centre `(−5 + (i+0.5)·0.05, −5 + (j+0.5)·0.05)`. Plot with `imshow(label.T, origin="lower")` and the y axis reversed to get forward-up, left-left. |
| Time | Simulator time (`/clock`), nanoseconds in `stamp_ns` fields |

## Per-sample records

`robot_states.jsonl`: `sample`, `slot` (index on the 5 Hz grid from the window start), `stamp_ns` (image stamp), `pose_map_base`, `pose_map_camera` (TF at the image stamp), `odom_twist`.

`pedestrian_states.jsonl`: `sample`, `stamp_ns`, `peds_before_ns`, `peds_after_ns`, `alpha` (the two bracketing `arena_peds` messages and the interpolation weight), `pedestrians`: `name`, `id` (humansim agent id), `x, y, z, yaw, vx, vy, wz`, `animation_state`, `interaction_id`, `interaction_type`, `agent_type`, `model_uri` (body asset), `gestures`, `gait_phase`. Positions and yaw are interpolated to the image stamp; discrete fields come from the nearer message.

`interactions.jsonl`: `stamp_ns`, `interactions`: `interaction_id`, `interaction_type`, `outcome` (1 = ACTIVE), `object_id`, `member_ids` (agent ids), `member_roles` (0 participant, 1 queued).

## Episodes

An episode is one Arena episode. Its recording window opens `warmup_s` after the episode start stamp (`EpisodeRecord.start_time`, robot spawn) and lasts `duration_s`; samples sit on a `1/sample_hz` grid from the window start. If the recorder attaches after the window opened, it waits for the next episode, so every window has the same offset from its seeded start. A reset (episode id change, outcome ≠ RUNNING, simulator time going backwards) ends the episode; no sample crosses an episode boundary.

`episode_metadata.json` holds `scenario`, `run_seed` (launch `run_seed:=`), `episode_id`, `episode_seed` (Arena's per-episode seed, derived from `run_seed`), `bodies` (pedestrian body assets), window stamps, `samples` / `expected_samples`, recorder `stats` (frames without depth, TF failures, skipped slots, pedestrian stream gaps, end reason), `label` (params, grid, counts), `live_agreement` (per sample) and `valid`.

An episode is valid with ≥ 80 % of expected samples, no TF failure and at least one label with cost.

## Splits

`make_splits` assigns whole episodes, by the first matching rule in `test`, then `val`, else `train`: any listed `worlds`, any listed pedestrian `bodies`, or any listed run `seeds`. Matching whole seeds keeps correlated trajectories in one split. The smoke test has a single world, so it puts seed 2 in `test`; it supports no claim of generalisation across worlds or bodies.
