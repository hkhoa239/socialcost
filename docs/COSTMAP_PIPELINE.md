# Social cost ground truth

## What the Arena layer computes

`arena_social_cost_layer/SocialCostLayer` (`arena_planners/arena_social_cost_layer`) subscribes to `arena_people_msgs/Pedestrians` on `peds_topic` (SensorData QoS) and, on each costmap update, paints:

1. **Personal space**, per pedestrian, on every cell whose centre lies within `personal_space_radius`:
   `amplitude · exp(−(d² / 2σ²)^falloff_shape)`. σ and amplitude depend on `animation_state`:

   | animation_state | σ | amplitude |
   |---|---|---|
   | IDLE, WALKING, RUNNING | `personal_space_sigma` | 0.5 |
   | SURPRISED, CURIOUS | 0.75/0.45 · `personal_space_sigma` | 0.75 |
   | PANIC, THREATENING | 1.2/0.45 · `personal_space_sigma` | 1.0 |

   The term is **isotropic**: pedestrian heading and velocity are not used.
2. **Interaction region**, for every pair of pedestrians sharing an `interaction_id ≥ 0`, on every cell within `group_extra_radius` of the segment between them:
   `w · exp(−(d_seg² / 2·group_sigma²)^falloff_shape)`, with `w = interaction_weights[interaction_type]` (type of the group's first member in roster order). A weight of 0 paints nothing.
3. Each term is mapped to a cell value `⌊clamp(v · cost_scale + cost_bias, 0, 1) · max_cost_value⌋` (`max_cost_value` clamped to 252). A cell keeps the maximum of all terms and of what lower layers wrote.

Pedestrian messages older than `stale_timeout` are ignored.

### Overlay parameters (`social_cost_layer.yaml`)

| Parameter | local_costmap | global_costmap |
|---|---|---|
| personal_space_radius | 1.0 | 1.5 |
| personal_space_sigma | 0.9 | 1.0 |
| group_extra_radius | 2.0 | 2.5 |
| group_sigma | 0.85 | 1.0 |
| falloff_shape | 3.0 | 2.0 |
| max_cost_value | 252 | 200 |
| stale_timeout | 0.5 | 1.0 |
| cost_scale / cost_bias | 0.8 / 0.05 | 1.0 / 0.0 |
| peds_topic | `${env_namespace}/arena_peds` | same |

`interaction_weights`, indexed by humansim `InteractionType`, identical in both blocks:

| idx | type | weight | idx | type | weight |
|---|---|---|---|---|---|
| 0 | TALK_TO | 0.9 | 7 | WAVE_AT | 0.6 |
| 1 | GROUP_CONVERSATION | 1.2 | 8 | BLOCK | 0.8 |
| 2 | (unused) | 0.0 | 9 | SERVICE | 0.8 |
| 3 | SIT_ON | 0.8 | 10 | HUG | 1.5 |
| 4 | LIE_ON | 0.8 | 11 | SHAKE_HAND | 0.8 |
| 5 | USE | 0.8 | 12 | GROUP_WALK | **0.0** |
| 6 | QUEUE_USE | 1.0 | | | |

The overlay loads the layer into both costmaps (local: obstacle, voxel, **social**, inflation; global: static, obstacle, **social**, inflation, mask overlay). Verified on the running TurtleBot with `ros2 param get .../local_costmap/local_costmap plugins`.

### What the layer does not cover, per scenario

| Scenario | Interaction published | Region painted by the layer |
|---|---|---|
| Conversation | `TALK_TO` (id shared by both talkers) | yes, weight 0.9 |
| Queue | `QUEUE_USE`, but **only the person being served** carries the id; queued people publish `interaction_id = −1` because humansim reports only ACTIVE *participants* (`InteractionManager.active_interaction`) and `QUEUE_USE` admits one participant | **no**: a single member makes no pair, so the QUEUE_USE weight never fires on a queue |
| Group walking | `GROUP_WALK` (all members) | **no**: weight 0.0 in the overlay; personal space only |

## Ground-truth extraction

The merged Nav2 costmap is not used as ground truth: it mixes obstacle, voxel and inflation costs, and the layer has no own costmap (it derives from `Layer`, not `CostmapLayer`), so Nav2 publishes no per-layer topic for it.

### Offline painter (the labels)

`socialcost/labels.py` → `libsocialcost_grid.so` (`src/socialcost_grid.cpp`). The C++ shim **includes `arena_social_cost_layer/cost_math.hpp`** and calls the layer's own `personalSpaceCost`, `groupBetweenCost` and `pointToSegmentDistSq`. It repeats `updateCosts`' compositing exactly: cell centres, the radius clip, `toCellCost` truncation, max-composition, the group typing rule and the derived alert/danger σ. Parameters are read from the overlay yaml, so labels and the live layer cannot drift apart. A grid can be placed at any pose and yaw, so labels are robot-centred and yaw-aligned.

### Live reference

`socialcost/live.py` starts Nav2's standalone `Costmap2DROS` (`nav2_costmap_2d` executable) with `plugins: [social_cost_layer]` and the same parameter block. Its `<ns>/costmap_raw` is the layer's output alone, painted by the plugin binary. `collect_dataset.py` runs one as a rolling 10 m window around the robot during every recording.

### Agreement

| Test | Setup | Result |
|---|---|---|
| `tests/test_live_agreement.py` | real plugin, fixed rosters: 6 random scenes × {local, global} × {static, rolling window}, random animation states, groups of 2–4 with random interaction types including weight-0 ones | **0 differing cells** in all 24 scenes |
| `tests/test_labels.py` | shim vs. an independent NumPy rewrite of the formulas | ≤ 1 unit on < 0.1 % of cells (float rounding at the truncation boundary) |
| In simulation (`live_agreement` per sample) | live social costmap during recording vs. offline paint on the same grid from pedestrians interpolated at the costmap's stamp | see VALIDATION_REPORT.md |

## Label definition

- `social_cost/NNNNNN.npy`: `uint8 [200, 200]`, the **exact Arena layer** (`local_costmap` block) for the pedestrian states at the image stamp. Grid: 10 m × 10 m at 0.05 m, centred on `base_link`, x forward, y left, `array[j, i]` = cell (i, j), centre `(−5 + (i+0.5)·0.05, −5 + (j+0.5)·0.05)` m. 0 = no social cost, max 252.
- `social_cost_membership/NNNNNN.npy`: same grid and function; pedestrian interaction membership is completed from humansim's `InteractionFrame` (every member of an ACTIVE interaction, queued ones included, takes its id and type). This is the **minimal queue term**: the layer's existing group term with its existing `QUEUE_USE` weight 1.0, applied to the people actually in the queue. It differs from `social_cost` only where queued members exist. Nothing in the layer's mathematics or in humansim was changed.
- Group walking keeps weight 0 in both channels. A group-walk weight is a parameter choice for the study (task.txt "a group-walk weight") and can be set in the overlay or in a label config without code changes.

Labels are recomputed from the recorded `pedestrian_states.jsonl` (states interpolated to the image stamp) and `robot_states.jsonl` (TF at the image stamp), never from the live grid, so ROS scheduling and costmap latency do not enter the labels.
