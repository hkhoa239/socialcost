# Label definitions

Every dataset carries both label channels side by side and a machine-readable copy of this page in `<root>/labels.json`; every `episode_metadata.json` names the versions it holds (`label_definitions`). Neither channel replaces the other: training code picks one by directory name.

| Directory | Name / version | One line |
|---|---|---|
| `social_cost/` | **arena-native/1.0** | Exact output of `arena_social_cost_layer` for the pedestrian states as Arena publishes them |
| `social_cost_membership/` | **extended-membership/1.0** | arena-native with interaction membership completed from humansim's `InteractionFrame` (adds queues) |

## Shared definition

| Item | Value |
|---|---|
| Grid | 200 × 200 cells, 0.05 m, 10 m × 10 m centred on `base_link` at the image stamp; x forward, y left; `label[j, i]`, cell centre (−5 + (i+½)·0.05, −5 + (j+½)·0.05) m |
| Type / range | `uint8`, 0 (no social cost) … `max_cost_value` (252) |
| Time reference | Pedestrian states interpolated to the RGB-D image stamp; robot pose = TF map→base_link at that stamp |
| Parameters | `social_cost_layer.yaml`, block `local_costmap`: radius 1.0, σ 0.9, group radius 2.0, group σ 0.85, falloff 3.0, max 252, scale 0.8, bias 0.05, weights `[0.9, 1.2, 0, 0.8, 0.8, 0.8, 1.0, 0.6, 0.8, 0.8, 1.5, 0.8, 0.0]` |
| Personal-space term | per pedestrian, cells with centre distance d ≤ 1.0 m: A·exp(−(d²/2σ²)^3); (σ, A) = (0.9, 0.5) by default, (1.5, 0.75) SURPRISED/CURIOUS, (2.4, 1.0) PANIC/THREATENING. Isotropic: heading is not used. |
| Interaction term | per pair sharing an `interaction_id ≥ 0`, cells within 2.0 m of the segment between them: w·exp(−(d_seg²/2·0.85²)^3), w = weight of the group's interaction type; w = 0 paints nothing. All pairs, not only neighbours. |
| Cell value | ⌊clamp(v·0.8 + 0.05, 0, 1)·252⌋ |
| Combination | per-cell **maximum** over all terms, from 0. Not a sum: overlapping people or groups never exceed the strongest single term. |
| Implementation | `libsocialcost_grid.so` over `arena_social_cost_layer/cost_math.hpp`; agreement with the plugin binary in `docs/VALIDATION_REPORT.md` |

Characteristic values: personal-space peak 113 (default state); TALK_TO capsule centre 194; QUEUE_USE 214; GROUP_CONVERSATION and HUG saturate at 252.

## arena-native/1.0

Inputs: `arena_people_msgs/Pedestrian` x, y, `animation_state`, `interaction_id`, `interaction_type`, exactly as published. humansim publishes `interaction_id` only for **ACTIVE participants** (`InteractionManager.active_interaction`), so:

- TALK_TO, GROUP_CONVERSATION, WAVE_AT, HUG, SHAKE_HAND, SERVICE, BLOCK: interaction term present while the interaction is ACTIVE with ≥ 2 participants.
- QUEUE_USE, USE, SIT_ON, LIE_ON: one participant by default (object capacity 1), so **no interaction term**; queued people publish −1.
- GROUP_WALK: weight 0.0 by design (Arena commit `edf7884`, "social cost layer: no group capsule for walkers"), personal space only.

## extended-membership/1.0

Inputs: arena-native inputs plus the `InteractionFrame` in force at the image stamp (`interactions.jsonl`).

Single extension, `membership-completion/1.0`: every member of an ACTIVE interaction, participant or QUEUED, whose published `interaction_id` is −1 takes that interaction's id and type. Cost functions, weights, parameters and the max operator are unchanged, so:

- a queue gets the layer's own QUEUE_USE capsule (weight 1.0) between all members, served person included;
- any other queueable kind with waiting people (USE, SIT_ON, LIE_ON, SERVICE) would likewise get its own weight's capsule; none occur in the current scenes;
- extended ≥ native cell-wise, and equal wherever nobody is queued.

Feeding the completed states to the plugin binary reproduces the extended label exactly (`G_offline_vs_live_identical_states`, `+membership` cases), so the extension is "Arena's layer given the full membership", not a new cost function.

## Not in either label

- Group walking: no group-level term. Personal space covers the space between walkers whenever they are closer than 2 m (all observed pairs: 0.87–1.86 m); see `semantic_report.md`.
- Heading, gaze, gestures, velocity: unused by the layer.
- Interaction region of a single active participant (e.g. a person using an object): none.

## Changing a definition

A new term or weight is a new label channel with its own name and version (e.g. `extended-groupwalk/1.0`), written next to the existing ones, never in place of them. Bump the version when parameters or semantics change; the parameters are stored per episode, and `scripts/label_manifest.py` refuses to label a dataset whose episodes disagree.
