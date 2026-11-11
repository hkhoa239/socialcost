#!/usr/bin/env python3
"""Record one episode of the running Arena env (RGB-D, robot pose, pedestrians) into --out."""

from __future__ import annotations

import argparse
import json
import os
import sys

import rclpy
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from socialcost.recorder import EpisodeRecorder, RecorderConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensors", required=True)
    ap.add_argument("--sample-hz", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--fresh-margin", type=float, default=0.5)
    ap.add_argument("--live-costmap-topic", default=None)
    ap.add_argument("--min-episode-id", type=int, default=1)
    ap.add_argument("--metadata", default="{}")
    args = ap.parse_args()
    with open(args.sensors) as f:
        s = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)
    cfg = RecorderConfig(
        out_dir=args.out, rgb_topic=s["rgb_topic"], depth_topic=s["depth_topic"], camera_info_topic=s["camera_info_topic"],
        odom_topic=s["odom_topic"], peds_topic=s["peds_topic"], episode_topic=s["episode_topic"],
        interactions_topic=s["interactions_topic"], agent_meta_topic=s["agent_meta_topic"],
        map_frame=s["map_frame"], base_frame=s["base_frame"], camera_frame=s["camera_frame"],
        sample_hz=args.sample_hz, duration_s=args.duration, warmup_s=args.warmup, fresh_margin_s=args.fresh_margin,
        live_costmap_topic=args.live_costmap_topic, metadata=json.loads(args.metadata), min_episode_id=args.min_episode_id,
    )
    rclpy.init()
    node = EpisodeRecorder(cfg)
    try:
        stats = node.run()
        if node.state != "done":
            node._finish(stats.get("end_reason") or "interrupted")
    except KeyboardInterrupt:
        node._finish("interrupted")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0 if node.samples > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
