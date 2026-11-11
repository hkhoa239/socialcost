"""A Nav2 costmap that holds the Arena social layer and nothing else.

Runs nav2_costmap_2d's standalone Costmap2DROS node with plugins: [social_cost_layer],
so the grid on `<ns>/<name>/costmap_raw` is the layer's output alone, painted by the
plugin binary itself. Used as the live reference for the offline labels.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

import yaml

from .labels import LayerParams

LAYER_PLUGIN = "arena_social_cost_layer/SocialCostLayer"


def costmap_params(
    params: LayerParams,
    peds_topic: str,
    global_frame: str,
    robot_base_frame: str,
    size_m: float,
    resolution: float,
    rolling_window: bool = True,
    origin: tuple[float, float] = (0.0, 0.0),
    rate_hz: float = 5.0,
    use_sim_time: bool = True,
    name: str = "social_costmap",
) -> dict:
    layer = {"plugin": LAYER_PLUGIN, "enabled": True, "peds_topic": peds_topic, **params.to_dict()}
    return {
        name: {
            "ros__parameters": {
                "use_sim_time": use_sim_time,
                "global_frame": global_frame,
                "robot_base_frame": robot_base_frame,
                "rolling_window": rolling_window,
                "width": int(round(size_m)),
                "height": int(round(size_m)),
                "resolution": resolution,
                "origin_x": float(origin[0]),
                "origin_y": float(origin[1]),
                "update_frequency": rate_hz,
                "publish_frequency": rate_hz,
                "always_send_full_costmap": True,
                "track_unknown_space": False,
                "robot_radius": 0.2,
                "transform_tolerance": 0.5,
                "plugins": ["social_cost_layer"],
                "social_cost_layer": layer,
            }
        }
    }


class LiveSocialCostmap:
    """Context manager running the social-only costmap as a subprocess and activating it."""

    def __init__(self, param_doc: dict, namespace: str = "/socialcost", log_path: str | None = None) -> None:
        self.name = next(iter(param_doc))
        self.namespace = namespace.rstrip("/") or ""
        self.param_doc = param_doc
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._param_file: str | None = None

    @property
    def node(self) -> str:
        return f"{self.namespace}/{self.name}"

    @property
    def raw_topic(self) -> str:
        # Costmap2DROS publishes in its namespace, not under its node name.
        return f"{self.namespace}/costmap_raw"

    def __enter__(self) -> "LiveSocialCostmap":
        fd, self._param_file = tempfile.mkstemp(prefix="social_costmap_", suffix=".yaml")
        # A params file keyed by the bare node name does not match a namespaced node.
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump({self.node: self.param_doc[self.name]}, f)
        out = open(self.log_path, "a") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            ["ros2", "run", "nav2_costmap_2d", "nav2_costmap_2d", "--ros-args",
             "-r", f"__node:={self.name}", "-r", f"__ns:={self.namespace or '/'}",
             "--params-file", self._param_file],
            stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            for transition in ("configure", "activate"):
                self._lifecycle(transition)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    _TARGET = {"configure": "inactive", "activate": "active"}

    def _state(self) -> str:
        r = subprocess.run(["ros2", "lifecycle", "get", self.node], capture_output=True, text=True, timeout=30)
        return r.stdout.strip().split(" ")[0] if r.returncode == 0 else ""

    def _lifecycle(self, transition: str, timeout_s: float = 180.0) -> None:
        # A `set` that times out on our side may still complete in the node, so the
        # node's own state decides, not the reply of the last call.
        deadline = time.monotonic() + timeout_s
        last = ""
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"social costmap exited with {self.proc.returncode} before {transition}")
            state = self._state()
            if state == self._TARGET[transition]:
                return
            if state in ("unconfigured", "inactive"):
                try:
                    r = subprocess.run(["ros2", "lifecycle", "set", self.node, transition], capture_output=True, text=True, timeout=60)
                    last = (r.stdout + r.stderr).strip()
                except subprocess.TimeoutExpired:
                    last = "set timed out"
            else:
                last = f"state {state!r}"
            time.sleep(1.0)
        raise TimeoutError(f"lifecycle {transition} of {self.node} failed: {last}")

    def __exit__(self, *exc) -> None:
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, 2)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, 9)
        if self._param_file:
            os.unlink(self._param_file)
