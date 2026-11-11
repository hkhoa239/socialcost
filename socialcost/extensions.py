"""extended/2.0: arena-native plus documented, switchable extensions.

  membership-completion/1.0   queued members join their interaction (as extended-membership/1.1)
  halt-gesture/1.0            a raised stop hand adds a wedge of cost ahead of the person
  mobility-aid/1.0-hypothesis wheelchair users get a scaled personal space (a hypothesis to evaluate,
                              off unless enabled in the extension config)

Every term is painted on the same grid and combined with the native label by the layer's own
operator, a per-cell maximum, so extended >= native everywhere and equals it when no extension
applies. Parameters live in configs/labels/extended_v2.yaml and are stored with each dataset.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass

import numpy as np
import yaml

from .dataset import with_membership
from .labels import GridSpec, LayerParams, Ped, paint
from .reference import cell_centres

VERSION = "2.0"


@dataclass(frozen=True)
class ExtParams:
    membership: bool = True
    halt: bool = True
    halt_weight: float = 1.0  # normalized amplitude, same scale as interaction_weights
    halt_length: float = 2.5  # m ahead of the person
    halt_sigma: float = 1.2  # along-axis falloff, layer's super-Gaussian with falloff_shape
    halt_half_angle_deg: float = 30.0
    halt_base_halfwidth: float = 0.5  # m, wedge half-width at the person
    mobility: bool = False
    mobility_markers: tuple[str, ...] = ("wheelchair",)
    mobility_radius_scale: float = 1.5
    mobility_sigma_scale: float = 1.5

    @classmethod
    def load(cls, path: str) -> "ExtParams":
        d = yaml.safe_load(open(path)) or {}
        if "mobility_markers" in d:
            d["mobility_markers"] = tuple(d["mobility_markers"])
        return cls(**d)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["mobility_markers"] = list(self.mobility_markers)
        return d


def is_mobility_aid(p: dict, ext: ExtParams) -> bool:
    """From the body asset name only. agent_type is often a yaml path whose directory names the
    scenario (e.g. ..._wheelchair_off/person.yaml), so it is used only when it is a bare type name."""
    body = os.path.basename(str(p.get("model_uri", ""))).lower()
    agent = str(p.get("agent_type", ""))
    agent = agent.lower() if "/" not in agent else ""
    return any(m in body or m in agent for m in ext.mobility_markers)


def halt_direction(p: dict) -> tuple[float, float] | None:
    """Unit vector from the person toward its raised-hand target, None without a halt gesture."""
    for g in p.get("gestures") or []:
        if not isinstance(g, dict) or not str(g.get("slot", "")).startswith("halt"):
            continue
        at = g.get("at") or {}
        dx, dy = float(at.get("x", 0.0)) - p["x"], float(at.get("y", 0.0)) - p["y"]
        n = math.hypot(dx, dy)
        if n > 1e-3:
            return dx / n, dy / n
        return math.cos(p["yaw"]), math.sin(p["yaw"])
    return None


def halt_term(params: LayerParams, ext: ExtParams, p: dict, grid: GridSpec, pose) -> np.ndarray:
    d = halt_direction(p)
    if d is None:
        return np.zeros((grid.height, grid.width), np.uint8)
    wx, wy = cell_centres(grid, pose)
    rx, ry = wx - p["x"], wy - p["y"]
    along = rx * d[0] + ry * d[1]
    lateral = np.abs(-rx * d[1] + ry * d[0])
    inside = (along >= 0) & (along <= ext.halt_length) & (lateral <= ext.halt_base_halfwidth + along * math.tan(math.radians(ext.halt_half_angle_deg)))
    v = ext.halt_weight * np.exp(-np.power(along ** 2 / (2 * ext.halt_sigma ** 2), params.falloff_shape))
    cell = (np.clip(v * params.cost_scale + params.cost_bias, 0, 1) * params.max_cost_value).astype(np.uint8)
    return np.where(inside, cell, 0).astype(np.uint8)


def paint_extended(params: LayerParams, ext: ExtParams, states: list[dict], frame: dict | None, grid: GridSpec, pose) -> tuple[np.ndarray, dict]:
    """extended/2.0 label and which extensions fired."""
    s = with_membership(states, frame) if ext.membership else states
    fired = {"membership": sum(1 for a, b in zip(states, s) if a["interaction_id"] != b["interaction_id"]), "halt": 0, "mobility": 0}
    peds = [Ped(p["x"], p["y"], p["animation_state"], p["interaction_id"], p["interaction_type"]) for p in s]
    out = paint(params, peds, grid, pose)
    if ext.mobility:
        mob = dataclasses.replace(params, personal_space_radius=params.personal_space_radius * ext.mobility_radius_scale,
                                  personal_space_sigma=params.personal_space_sigma * ext.mobility_sigma_scale)
        aids = [Ped(p["x"], p["y"], p["animation_state"]) for p in s if is_mobility_aid(p, ext)]
        if aids:
            fired["mobility"] = len(aids)
            out = np.maximum(out, paint(mob, aids, grid, pose))
    if ext.halt:
        for p in s:
            h = halt_term(params, ext, p, grid, pose)
            if h.any():
                fired["halt"] += 1
                out = np.maximum(out, h)
    return out, fired


def definition(ext: ExtParams) -> dict:
    return {"name": "extended", "version": VERSION, "extension_params": ext.to_dict(),
            "definition": "arena-native cost functions, parameters and max operator, plus the enabled extensions below, each combined by per-cell maximum",
            "extensions": [
                {"id": "membership-completion/1.0", "enabled": ext.membership, "effect": "as extended-membership/1.1"},
                {"id": "halt-gesture/1.0", "enabled": ext.halt,
                 "effect": f"person with a halt/halt_l/halt_r gesture: wedge from the person toward the gesture target, length {ext.halt_length} m, "
                           f"half-width {ext.halt_base_halfwidth} m + along*tan({ext.halt_half_angle_deg} deg), value w*exp(-(along^2/2 sigma^2)^falloff_shape), "
                           f"w={ext.halt_weight}, sigma={ext.halt_sigma}, mapped by the layer's cell_value"},
                {"id": "mobility-aid/1.0-hypothesis", "enabled": ext.mobility,
                 "effect": f"agent_type/model containing {list(ext.mobility_markers)}: personal space radius x{ext.mobility_radius_scale}, sigma x{ext.mobility_sigma_scale}. "
                           "A hypothesis to evaluate, not an established rule."},
            ]}
