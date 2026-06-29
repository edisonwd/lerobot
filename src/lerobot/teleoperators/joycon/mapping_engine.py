#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configurable mapping engine: Joy-Con inputs → motor target positions.

Loads mapping rules from YAML and converts Joy-Con raw inputs (sticks,
buttons, IMU) into per-motor position targets. Pure function — no
hardware dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ── Input constants ─────────────────────────────────────────────────────────

VALID_INPUTS = {
    # Sticks
    "left_stick_x",
    "left_stick_y",
    "right_stick_x",
    "right_stick_y",
    # Buttons
    "a",
    "b",
    "x",
    "y",
    "l",
    "zl",
    "r",
    "zr",
    "dpad_up",
    "dpad_down",
    "dpad_left",
    "dpad_right",
    "plus",
    "minus",
    "home",
    "capture",
    "l_stick_press",
    "r_stick_press",
    # IMU
    "imu_tilt",
    "imu_roll",
    "imu_pitch",
    "imu_roll_fused",
    "gyro_pitch_delta",
    "gyro_roll_delta",
    # Side-specific SL/SR buttons (for mode switch and filter toggle)
    "sl_right",
    "sl_left",
    "sr_right",
    "sr_left",
}


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class MappingEntry:
    """One mapping rule: one Joy-Con input → one motor."""

    input: str
    motor: str
    control: str  # "incremental" | "absolute"
    speed: float = 1.5
    scale: float = 1.0
    invert: bool = False

    def __post_init__(self):
        if self.input not in VALID_INPUTS:
            raise ValueError(f"Unknown input: {self.input!r}. Valid: {sorted(VALID_INPUTS)}")
        if self.control not in ("incremental", "absolute"):
            raise ValueError(
                f"control must be 'incremental' or 'absolute', got {self.control!r}"
            )


@dataclass
class MetaControls:
    """Configurable meta-control button assignments."""

    speed_up: str = "dpad_up"
    fine_tune_toggle: str = "l_stick_press"
    # New fields:
    reset_to_center: str = "dpad_down"
    recalibrate_imu: str = "dpad_left"
    pose_lock: str = "dpad_right"
    mode_switch: str = "sl_right"
    filter_toggle: str = "sr_right"

    def __post_init__(self):
        field_names = (
            "speed_up", "fine_tune_toggle", "reset_to_center",
            "recalibrate_imu", "pose_lock", "mode_switch", "filter_toggle",
        )
        for name in field_names:
            val = getattr(self, name)
            if val not in VALID_INPUTS:
                raise ValueError(f"meta_controls.{name}: unknown input {val!r}")


# ── Built-in default mapping ────────────────────────────────────────────────

_DEFAULT_MAPPINGS = [
    MappingEntry("left_stick_x", "shoulder_pan", "incremental", speed=1.5),
    MappingEntry("left_stick_y", "shoulder_lift", "incremental", speed=1.5, invert=True),
    MappingEntry("l", "elbow_flex", "incremental", speed=1.5),
    MappingEntry("zl", "elbow_flex", "incremental", speed=1.5, invert=True),
    MappingEntry("imu_tilt", "wrist_flex", "absolute", scale=0.5),
    MappingEntry("imu_roll", "wrist_roll", "absolute", scale=0.5),
    MappingEntry("a", "gripper", "incremental", speed=3.0),
    MappingEntry("b", "gripper", "incremental", speed=3.0, invert=True),
]


# ── Mapping engine ──────────────────────────────────────────────────────────


class MappingEngine:
    """Convert Joy-Con raw inputs to motor target positions."""

    def __init__(
        self,
        mappings: list[MappingEntry],
        meta_controls: MetaControls,
        joint_limits: dict[str, tuple[float, float]],
        presets: dict[str, dict[str, float]] | None = None,
    ):
        self.mappings = mappings
        self.meta_controls = meta_controls
        self.joint_limits = joint_limits
        self.presets: dict[str, dict[str, float]] = presets or {}

        for entry in mappings:
            if entry.motor not in joint_limits:
                raise ValueError(
                    f"Motor {entry.motor!r} not found in joint_limits. "
                    f"Available: {sorted(joint_limits.keys())}"
                )

    @property
    def motors(self) -> list[str]:
        """Return all mapped motor names (deduplicated, order preserved)."""
        seen: set[str] = set()
        result: list[str] = []
        for entry in self.mappings:
            if entry.motor not in seen:
                seen.add(entry.motor)
                result.append(entry.motor)
        return result

    def compute_targets(
        self,
        input_state: dict[str, float | bool],
        current_targets: dict[str, float],
        speed_multiplier: float = 1.0,
        fine_tune: bool = False,
    ) -> dict[str, float]:
        """Compute new motor target positions from Joy-Con inputs.

        Args:
            input_state: Current Joy-Con input values.
                Sticks: float [-1, 1]. Buttons: bool. IMU: float degrees.
            current_targets: Current motor target positions.
            speed_multiplier: Speed scale (0.2–2.0).
            fine_tune: Whether fine-tune mode halves the speed.

        Returns:
            Updated motor target positions.
        """
        targets = dict(current_targets)
        fine_scale = 0.5 if fine_tune else 1.0

        for entry in self.mappings:
            raw = input_state.get(entry.input)
            if raw is None:
                continue

            value = float(raw)

            if entry.control == "incremental":
                if entry.invert:
                    value = -value
                delta = value * entry.speed * speed_multiplier * fine_scale
                lo, hi = self.joint_limits[entry.motor]
                targets[entry.motor] = max(lo, min(hi, targets[entry.motor] + delta))

            elif entry.control == "absolute":
                target = value * entry.scale
                if entry.invert:
                    target = -target
                lo, hi = self.joint_limits[entry.motor]
                targets[entry.motor] = max(lo, min(hi, target))

        return targets

    def apply_preset(
        self,
        preset_name: str,
        current_targets: dict[str, float],
    ) -> dict[str, float]:
        """Apply a named preset to current targets.

        Only motors listed in the preset are changed; others stay.
        All values are clamped to joint limits.

        Returns:
            Updated targets dict (new dict, original unchanged).
        """
        if preset_name not in self.presets:
            logger.warning("Unknown preset: %r", preset_name)
            return dict(current_targets)

        targets = dict(current_targets)
        for motor, value in self.presets[preset_name].items():
            if motor in self.joint_limits:
                lo, hi = self.joint_limits[motor]
                targets[motor] = max(lo, min(hi, value))
            else:
                logger.warning("Preset motor %r not in joint_limits, skipping.", motor)

        logger.info("Applied preset %r: %s", preset_name, self.presets[preset_name])
        return targets

    @classmethod
    def from_yaml(cls, path: str | Path, joint_limits: dict) -> MappingEngine:
        """Load mapping configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        mappings = []
        for item in data.get("mappings", []):
            mappings.append(MappingEntry(
                input=item["input"],
                motor=item["motor"],
                control=item["control"],
                speed=item.get("speed", 1.5),
                scale=item.get("scale", 1.0),
                invert=item.get("invert", False),
            ))

        meta_data = data.get("meta_controls", {})
        meta = MetaControls(
            speed_up=meta_data.get("speed_up", "dpad_up"),
            fine_tune_toggle=meta_data.get("fine_tune_toggle", "l_stick_press"),
            reset_to_center=meta_data.get("reset_to_center", "dpad_down"),
            recalibrate_imu=meta_data.get("recalibrate_imu", "dpad_left"),
            pose_lock=meta_data.get("pose_lock", "dpad_right"),
            mode_switch=meta_data.get("mode_switch", "sl_right"),
            filter_toggle=meta_data.get("filter_toggle", "sr_right"),
        )

        presets = data.get("presets", {})

        logger.info(
            "Loaded mapping from %s: %d entries, %d motors, %d presets",
            path, len(mappings), len(set(e.motor for e in mappings)), len(presets),
        )
        return cls(mappings, meta, joint_limits, presets=presets)

    @classmethod
    def default(cls, joint_limits: dict) -> MappingEngine:
        """Create engine with built-in default mapping."""
        return cls(list(_DEFAULT_MAPPINGS), MetaControls(), joint_limits, presets={})
