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
    speed_down: str = "dpad_down"
    fine_tune_toggle: str = "l_stick_press"

    def __post_init__(self):
        for name in ("speed_up", "speed_down", "fine_tune_toggle"):
            val = getattr(self, name)
            if val not in VALID_INPUTS:
                raise ValueError(f"meta_controls.{name}: unknown input {val!r}")


# ── Default mappings (placeholder for YAML loading in future task) ──────────

_DEFAULT_MAPPINGS: list[MappingEntry] = []


# ── Mapping engine ──────────────────────────────────────────────────────────


class MappingEngine:
    """Convert Joy-Con raw inputs to motor target positions."""

    def __init__(
        self,
        mappings: list[MappingEntry],
        meta_controls: MetaControls,
        joint_limits: dict[str, tuple[float, float]],
    ):
        self.mappings = mappings
        self.meta_controls = meta_controls
        self.joint_limits = joint_limits

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
