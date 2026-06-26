# Joy-Con Configurable Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Joy-Con → SO-101 motor mapping configurable via YAML, with the mapping engine integrated into `JoyConTeleop.get_action()` so it returns motor-level position targets directly.

**Architecture:** A new `MappingEngine` class (pure function, no hardware dependency) loads mapping rules from YAML and converts Joy-Con raw inputs into motor target positions. `JoyConTeleop` integrates the engine, replacing its delta-control action output with motor-level position commands. The example script simplifies to a thin connect-and-forward loop.

**Tech Stack:** Python 3.12+, PyYAML (already transitively available), pytest, existing lerobot/draccus/hidapi stack

**Spec:** `docs/superpowers/specs/2026-06-26-joycon-configurable-mapping-design.md`

## Global Constraints

- `joycon_utils.py` must NOT be modified except for adding `get_raw_left_stick()` and `get_raw_right_stick()` accessor methods
- All 78 existing tests in `tests/teleoperators/test_joycon.py` must continue to pass
- YAML loading uses `yaml.safe_load` only (never `yaml.load`)
- Joint limits are read from the robot config, never hard-coded in mapping YAML
- Meta controls `home`, `plus`, `minus` remain hard-coded and non-configurable
- New tests go in `tests/teleoperators/test_joycon.py` (existing file, appended)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/lerobot/teleoperators/joycon/mapping_engine.py` | Create | MappingEntry, MetaControls, MappingEngine classes |
| `src/lerobot/teleoperators/joycon/configuration_joycon.py` | Modify | Add `mapping_path` field |
| `src/lerobot/teleoperators/joycon/joycon_utils.py` | Modify | Add `get_raw_left_stick()`, `get_raw_right_stick()` |
| `src/lerobot/teleoperators/joycon/teleop_joycon.py` | Modify | Integrate MappingEngine, rewrite `get_action()` |
| `src/lerobot/teleoperators/joycon/__init__.py` | Modify | Export MappingEngine |
| `src/lerobot/robots/so_follower/config_so_follower.py` | Modify | Add `joint_limits` field |
| `src/lerobot/robots/so_follower/so_follower.py` | Modify | Add `get_joint_limits()` |
| `examples/joycon_to_so101/teleoperate.py` | Rewrite | Simplified ~80 line script |
| `examples/joycon_to_so101/default_mapping.yaml` | Create | Reference YAML for users |
| `tests/teleoperators/test_joycon.py` | Extend | ~25 new tests for mapping engine + integration |
| `pyproject.toml` | Modify | Add `pyyaml` to `joycon` extra |

---

### Task 1: Add joint_limits to SOFollowerRobotConfig

**Files:**
- Modify: `src/lerobot/robots/so_follower/config_so_follower.py:45-49`
- Modify: `src/lerobot/robots/so_follower/so_follower.py` (add method)

**Interfaces:**
- Produces: `SOFollowerRobotConfig.joint_limits` (dict[str, tuple[float, float]])
- Produces: `SOFollower.get_joint_limits()` → dict[str, tuple[float, float]]

- [ ] **Step 1: Add joint_limits field to SOFollowerRobotConfig**

In `src/lerobot/robots/so_follower/config_so_follower.py`, replace the current `SOFollowerRobotConfig` class:

```python
@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-110.0, 110.0),
            "shoulder_lift": (-100.0, 100.0),
            "elbow_flex": (-97.0, 97.0),
            "wrist_flex": (-95.0, 95.0),
            "wrist_roll": (-157.0, 163.0),
            "gripper": (0.0, 100.0),
        }
    )
```

- [ ] **Step 2: Add get_joint_limits() method to SOFollower**

In `src/lerobot/robots/so_follower/so_follower.py`, add this method to the `SOFollower` class (after the `action_features` property, around line 82):

```python
    def get_joint_limits(self) -> dict[str, tuple[float, float]]:
        """Return {motor_name: (min_deg, max_deg)} from config."""
        return self.config.joint_limits
```

- [ ] **Step 3: Verify no import errors**

Run: `uv run python -c "from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig; c = SO100FollowerConfig(port='/dev/null'); print(c.joint_limits)"`
Expected: Prints the joint_limits dict with 6 motors

- [ ] **Step 4: Commit**

```bash
git add src/lerobot/robots/so_follower/config_so_follower.py src/lerobot/robots/so_follower/so_follower.py
git commit -m "feat(robot): add joint_limits to SOFollowerRobotConfig and get_joint_limits() method"
```

---

### Task 2: Add raw stick accessors to JoyConHIDController

**Files:**
- Modify: `src/lerobot/teleoperators/joycon/joycon_utils.py:392-401`

**Interfaces:**
- Produces: `JoyConHIDController.get_raw_left_stick()` → tuple[float, float]
- Produces: `JoyConHIDController.get_raw_right_stick()` → tuple[float, float]

These return normalized stick values after deadzone but WITHOUT speed/fine-tune multiplier applied. The MappingEngine applies its own speed in `compute_targets()`.

- [ ] **Step 1: Add get_raw_left_stick() and get_raw_right_stick()**

In `src/lerobot/teleoperators/joycon/joycon_utils.py`, add these methods after `get_left_stick()` (after line 401):

```python
    def get_raw_left_stick(self) -> tuple[float, float]:
        """Get raw normalized left stick values (after deadzone, without speed/fine-tune).

        Returns:
            (stick_x, stick_y) in range [-1.0, 1.0].
        """
        if not self.stick_available:
            return 0.0, 0.0
        return self.left_x, self.left_y

    def get_raw_right_stick(self) -> tuple[float, float]:
        """Get raw normalized right stick values (after deadzone, without speed/fine-tune).

        Returns:
            (stick_x, stick_y) in range [-1.0, 1.0].
        """
        if not self.stick_available:
            return 0.0, 0.0
        return self.right_x, self.right_y
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py -x -q`
Expected: All 78 tests pass

- [ ] **Step 3: Commit**

```bash
git add src/lerobot/teleoperators/joycon/joycon_utils.py
git commit -m "feat(joycon): add get_raw_left_stick() and get_raw_right_stick() accessors"
```

---

### Task 3: Create MappingEngine — data classes and compute_targets

**Files:**
- Create: `src/lerobot/teleoperators/joycon/mapping_engine.py`
- Test: `tests/teleoperators/test_joycon.py` (append new test classes)

**Interfaces:**
- Consumes: nothing from other tasks (standalone module)
- Produces: `MappingEntry`, `MetaControls`, `MappingEngine` classes
- Produces: `MappingEngine.compute_targets(input_state, current_targets, speed_multiplier, fine_tune)` → dict[str, float]
- Produces: `MappingEngine.motors` → list[str]

- [ ] **Step 1: Write tests for MappingEntry and MetaControls validation**

Append to `tests/teleoperators/test_joycon.py`:

```python
# ── MappingEngine tests ────────────────────────────────────────────────────

from lerobot.teleoperators.joycon.mapping_engine import (
    MappingEntry,
    MappingEngine,
    MetaControls,
    VALID_INPUTS,
    _DEFAULT_MAPPINGS,
)

SO101_JOINT_LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-97.0, 97.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.0, 163.0),
    "gripper": (0.0, 100.0),
}


class TestMappingEntry:
    def test_valid_entry(self):
        e = MappingEntry("left_stick_x", "shoulder_pan", "incremental", speed=1.5)
        assert e.input == "left_stick_x"
        assert e.motor == "shoulder_pan"
        assert e.control == "incremental"
        assert e.speed == 1.5
        assert e.invert is False

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="Unknown input"):
            MappingEntry("fake_input", "shoulder_pan", "incremental")

    def test_invalid_control_raises(self):
        with pytest.raises(ValueError, match="control must be"):
            MappingEntry("left_stick_x", "shoulder_pan", "bad_mode")


class TestMetaControls:
    def test_defaults(self):
        m = MetaControls()
        assert m.speed_up == "dpad_up"
        assert m.speed_down == "dpad_down"
        assert m.fine_tune_toggle == "l_stick_press"

    def test_custom_buttons(self):
        m = MetaControls(speed_up="r", speed_down="zr", fine_tune_toggle="r_stick_press")
        assert m.speed_up == "r"

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="unknown input"):
            MetaControls(speed_up="fake_button")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py::TestMappingEntry -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'lerobot.teleoperators.joycon.mapping_engine'`

- [ ] **Step 3: Implement MappingEntry, MetaControls, VALID_INPUTS, and MappingEngine core**

Create `src/lerobot/teleoperators/joycon/mapping_engine.py`:

```python
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
    "left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y",
    # Buttons
    "a", "b", "x", "y", "l", "zl", "r", "zr",
    "dpad_up", "dpad_down", "dpad_left", "dpad_right",
    "plus", "minus", "home", "capture",
    "l_stick_press", "r_stick_press",
    # IMU
    "imu_tilt", "imu_roll",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py::TestMappingEntry tests/teleoperators/test_joycon.py::TestMetaControls -v`
Expected: All PASS

- [ ] **Step 5: Write compute_targets tests**

Append to `tests/teleoperators/test_joycon.py`:

```python
class TestComputeTargets:
    def _make_engine(self, mappings=None):
        if mappings is None:
            mappings = [
                MappingEntry("left_stick_x", "shoulder_pan", "incremental", speed=1.5),
            ]
        return MappingEngine(mappings, MetaControls(), SO101_JOINT_LIMITS)

    def test_incremental_stick(self):
        engine = self._make_engine()
        result = engine.compute_targets(
            {"left_stick_x": 0.5},
            {"shoulder_pan": 10.0},
        )
        assert result["shoulder_pan"] == pytest.approx(10.75)  # 10 + 0.5*1.5

    def test_incremental_stick_inverted(self):
        engine = self._make_engine([
            MappingEntry("left_stick_x", "shoulder_pan", "incremental", speed=1.5, invert=True),
        ])
        result = engine.compute_targets(
            {"left_stick_x": 0.5},
            {"shoulder_pan": 10.0},
        )
        assert result["shoulder_pan"] == pytest.approx(9.25)  # 10 - 0.5*1.5

    def test_incremental_button_press(self):
        engine = self._make_engine([
            MappingEntry("l", "elbow_flex", "incremental", speed=1.5),
        ])
        result = engine.compute_targets(
            {"l": True},
            {"elbow_flex": 0.0},
        )
        assert result["elbow_flex"] == pytest.approx(1.5)  # 0 + 1.0*1.5

    def test_incremental_button_not_pressed(self):
        engine = self._make_engine([
            MappingEntry("l", "elbow_flex", "incremental", speed=1.5),
        ])
        result = engine.compute_targets(
            {"l": False},
            {"elbow_flex": 5.0},
        )
        assert result["elbow_flex"] == pytest.approx(5.0)  # 0 + 0.0*1.5

    def test_absolute_imu(self):
        engine = self._make_engine([
            MappingEntry("imu_tilt", "wrist_flex", "absolute", scale=0.5),
        ])
        result = engine.compute_targets(
            {"imu_tilt": 30.0},
            {"wrist_flex": 0.0},
        )
        assert result["wrist_flex"] == pytest.approx(15.0)  # 30*0.5

    def test_absolute_imu_inverted(self):
        engine = self._make_engine([
            MappingEntry("imu_tilt", "wrist_flex", "absolute", scale=0.5, invert=True),
        ])
        result = engine.compute_targets(
            {"imu_tilt": 30.0},
            {"wrist_flex": 0.0},
        )
        assert result["wrist_flex"] == pytest.approx(-15.0)

    def test_clamp_to_joint_limits(self):
        engine = self._make_engine([
            MappingEntry("left_stick_x", "shoulder_pan", "incremental", speed=100.0),
        ])
        result = engine.compute_targets(
            {"left_stick_x": 1.0},
            {"shoulder_pan": 100.0},
        )
        assert result["shoulder_pan"] == pytest.approx(110.0)  # clamped to max

    def test_clamp_lower_limit(self):
        engine = self._make_engine([
            MappingEntry("left_stick_x", "shoulder_pan", "incremental", speed=100.0, invert=True),
        ])
        result = engine.compute_targets(
            {"left_stick_x": 1.0},
            {"shoulder_pan": -100.0},
        )
        assert result["shoulder_pan"] == pytest.approx(-110.0)

    def test_multiple_inputs_same_motor(self):
        engine = self._make_engine([
            MappingEntry("l", "elbow_flex", "incremental", speed=1.5),
            MappingEntry("zl", "elbow_flex", "incremental", speed=1.5, invert=True),
        ])
        result = engine.compute_targets(
            {"l": True, "zl": True},
            {"elbow_flex": 10.0},
        )
        assert result["elbow_flex"] == pytest.approx(10.0)  # cancel out

    def test_opposing_buttons_cancel(self):
        engine = self._make_engine([
            MappingEntry("a", "gripper", "incremental", speed=3.0),
            MappingEntry("b", "gripper", "incremental", speed=3.0, invert=True),
        ])
        result = engine.compute_targets(
            {"a": True, "b": True},
            {"gripper": 50.0},
        )
        assert result["gripper"] == pytest.approx(50.0)

    def test_unmapped_motor_stays(self):
        engine = self._make_engine([
            MappingEntry("left_stick_x", "shoulder_pan", "incremental", speed=1.5),
        ])
        result = engine.compute_targets(
            {"left_stick_x": 0.5},
            {"shoulder_pan": 10.0, "elbow_flex": 20.0},
        )
        assert result["elbow_flex"] == pytest.approx(20.0)  # untouched

    def test_speed_multiplier(self):
        engine = self._make_engine()
        result = engine.compute_targets(
            {"left_stick_x": 0.5},
            {"shoulder_pan": 10.0},
            speed_multiplier=2.0,
        )
        assert result["shoulder_pan"] == pytest.approx(11.5)  # 10 + 0.5*1.5*2.0

    def test_fine_tune_halves_speed(self):
        engine = self._make_engine()
        result = engine.compute_targets(
            {"left_stick_x": 0.5},
            {"shoulder_pan": 10.0},
            fine_tune=True,
        )
        assert result["shoulder_pan"] == pytest.approx(10.375)  # 10 + 0.5*1.5*0.5

    def test_invalid_motor_raises(self):
        with pytest.raises(ValueError, match="Motor 'fake_motor' not found"):
            MappingEngine(
                [MappingEntry("left_stick_x", "fake_motor", "incremental")],
                MetaControls(),
                SO101_JOINT_LIMITS,
            )

    def test_motors_property(self):
        engine = self._make_engine([
            MappingEntry("left_stick_x", "shoulder_pan", "incremental"),
            MappingEntry("l", "elbow_flex", "incremental"),
            MappingEntry("zl", "elbow_flex", "incremental"),
        ])
        assert engine.motors == ["shoulder_pan", "elbow_flex"]

    def test_missing_input_skipped(self):
        engine = self._make_engine()
        result = engine.compute_targets(
            {},  # no left_stick_x in input
            {"shoulder_pan": 10.0},
        )
        assert result["shoulder_pan"] == pytest.approx(10.0)
```

- [ ] **Step 6: Run compute_targets tests**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py::TestComputeTargets -v`
Expected: All 16 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/lerobot/teleoperators/joycon/mapping_engine.py tests/teleoperators/test_joycon.py
git commit -m "feat(joycon): add MappingEngine core with compute_targets()"
```

---

### Task 4: Add YAML loading and default mapping to MappingEngine

**Files:**
- Modify: `src/lerobot/teleoperators/joycon/mapping_engine.py` (append methods)
- Test: `tests/teleoperators/test_joycon.py` (append tests)

**Interfaces:**
- Consumes: `MappingEngine.__init__`, `MappingEntry`, `MetaControls` (from Task 3)
- Produces: `MappingEngine.from_yaml(path, joint_limits)` → MappingEngine
- Produces: `MappingEngine.default(joint_limits)` → MappingEngine
- Produces: `_DEFAULT_MAPPINGS` list

- [ ] **Step 1: Write tests for YAML loading and default mapping**

Append to `tests/teleoperators/test_joycon.py`:

```python
class TestMappingEngineFromYaml:
    def test_load_from_yaml(self, tmp_path):
        yaml_content = """\
mappings:
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
    speed: 2.0
    invert: false
  - input: imu_tilt
    motor: wrist_flex
    control: absolute
    scale: 0.3
meta_controls:
  speed_up: r
  speed_down: zr
  fine_tune_toggle: r_stick_press
"""
        yaml_file = tmp_path / "test_mapping.yaml"
        yaml_file.write_text(yaml_content)
        engine = MappingEngine.from_yaml(yaml_file, SO101_JOINT_LIMITS)
        assert len(engine.mappings) == 2
        assert engine.mappings[0].speed == 2.0
        assert engine.mappings[1].scale == 0.3
        assert engine.meta_controls.speed_up == "r"
        assert engine.meta_controls.fine_tune_toggle == "r_stick_press"

    def test_load_yaml_without_meta_uses_defaults(self, tmp_path):
        yaml_content = """\
mappings:
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
"""
        yaml_file = tmp_path / "test_mapping.yaml"
        yaml_file.write_text(yaml_content)
        engine = MappingEngine.from_yaml(yaml_file, SO101_JOINT_LIMITS)
        assert engine.meta_controls.speed_up == "dpad_up"
        assert engine.meta_controls.speed_down == "dpad_down"

    def test_default_mapping_covers_all_motors(self):
        engine = MappingEngine.default(SO101_JOINT_LIMITS)
        assert set(engine.motors) == {
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py::TestMappingEngineFromYaml -x -q`
Expected: FAIL with `AttributeError: type object 'MappingEngine' has no attribute 'from_yaml'`

- [ ] **Step 3: Add from_yaml(), default(), and _DEFAULT_MAPPINGS to mapping_engine.py**

Append to `src/lerobot/teleoperators/joycon/mapping_engine.py` (inside the `MappingEngine` class, after `compute_targets`):

```python
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
            speed_down=meta_data.get("speed_down", "dpad_down"),
            fine_tune_toggle=meta_data.get("fine_tune_toggle", "l_stick_press"),
        )

        logger.info(
            "Loaded mapping from %s: %d entries, %d motors",
            path, len(mappings), len(set(e.motor for e in mappings)),
        )
        return cls(mappings, meta, joint_limits)

    @classmethod
    def default(cls, joint_limits: dict) -> MappingEngine:
        """Create engine with built-in default mapping."""
        return cls(list(_DEFAULT_MAPPINGS), MetaControls(), joint_limits)


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py::TestMappingEngineFromYaml -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py -x -q`
Expected: All tests PASS (78 original + new ones)

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/teleoperators/joycon/mapping_engine.py tests/teleoperators/test_joycon.py
git commit -m "feat(joycon): add MappingEngine.from_yaml() and default mapping"
```

---

### Task 5: Add mapping_path to JoyConTeleopConfig

**Files:**
- Modify: `src/lerobot/teleoperators/joycon/configuration_joycon.py:55-64`

**Interfaces:**
- Produces: `JoyConTeleopConfig.mapping_path: str | None`

- [ ] **Step 1: Add mapping_path field**

In `src/lerobot/teleoperators/joycon/configuration_joycon.py`, add the field after `fine_tune_multiplier`:

```python
    fine_tune_multiplier: float = 0.5
    mapping_path: str | None = None
```

Also update the docstring to include:

```
        mapping_path: Path to YAML mapping file. None uses built-in default.
```

- [ ] **Step 2: Verify config works**

Run: `uv run python -c "from lerobot.teleoperators.joycon import JoyConTeleopConfig; c = JoyConTeleopConfig(); print(c.mapping_path)"`
Expected: `None`

- [ ] **Step 3: Run existing tests**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py::TestJoyConConfig -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/lerobot/teleoperators/joycon/configuration_joycon.py
git commit -m "feat(joycon): add mapping_path to JoyConTeleopConfig"
```

---

### Task 6: Update __init__.py exports

**Files:**
- Modify: `src/lerobot/teleoperators/joycon/__init__.py`

**Interfaces:**
- Produces: public export of `MappingEngine` from the `joycon` package

- [ ] **Step 1: Add MappingEngine to exports**

Replace `src/lerobot/teleoperators/joycon/__init__.py` with:

```python
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

from .configuration_joycon import JoyConMode, JoyConTeleopConfig
from .mapping_engine import MappingEngine
from .teleop_joycon import JoyConTeleop

__all__ = ["JoyConTeleop", "JoyConTeleopConfig", "JoyConMode", "MappingEngine"]
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from lerobot.teleoperators.joycon import MappingEngine; print(MappingEngine)"`
Expected: `<class 'lerobot.teleoperators.joycon.mapping_engine.MappingEngine'>`

- [ ] **Step 3: Commit**

```bash
git add src/lerobot/teleoperators/joycon/__init__.py
git commit -m "feat(joycon): export MappingEngine from package"
```

---

### Task 7: Integrate MappingEngine into JoyConTeleop

**Files:**
- Modify: `src/lerobot/teleoperators/joycon/teleop_joycon.py`
- Test: `tests/teleoperators/test_joycon.py` (append integration tests)

**Interfaces:**
- Consumes: `MappingEngine` (from Task 3-4), `JoyConTeleopConfig.mapping_path` (from Task 5)
- Consumes: `JoyConHIDController.get_raw_left_stick()`, `get_raw_right_stick()` (from Task 2)
- Consumes: `JoyConHIDController.buttons` (dict[str, bool], existing)
- Consumes: `JoyConHIDController.get_wrist_angles()` → tuple[float, float] (existing)
- Consumes: `JoyConHIDController.speed_multiplier` (float, existing)
- Consumes: `JoyConHIDController.fine_tune` (bool, existing)
- Produces: `JoyConTeleop.connect(joint_limits=...)` — loads MappingEngine
- Produces: `JoyConTeleop.init_targets(observation)` — sets initial positions
- Produces: `JoyConTeleop.get_action()` → dict with `{motor}.pos` keys
- Produces: `JoyConTeleop.action_features` — dynamic based on mapping

- [ ] **Step 1: Write integration tests**

Append to `tests/teleoperators/test_joycon.py`:

```python
class TestJoyConTeleopMappingIntegration:
    """Integration tests for JoyConTeleop with MappingEngine.

    Uses mock controller — no real hardware needed.
    """

    def _make_teleop_with_mock_controller(self, mapping_path=None):
        """Create a JoyConTeleop with a mock controller and default mapping."""
        config = JoyConTeleopConfig(mode=JoyConMode.SINGLE_LEFT, mapping_path=mapping_path)
        teleop = JoyConTeleop(config)

        # Create a mock controller
        from unittest.mock import MagicMock
        ctrl = MagicMock()
        ctrl.mode = JoyConMode.SINGLE_LEFT
        ctrl.devices = {"left": MagicMock()}
        ctrl.stick_available = True
        ctrl.buttons = {
            "a": False, "b": False, "x": False, "y": False,
            "l": False, "zl": False, "r": False, "zr": False,
            "up": False, "down": False, "left": False, "right": False,
            "plus": False, "minus": False, "home": False, "capture": False,
            "l_stick_press": False, "r_stick_press": False,
        }
        ctrl.get_raw_left_stick.return_value = (0.0, 0.0)
        ctrl.get_raw_right_stick.return_value = (0.0, 0.0)
        ctrl.get_wrist_angles.return_value = (0.0, 0.0)
        ctrl.speed_multiplier = 1.0
        ctrl.fine_tune = False
        ctrl.update = MagicMock()

        teleop.controller = ctrl
        teleop.mapping_engine = MappingEngine.default(SO101_JOINT_LIMITS)
        teleop._current_targets = {m: 0.0 for m in teleop.mapping_engine.motors}
        teleop._prev_speed_up = False
        teleop._prev_speed_down = False
        teleop._prev_fine_tune = False
        return teleop

    def test_get_action_returns_motor_positions(self):
        teleop = self._make_teleop_with_mock_controller()
        action = teleop.get_action()
        # All motors should have .pos keys
        for motor in teleop.mapping_engine.motors:
            assert f"{motor}.pos" in action

    def test_get_action_with_stick_input(self):
        teleop = self._make_teleop_with_mock_controller()
        teleop.init_targets({"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0,
                             "elbow_flex.pos": 0.0, "wrist_flex.pos": 0.0,
                             "wrist_roll.pos": 0.0, "gripper.pos": 50.0})
        teleop.controller.get_raw_left_stick.return_value = (0.5, 0.0)
        action = teleop.get_action()
        assert action["shoulder_pan.pos"] == pytest.approx(0.75)  # 0 + 0.5*1.5

    def test_init_targets_from_observation(self):
        teleop = self._make_teleop_with_mock_controller()
        obs = {
            "shoulder_pan.pos": 45.0,
            "shoulder_lift.pos": -30.0,
            "elbow_flex.pos": 10.0,
            "wrist_flex.pos": 5.0,
            "wrist_roll.pos": -20.0,
            "gripper.pos": 50.0,
        }
        teleop.init_targets(obs)
        assert teleop._current_targets["shoulder_pan"] == 45.0
        assert teleop._current_targets["shoulder_lift"] == -30.0

    def test_action_features_match_mapping(self):
        teleop = self._make_teleop_with_mock_controller()
        features = teleop.action_features
        names = features["names"]
        assert "shoulder_pan.pos" in names
        assert "gripper.pos" in names
        assert features["shape"] == (6,)  # 6 motors

    def test_meta_speed_up(self):
        teleop = self._make_teleop_with_mock_controller()
        teleop.controller.buttons["up"] = True
        teleop.get_action()
        assert teleop.controller.speed_multiplier == pytest.approx(1.2)

    def test_meta_fine_tune_toggle(self):
        teleop = self._make_teleop_with_mock_controller()
        assert teleop.controller.fine_tune is False
        teleop.controller.buttons["l_stick_press"] = True
        teleop.get_action()
        assert teleop.controller.fine_tune is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py::TestJoyConTeleopMappingIntegration -x -q`
Expected: FAIL — `get_action()` doesn't use MappingEngine yet

- [ ] **Step 3: Rewrite teleop_joycon.py to integrate MappingEngine**

Replace the entire content of `src/lerobot/teleoperators/joycon/teleop_joycon.py`:

```python
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

"""Nintendo Switch Joy-Con teleoperator with configurable motor mapping.

Produces motor-level position commands based on a YAML mapping configuration.
By default, maps Joy-Con inputs to SO-101 joints:

    Left stick X  → Shoulder Pan (motor 1)
    Left stick Y  → Shoulder Lift (motor 2)
    L / ZL        → Elbow Flex (motor 3)
    IMU tilt      → Wrist Flex (motor 4)
    IMU roll      → Wrist Roll (motor 5)
    A / B         → Gripper (motor 6)
"""

import logging
from typing import Any

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_joycon import JoyConTeleopConfig
from .mapping_engine import MappingEngine

logger = logging.getLogger(__name__)

# Default joint limits used when none are provided to connect()
_DEFAULT_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-97.0, 97.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.0, 163.0),
    "gripper": (0.0, 100.0),
}


class JoyConTeleop(Teleoperator):
    """Nintendo Switch Joy-Con teleoperator with configurable mapping.

    Uses a MappingEngine to convert Joy-Con inputs into motor position
    targets. The mapping is loaded from a YAML file or uses built-in defaults.

    Usage::

        from lerobot.teleoperators.joycon import JoyConTeleop, JoyConTeleopConfig

        config = JoyConTeleopConfig(mapping_path="my_mapping.yaml")
        teleop = JoyConTeleop(config)
        teleop.connect(joint_limits=robot.get_joint_limits())
        teleop.init_targets(robot.get_observation())
        action = teleop.get_action()  # {"shoulder_pan.pos": 45.2, ...}
    """

    config_class = JoyConTeleopConfig
    name = "joycon"

    def __init__(self, config: JoyConTeleopConfig):
        super().__init__(config)
        self.config = config
        self.controller = None  # JoyConHIDController, set on connect()
        self.mapping_engine: MappingEngine | None = None
        self._current_targets: dict[str, float] = {}
        self._prev_speed_up: bool = False
        self._prev_speed_down: bool = False
        self._prev_fine_tune: bool = False

    # ── Action / feedback descriptors ────────────────────────────────────────

    @property
    def action_features(self) -> dict:
        if self.mapping_engine:
            names = {
                f"{motor}.pos": i
                for i, motor in enumerate(self.mapping_engine.motors)
            }
            shape = len(names)
        else:
            names = {}
            shape = 0
        return {"dtype": "float32", "shape": (shape,), "names": names}

    @property
    def feedback_features(self) -> dict:
        return {}

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def connect(
        self,
        calibrate: bool = True,
        joint_limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Connect to Joy-Con device(s) and load mapping engine.

        Args:
            calibrate: Whether to calibrate IMU offsets.
            joint_limits: Motor joint limits as {name: (min_deg, max_deg)}.
                If None, uses built-in SO-101 defaults.
        """
        from .joycon_utils import JoyConHIDController

        self.controller = JoyConHIDController(
            mode=self.config.mode,
            deadzone=self.config.deadzone,
            x_step_size=self.config.step_size,
            y_step_size=self.config.step_size,
            z_step_size=self.config.z_step_size,
            rotation_step=self.config.rotation_step,
            min_speed=self.config.min_speed,
            max_speed=self.config.max_speed,
            speed_step=self.config.speed_step,
            fine_tune_multiplier=self.config.fine_tune_multiplier,
        )
        self.controller.start()

        if not self.controller.devices:
            self.controller = None
            raise ConnectionError(
                "Failed to connect to any Joy-Con. Make sure a Joy-Con is paired "
                "via Bluetooth and that you have permission to access HID devices."
            )

        if self.controller.stick_available and calibrate:
            logger.info("Calibrating IMU... hold Joy-Con in natural position.")
            self.controller.calibrate_imu()

        # Load mapping engine
        if joint_limits is None:
            joint_limits = _DEFAULT_JOINT_LIMITS

        if self.config.mapping_path:
            self.mapping_engine = MappingEngine.from_yaml(
                self.config.mapping_path, joint_limits
            )
        else:
            self.mapping_engine = MappingEngine.default(joint_limits)

        # Initialize state
        self._current_targets = {}
        self._prev_speed_up = False
        self._prev_speed_down = False
        self._prev_fine_tune = False

        logger.info(
            "Joy-Con teleop connected in %s mode (%d device(s)), %d motors mapped.",
            self.controller.mode.value,
            len(self.controller.devices),
            len(self.mapping_engine.motors),
        )

    @property
    def is_connected(self) -> bool:
        return self.controller is not None and bool(self.controller.devices)

    def disconnect(self) -> None:
        """Disconnect from Joy-Con device(s)."""
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
        self.mapping_engine = None

    # ── Calibration (no-op; Joy-Con self-centers on connect) ─────────────────

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ── Target initialization ────────────────────────────────────────────────

    def init_targets(self, observation: dict) -> None:
        """Initialize target positions from robot observation.

        Call after connect() and before the main control loop.

        Args:
            observation: Robot observation dict with keys like "shoulder_pan.pos".
        """
        if self.mapping_engine is None:
            raise RuntimeError("Call connect() before init_targets()")
        self._current_targets = {}
        for motor in self.mapping_engine.motors:
            key = f"{motor}.pos"
            self._current_targets[motor] = float(observation.get(key, 0.0))

    # ── The action loop ──────────────────────────────────────────────────────

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Read Joy-Con inputs and return motor position targets."""
        self.controller.update()

        input_state = self._build_input_state()
        self._handle_meta_controls(input_state)

        targets = self.mapping_engine.compute_targets(
            input_state,
            self._current_targets,
            speed_multiplier=self.controller.speed_multiplier,
            fine_tune=self.controller.fine_tune,
        )
        self._current_targets = targets

        return {f"{motor}.pos": pos for motor, pos in targets.items()}

    def _build_input_state(self) -> dict[str, float | bool]:
        """Extract all Joy-Con inputs into a flat dict for MappingEngine."""
        ctrl = self.controller
        lx, ly = ctrl.get_raw_left_stick()
        rx, ry = ctrl.get_raw_right_stick()
        imu_tilt, imu_roll = ctrl.get_wrist_angles()
        buttons = ctrl.buttons

        return {
            # Sticks (float [-1, 1])
            "left_stick_x": lx,
            "left_stick_y": ly,
            "right_stick_x": rx,
            "right_stick_y": ry,
            # Buttons (bool)
            "a": buttons.get("a", False),
            "b": buttons.get("b", False),
            "x": buttons.get("x", False),
            "y": buttons.get("y", False),
            "l": buttons.get("l", False),
            "zl": buttons.get("zl", False),
            "r": buttons.get("r", False),
            "zr": buttons.get("zr", False),
            "dpad_up": buttons.get("up", False),
            "dpad_down": buttons.get("down", False),
            "dpad_left": buttons.get("left", False),
            "dpad_right": buttons.get("right", False),
            "plus": buttons.get("plus", False),
            "minus": buttons.get("minus", False),
            "home": buttons.get("home", False),
            "capture": buttons.get("capture", False),
            "l_stick_press": buttons.get("l_stick_press", False),
            "r_stick_press": buttons.get("r_stick_press", False),
            # IMU (float, degrees)
            "imu_tilt": imu_tilt,
            "imu_roll": imu_roll,
        }

    def _handle_meta_controls(self, input_state: dict) -> None:
        """Process speed and fine-tune meta controls (edge detection)."""
        meta = self.mapping_engine.meta_controls

        speed_up = input_state.get(meta.speed_up, False)
        speed_down = input_state.get(meta.speed_down, False)
        fine_toggle = input_state.get(meta.fine_tune_toggle, False)

        if speed_up and not self._prev_speed_up:
            self.controller.speed_multiplier = min(
                self.controller.speed_multiplier + self.config.speed_step,
                self.config.max_speed,
            )
        if speed_down and not self._prev_speed_down:
            self.controller.speed_multiplier = max(
                self.controller.speed_multiplier - self.config.speed_step,
                self.config.min_speed,
            )
        if fine_toggle and not self._prev_fine_tune:
            self.controller.fine_tune = not self.controller.fine_tune

        self._prev_speed_up = speed_up
        self._prev_speed_down = speed_down
        self._prev_fine_tune = fine_toggle

    def get_teleop_events(self) -> dict[str, Any]:
        """Get episode control events from the Joy-Con."""
        if self.controller is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
                "emergency_stop": False,
                "speed_multiplier": 1.0,
                "fine_tune": False,
            }

        self.controller.update()

        is_intervention = self.controller.should_intervene()
        status = self.controller.get_episode_end_status()
        terminate = status in (TeleopEvents.RERECORD_EPISODE, TeleopEvents.FAILURE)
        success = status == TeleopEvents.SUCCESS
        rerecord = status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord,
            "emergency_stop": self.controller.emergency_stop,
            "speed_multiplier": self.controller.speed_multiplier,
            "fine_tune": self.controller.fine_tune,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Send feedback to the Joy-Con (not supported)."""
        pass
```

- [ ] **Step 4: Run integration tests**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py::TestJoyConTeleopMappingIntegration -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py -x -q`
Expected: All tests PASS (78 original + new mapping + integration tests)

- [ ] **Step 6: Commit**

```bash
git add src/lerobot/teleoperators/joycon/teleop_joycon.py tests/teleoperators/test_joycon.py
git commit -m "feat(joycon): integrate MappingEngine into JoyConTeleop.get_action()"
```

---

### Task 8: Create default_mapping.yaml

**Files:**
- Create: `examples/joycon_to_so101/default_mapping.yaml`

**Interfaces:**
- Reference YAML file for users to copy and customize

- [ ] **Step 1: Create the default mapping YAML**

Create `examples/joycon_to_so101/default_mapping.yaml`:

```yaml
# Joy-Con → SO-101 舵机映射配置 (默认)
#
# 输入类型:
#   摇杆:  left_stick_x, left_stick_y, right_stick_x, right_stick_y
#   按键:  a, b, x, y, l, zl, r, zr
#          dpad_up, dpad_down, dpad_left, dpad_right
#          plus, minus, home, capture
#          l_stick_press, r_stick_press
#   体感:  imu_tilt (前后倾斜), imu_roll (左右翻滚)
#
# 控制模式:
#   incremental — 每帧增减角度 (摇杆/按键)
#   absolute    — 直接设定目标角度 (IMU)

mappings:
  # Motor 1: Shoulder Pan (左右转)
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
    speed: 1.5
    invert: false

  # Motor 2: Shoulder Lift (上下抬)
  - input: left_stick_y
    motor: shoulder_lift
    control: incremental
    speed: 1.5
    invert: true

  # Motor 3: Elbow Flex (肘部)
  - input: l
    motor: elbow_flex
    control: incremental
    speed: 1.5
    invert: false

  - input: zl
    motor: elbow_flex
    control: incremental
    speed: 1.5
    invert: true

  # Motor 4: Wrist Flex (手腕俯仰)
  - input: imu_tilt
    motor: wrist_flex
    control: absolute
    scale: 0.5

  # Motor 5: Wrist Roll (手腕旋转)
  - input: imu_roll
    motor: wrist_roll
    control: absolute
    scale: 0.5

  # Motor 6: Gripper (夹爪)
  - input: a
    motor: gripper
    control: incremental
    speed: 3.0
    invert: false

  - input: b
    motor: gripper
    control: incremental
    speed: 3.0
    invert: true

# 功能按键 (可配置)
meta_controls:
  speed_up: dpad_up        # 速度 +20%
  speed_down: dpad_down    # 速度 -20%
  fine_tune_toggle: l_stick_press  # 微调模式切换

# 以下固定不可配置:
#   home    → 紧急停止
#   plus    → 成功标记
#   minus   → 失败标记
```

- [ ] **Step 2: Commit**

```bash
git add examples/joycon_to_so101/default_mapping.yaml
git commit -m "docs: add default Joy-Con mapping YAML reference file"
```

---

### Task 9: Rewrite example teleoperate.py script

**Files:**
- Rewrite: `examples/joycon_to_so101/teleoperate.py`

**Interfaces:**
- Consumes: `JoyConTeleop`, `JoyConTeleopConfig` (from Task 7)
- Consumes: `SO100Follower`, `SO100FollowerConfig`, `get_joint_limits()` (from Task 1)

- [ ] **Step 1: Rewrite teleoperate.py**

Replace `examples/joycon_to_so101/teleoperate.py` with:

```python
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

"""Teleoperate an SO-101 follower arm using a Nintendo Switch Joy-Con.

Uses a configurable YAML mapping to assign Joy-Con inputs to motors.

Prerequisites:
  1. Install dependencies:
     uv pip install 'lerobot[joycon,feetech]'

  2. Calibrate your SO-101 follower arm:
     uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem*

  3. Pair your Joy-Con(s) via Bluetooth (see docs/joycon_teleoperation.md)

Usage:
  uv run python examples/joycon_to_so101/teleoperate.py \\
      --port=/dev/tty.usbmodem5B7B0137181

  # Custom mapping file:
  uv run python examples/joycon_to_so101/teleoperate.py \\
      --port=/dev/tty.usbmodem5B7B0137181 \\
      --mapping=my_mapping.yaml
"""

import argparse
import signal
import time

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.joycon import JoyConMode, JoyConTeleop, JoyConTeleopConfig
from lerobot.utils.robot_utils import precise_sleep


def teleoperate(
    port: str,
    mode: str = "auto",
    mapping_path: str | None = None,
    fps: int = 30,
):
    """Run Joy-Con teleoperation with configurable mapping."""

    # ── Setup ────────────────────────────────────────────────────────────
    joycon_mode = JoyConMode(mode)
    teleop_config = JoyConTeleopConfig(
        mode=joycon_mode,
        mapping_path=mapping_path,
    )
    follower_config = SO100FollowerConfig(
        port=port,
        use_degrees=True,
        disable_torque_on_disconnect=True,
    )

    teleop = JoyConTeleop(teleop_config)
    follower = SO100Follower(follower_config)

    # ── Connect ──────────────────────────────────────────────────────────
    print("Connecting to Joy-Con...")
    teleop.connect(joint_limits=follower_config.joint_limits)
    print(f"Joy-Con connected: mode={teleop.controller.mode.value}")

    print("Connecting to SO-101 follower...")
    follower.connect(calibrate=False)
    print("SO-101 follower connected.")

    # Initialize target positions from current robot state
    teleop.init_targets(follower.get_observation())

    # ── Main loop ────────────────────────────────────────────────────────
    shutdown = False

    def handle_signal(sig, frame):
        nonlocal shutdown
        print("\nShutting down...")
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)

    print("\n" + "=" * 50)
    print("  SO-101 Joy-Con Teleop (configurable mapping)")
    print("=" * 50)
    print(f"  Mapping: {mapping_path or 'built-in default'}")
    print(f"  Motors:  {', '.join(teleop.mapping_engine.motors)}")
    print(f"  FPS:     {fps}")
    print()
    print("  Home       → Emergency stop")
    print("  Plus (+)   → SUCCESS")
    print("  Minus (-)  → FAILURE")
    print(f"  {teleop.mapping_engine.meta_controls.speed_up:12s} → Speed +20%")
    print(f"  {teleop.mapping_engine.meta_controls.speed_down:12s} → Speed -20%")
    print(f"  {teleop.mapping_engine.meta_controls.fine_tune_toggle:12s} → Fine-tune toggle")
    print("=" * 50 + "\n")

    loop_period = 1.0 / fps
    last_speed = 1.0
    last_fine_tune = False

    try:
        while not shutdown and teleop.is_connected:
            loop_start = time.perf_counter()

            # Check system events
            events = teleop.get_teleop_events()
            if events.get("emergency_stop"):
                print("\n🛑 EMERGENCY STOP")
                break

            # Print speed/fine-tune changes
            cur_speed = events.get("speed_multiplier", 1.0)
            cur_fine = events.get("fine_tune", False)
            if cur_speed != last_speed or cur_fine != last_fine_tune:
                ft_str = "ON" if cur_fine else "OFF"
                print(f"  Speed: {int(cur_speed * 100)}% | Fine-tune: {ft_str}")
                last_speed = cur_speed
                last_fine_tune = cur_fine

            # Get motor targets from mapping engine → forward to robot
            action = teleop.get_action()
            follower.send_action(action)

            # Frame rate control
            dt = time.perf_counter() - loop_start
            precise_sleep(max(loop_period - dt, 0))

    finally:
        print("\nDisconnecting...")
        teleop.disconnect()
        follower.disconnect()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Teleoperate SO-101 arm with Joy-Con (configurable mapping)"
    )
    parser.add_argument(
        "--port", required=True, help="Serial port for SO-101 (e.g., /dev/tty.usbmodem*)"
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "single_left", "single_right", "dual"],
        help="Joy-Con mode (default: auto)",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Path to mapping YAML file (default: built-in)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency")

    args = parser.parse_args()

    teleoperate(
        port=args.port,
        mode=args.mode,
        mapping_path=args.mapping,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses correctly**

Run: `uv run python examples/joycon_to_so101/teleoperate.py --help`
Expected: Prints help with `--port`, `--mode`, `--mapping`, `--fps` arguments

- [ ] **Step 3: Commit**

```bash
git add examples/joycon_to_so101/teleoperate.py
git commit -m "refactor(joycon): simplify teleoperate.py with configurable mapping"
```

---

### Task 10: Add pyyaml to joycon extra and final verification

**Files:**
- Modify: `pyproject.toml:168`

**Interfaces:**
- Ensures PyYAML is an explicit dependency of the `joycon` extra

- [ ] **Step 1: Add pyyaml to joycon extra in pyproject.toml**

In `pyproject.toml`, find the `joycon` line (around line 168) and update it:

```toml
joycon = ["hidapi>=0.14.0,<0.15.0", "pyyaml>=6.0"]
```

- [ ] **Step 2: Run full test suite**

Run: `uv run python -m pytest tests/teleoperators/test_joycon.py -v --tb=short`
Expected: ALL tests PASS (78 original + ~25 new = ~103 total)

- [ ] **Step 3: Verify imports work end-to-end**

Run: `uv run python -c "
from lerobot.teleoperators.joycon import JoyConTeleop, JoyConTeleopConfig, MappingEngine
from lerobot.robots.so_follower import SO100FollowerConfig
c = SO100FollowerConfig(port='/dev/null')
print('joint_limits:', list(c.joint_limits.keys()))
e = MappingEngine.default(c.joint_limits)
print('motors:', e.motors)
print('meta:', e.meta_controls)
"`
Expected: Prints joint_limits keys, motor list, and meta controls

- [ ] **Step 4: Run pre-commit checks**

Run: `uv run pre-commit run --files src/lerobot/teleoperators/joycon/mapping_engine.py src/lerobot/teleoperators/joycon/teleop_joycon.py src/lerobot/teleoperators/joycon/configuration_joycon.py src/lerobot/teleoperators/joycon/__init__.py examples/joycon_to_so101/teleoperate.py`
Expected: All checks PASS (ruff format, ruff check, etc.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyyaml to joycon optional dependencies"
```

---

## Self-Review Checklist

After completing all tasks, verify:

- [ ] All 78 original tests in `test_joycon.py` still pass
- [ ] New MappingEngine tests pass (~25 tests)
- [ ] `joycon_utils.py` unchanged except for `get_raw_left_stick()` and `get_raw_right_stick()`
- [ ] `examples/joycon_to_so101/teleoperate.py` works with `--help`
- [ ] `default_mapping.yaml` matches `_DEFAULT_MAPPINGS` in code
- [ ] No TBD/TODO/placeholders remain
- [ ] pre-commit passes on all modified files
