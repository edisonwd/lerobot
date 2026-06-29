# Joy-Con 陀螺仪姿态主控遥控 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add gyroscope-based primary control to the Joy-Con teleoperator for SO101 6-axis arm, with complementary filter fusion, dual mode switching, and expanded meta controls.

**Architecture:** Extends existing MappingEngine with new `gyro_pitch_delta`/`gyro_roll_delta` inputs. Complementary filter in `joycon_utils.py` fuses accelerometer + gyroscope data. Dual YAML mapping files enable SL-toggled mode switching. Per-axis gains configured via YAML `speed` field.

**Tech Stack:** Python 3.12+, PyTorch, hidapi, PyYAML, pytest

## Global Constraints

- All new code follows existing module patterns (no new files except YAML mappings)
- Backward compatible: existing tests must pass, existing YAML mappings must work
- IMU filter α: 0.02 (normal) / 0.005 (stabilized), toggleable via SR
- Speed: 3 discrete levels [0.5, 1.0, 1.5], replacing continuous ±20%
- Gyro deadzone: ±0.1° on delta values
- Presets: only listed motors change, unlisted motors keep current position, clamped to joint limits
- All meta controls edge-triggered (rising edge detection)
- Tests use mock controllers — no real hardware needed

---

## File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `src/lerobot/teleoperators/joycon/joycon_utils.py` | HID protocol, IMU parsing, stick/button decoding, speed control | Add gyro parsing, complementary filter, delta computation, 3-level speed |
| `src/lerobot/teleoperators/joycon/configuration_joycon.py` | Config dataclass | Add `alt_mapping_path`, `speed_levels` |
| `src/lerobot/teleoperators/joycon/mapping_engine.py` | Input→motor mapping, YAML loading | Add gyro inputs, expanded MetaControls, presets |
| `src/lerobot/teleoperators/joycon/teleop_joycon.py` | Teleoperator orchestration | Dual engine, mode switch, pose lock, presets, meta controls |
| `examples/joycon_to_so101/gyro_primary_mapping.yaml` | Gyro mode mapping | New file |
| `examples/joycon_to_so101/stick_only_mapping.yaml` | Stick mode mapping | New file |
| `examples/joycon_to_so101/teleoperate.py` | Example script | Add `--alt-mapping` arg |
| `tests/teleoperators/test_joycon.py` | Unit tests | Add tests for all new features |

---

### Task 1: IMU Gyroscope Parsing + Complementary Filter + 3-Level Speed

**Files:**
- Modify: `src/lerobot/teleoperators/joycon/joycon_utils.py`
- Test: `tests/teleoperators/test_joycon.py`

**Interfaces:**
- Produces new attributes on `JoyConHIDController`:
  - `imu_pitch: float` — fused pitch angle (degrees)
  - `imu_roll: float` — fused roll angle (degrees)
  - `gyro_pitch_delta: float` — pitch change per frame (degrees)
  - `gyro_roll_delta: float` — roll change per frame (degrees)
  - `imu_filter_stabilized: bool` — current filter mode
  - `speed_levels: list[float]` — discrete speed levels
  - `speed_level_index: int` — current level index
- Produces new methods:
  - `toggle_filter()` — swap between normal/stabilized α
  - `recalibrate_im()` — re-zero gyro integrated angles
  - `get_gyro_deltas() -> tuple[float, float]` — return (pitch_delta, roll_delta)

- [ ] **Step 1: Write failing tests for gyro parsing and complementary filter**

Add to `tests/teleoperators/test_joycon.py` in the `TestIMU` class:

```python
    def test_gyro_bytes_parsed(self):
        """Gyroscope bytes 19-24 are parsed into raw gyro values."""
        report = _make_full_report()
        # gyro_x = 1000 (int16 LE), gyro_y = -500, gyro_z = 200
        report[19] = 0xE8  # 1000 lo
        report[20] = 0x03  # 1000 hi
        report[21] = 0x0C  # -500 lo (0xFE0C)
        report[22] = 0xFE  # -500 hi
        report[23] = 0xC8  # 200 lo
        report[24] = 0x00  # 200 hi
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        assert ctrl.imu_gyro_x == 1000
        assert ctrl.imu_gyro_y == -500
        assert ctrl.imu_gyro_z == 200

    def test_complementary_filter_initializes_from_accel(self):
        """First frame: fused angle ≈ accel angle (gyro integration starts from 0)."""
        report = _make_full_report()
        # acc_x = 4096, acc_z = 0 → accel pitch = 90°
        report[13] = 0x00; report[14] = 0x10  # acc_x = 4096
        report[15] = 0x00; report[16] = 0x00  # acc_y = 0
        report[17] = 0x00; report[18] = 0x00  # acc_z = 0
        # gyro all zero
        report[19] = 0x00; report[20] = 0x00
        report[21] = 0x00; report[22] = 0x00
        report[23] = 0x00; report[24] = 0x00
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        # With α=0.02: fused = 0.02*90 + 0.98*0 = 1.8°
        assert ctrl.imu_pitch == pytest.approx(1.8, abs=0.5)

    def test_gyro_delta_zero_when_stationary(self):
        """Two identical frames → delta ≈ 0 (only accel correction accumulates slowly)."""
        report = _make_full_report()
        report[13] = 0x00; report[14] = 0x00  # acc_x = 0
        report[15] = 0x00; report[16] = 0x00  # acc_y = 0
        report[17] = 0x00; report[18] = 0x10  # acc_z = 4096 (upright)
        for i in range(19, 25):
            report[i] = 0x00  # gyro = 0
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report, report],
        )
        # After two identical upright frames, deltas should be near zero
        assert abs(ctrl.gyro_pitch_delta) < 0.5
        assert abs(ctrl.gyro_roll_delta) < 0.5

    def test_gyro_delta_deadzone(self):
        """Deltas below ±0.1° threshold are zeroed."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        # Manually set a tiny delta
        ctrl._prev_imu_pitch = 0.0
        ctrl.imu_pitch = 0.05  # 0.05° change, below 0.1° deadzone
        ctrl._prev_imu_roll = 0.0
        ctrl.imu_roll = 0.05
        # Trigger delta computation by calling _update_gyro_deltas
        ctrl._update_gyro_deltas()
        assert ctrl.gyro_pitch_delta == 0.0
        assert ctrl.gyro_roll_delta == 0.0

    def test_gyro_delta_above_deadzone(self):
        """Deltas above threshold pass through."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl._prev_imu_pitch = 0.0
        ctrl.imu_pitch = 2.0
        ctrl._prev_imu_roll = 0.0
        ctrl.imu_roll = -1.5
        ctrl._update_gyro_deltas()
        assert ctrl.gyro_pitch_delta == pytest.approx(2.0)
        assert ctrl.gyro_roll_delta == pytest.approx(-1.5)

    def test_filter_toggle_changes_alpha(self):
        """toggle_filter() swaps between normal (0.02) and stabilized (0.005)."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        assert ctrl.imu_filter_stabilized is False
        assert ctrl._imu_filter_alpha == 0.02
        ctrl.toggle_filter()
        assert ctrl.imu_filter_stabilized is True
        assert ctrl._imu_filter_alpha == 0.005
        ctrl.toggle_filter()
        assert ctrl.imu_filter_stabilized is False
        assert ctrl._imu_filter_alpha == 0.02

    def test_recalibrate_imu_resets_gyro_angles(self):
        """recalibrate_imu() zeros the gyro-integrated pitch/roll."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl._gyro_pitch_angle = 45.0
        ctrl._gyro_roll_angle = -30.0
        ctrl.recalibrate_imu()
        assert ctrl._gyro_pitch_angle == 0.0
        assert ctrl._gyro_roll_angle == 0.0

    def test_get_gyro_deltas_returns_tuple(self):
        """get_gyro_deltas returns (pitch_delta, roll_delta)."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl.gyro_pitch_delta = 1.5
        ctrl.gyro_roll_delta = -0.8
        pd, rd = ctrl.get_gyro_deltas()
        assert pd == 1.5
        assert rd == -0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/teleoperators/test_joycon.py::TestIMU::test_gyro_bytes_parsed -xvs`
Expected: FAIL — `AttributeError: 'JoyConHIDController' object has no attribute 'imu_gyro_x'`

- [ ] **Step 3: Add gyro state variables to `__init__`**

In `joycon_utils.py`, add after the existing IMU state block (after line 233, the `self._imu_alpha` line):

```python
        # IMU gyroscope raw values
        self.imu_gyro_x: float = 0.0  # angular velocity X (raw LSB)
        self.imu_gyro_y: float = 0.0  # angular velocity Y (raw LSB)
        self.imu_gyro_z: float = 0.0  # angular velocity Z (raw LSB)

        # Complementary filter state
        self._gyro_pitch_angle: float = 0.0  # gyro-integrated pitch (degrees)
        self._gyro_roll_angle: float = 0.0   # gyro-integrated roll (degrees)
        self.imu_pitch: float = 0.0          # fused pitch angle (degrees)
        self.imu_roll: float = 0.0           # fused roll angle (degrees)

        # Delta computation
        self.gyro_pitch_delta: float = 0.0
        self.gyro_roll_delta: float = 0.0
        self._prev_imu_pitch: float = 0.0
        self._prev_imu_roll: float = 0.0
        self._imu_last_time: float = 0.0

        # Filter mode
        self.imu_filter_stabilized: bool = False
        self._imu_filter_alpha: float = 0.02   # normal mode
        self._imu_alpha_stabilized: float = 0.005
        self._imu_alpha_normal: float = 0.02

        # Gyro deadzone (degrees)
        self._gyro_delta_deadzone: float = 0.1
```

- [ ] **Step 4: Rewrite `_parse_imu` to include complementary filter**

Replace the existing `_parse_imu` method (lines 954-981) with:

```python
    def _parse_imu(self, data) -> None:
        """Parse accelerometer + gyroscope data and run complementary filter.

        Accelerometer: bytes 13-18 (int16 LE, ~1/4096 g per LSB).
        Gyroscope: bytes 19-24 (int16 LE, ~0.0027°/s per LSB).

        Complementary filter:
            fused = α * accel_angle + (1-α) * gyro_integrated_angle
        """
        # ── Accelerometer (bytes 13-18) ──────────────────────────────
        raw_x = self._int16_le(data[13], data[14])
        raw_y = self._int16_le(data[15], data[16])
        raw_z = self._int16_le(data[17], data[18])

        # EMA smoothing (keep existing behavior)
        a = self._imu_alpha
        self.imu_acc_x = a * raw_x + (1 - a) * self.imu_acc_x
        self.imu_acc_y = a * raw_y + (1 - a) * self.imu_acc_y
        self.imu_acc_z = a * raw_z + (1 - a) * self.imu_acc_z

        # Accelerometer-based tilt angles
        accel_pitch = math.degrees(math.atan2(self.imu_acc_x, self.imu_acc_z))
        accel_roll = math.degrees(math.atan2(self.imu_acc_y, self.imu_acc_z))

        # ── Gyroscope (bytes 19-24) ─────────────────────────────────
        self.imu_gyro_x = float(self._int16_le(data[19], data[20]))
        self.imu_gyro_y = float(self._int16_le(data[21], data[22]))
        self.imu_gyro_z = float(self._int16_le(data[23], data[24]))

        # Compute dt from wall clock
        now = time.monotonic()
        if self._imu_last_time > 0:
            dt = now - self._imu_last_time
        else:
            dt = 1.0 / 60.0  # assume 60Hz on first frame
        self._imu_last_time = now

        # Integrate gyroscope angular velocity → angle (degrees)
        # gyro_y → pitch, gyro_x → roll (matching Joy-Con axis convention)
        gyro_dps_scale = 0.0027  # °/s per LSB at ±2000°/s range
        self._gyro_pitch_angle += self.imu_gyro_y * gyro_dps_scale * dt
        self._gyro_roll_angle += self.imu_gyro_x * gyro_dps_scale * dt

        # ── Complementary filter ─────────────────────────────────────
        alpha = self._imu_filter_alpha
        self.imu_pitch = alpha * accel_pitch + (1 - alpha) * self._gyro_pitch_angle
        self.imu_roll = alpha * accel_roll + (1 - alpha) * self._gyro_roll_angle

        # Apply calibration offsets and clamp
        self.imu_pitch = max(-90.0, min(90.0, self.imu_pitch - self._imu_flex_offset))
        self.imu_roll = max(-90.0, min(90.0, self.imu_roll - self._imu_roll_offset))

        # Keep backward-compatible wrist angle aliases
        self.wrist_flex_angle = self.imu_pitch
        self.wrist_roll_angle = self.imu_roll

        # ── Delta computation ────────────────────────────────────────
        self._update_gyro_deltas()
```

- [ ] **Step 5: Add `_update_gyro_deltas`, `toggle_filter`, `recalibrate_imu`, `get_gyro_deltas` methods**

Add these methods right after `_parse_imu`:

```python
    def _update_gyro_deltas(self) -> None:
        """Compute per-frame angle deltas with deadzone."""
        raw_pitch_delta = self.imu_pitch - self._prev_imu_pitch
        raw_roll_delta = self.imu_roll - self._prev_imu_roll

        # Apply deadzone
        if abs(raw_pitch_delta) < self._gyro_delta_deadzone:
            raw_pitch_delta = 0.0
        if abs(raw_roll_delta) < self._gyro_delta_deadzone:
            raw_roll_delta = 0.0

        self.gyro_pitch_delta = raw_pitch_delta
        self.gyro_roll_delta = raw_roll_delta
        self._prev_imu_pitch = self.imu_pitch
        self._prev_imu_roll = self.imu_roll

    def toggle_filter(self) -> None:
        """Toggle between normal (α=0.02) and stabilized (α=0.005) filter."""
        self.imu_filter_stabilized = not self.imu_filter_stabilized
        if self.imu_filter_stabilized:
            self._imu_filter_alpha = self._imu_alpha_stabilized
        else:
            self._imu_filter_alpha = self._imu_alpha_normal
        mode = "stabilized" if self.imu_filter_stabilized else "normal"
        logger.info("IMU filter: %s (α=%.3f)", mode, self._imu_filter_alpha)

    def recalibrate_imu(self) -> None:
        """Re-zero gyro integrated angles and delta state.

        Call at runtime (e.g., D-pad ←) to eliminate accumulated drift.
        Unlike calibrate_imu(), this does NOT sample hardware — it just
        resets the internal state.
        """
        self._gyro_pitch_angle = 0.0
        self._gyro_roll_angle = 0.0
        self.imu_pitch = 0.0
        self.imu_roll = 0.0
        self._prev_imu_pitch = 0.0
        self._prev_imu_roll = 0.0
        self.gyro_pitch_delta = 0.0
        self.gyro_roll_delta = 0.0
        logger.info("IMU recalibrated: gyro angles zeroed.")

    def get_gyro_deltas(self) -> tuple[float, float]:
        """Get per-frame gyro angle deltas.

        Returns:
            (pitch_delta, roll_delta) in degrees.
        """
        return self.gyro_pitch_delta, self.gyro_roll_delta
```

- [ ] **Step 6: Update `calibrate_imu` to also reset gyro state**

In the existing `calibrate_imu` method (line 990), add gyro reset at the end, after the offset assignment (after line 1028):

```python
        # Also reset gyro integration state
        self._gyro_pitch_angle = 0.0
        self._gyro_roll_angle = 0.0
        self._prev_imu_pitch = 0.0
        self._prev_imu_roll = 0.0
```

- [ ] **Step 7: Replace speed system with 3 discrete levels**

In `__init__`, replace the speed-related lines (lines 175-181) with:

```python
        # Speed: 3 discrete levels
        self.speed_levels: list[float] = [0.5, 1.0, 1.5]  # fine, normal, fast
        self.speed_level_index: int = 1  # start at normal
        self.speed_multiplier: float = 1.0  # derived from speed_levels[index]
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_step = speed_step  # kept for backward compat but unused in new system
        self.fine_tune: bool = False
        self.fine_tune_multiplier = fine_tune_multiplier
        self.rotation_step = rotation_step
```

Replace `_handle_speed_buttons` (lines 897-917) with:

```python
    def _handle_speed_buttons(self, btns: dict[str, bool]) -> None:
        """Cycle through 3 speed levels on D-pad up (edge-triggered).

        D-pad down is no longer handled here — it's a meta control
        (reset_to_center) managed by JoyConTeleop.
        """
        dpad_up = btns.get("up", False)

        if dpad_up and not self._prev_dpad_up:
            self.speed_level_index = (self.speed_level_index + 1) % len(self.speed_levels)
            self.speed_multiplier = self.speed_levels[self.speed_level_index]
            logger.info("Speed level: %d (%.1fx)", self.speed_level_index, self.speed_multiplier)

        self._prev_dpad_up = dpad_up
```

Remove the `_prev_dpad_down` edge detection from `_handle_speed_buttons` (it's now handled in `JoyConTeleop._handle_meta_controls`).

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/teleoperators/test_joycon.py::TestIMU -xvs`
Expected: All `TestIMU` tests PASS (including existing ones — backward compat maintained via `wrist_flex_angle`/`wrist_roll_angle` aliases)

Run: `uv run pytest tests/teleoperators/test_joycon.py::TestSpeedControl -xvs`
Expected: `test_speed_increase` needs updating — speed is now level-based, not continuous. The existing test asserts `speed_multiplier == 1.2` but now it should be `1.5` (level 2). Update the test:

```python
    def test_speed_increase(self):
        """D-pad up → cycles to next speed level."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        assert ctrl.speed_multiplier == 1.0  # level 1 (normal)
        ctrl.buttons = {}
        ctrl._map_buttons_to_actions()  # initial
        ctrl.buttons = {"up": True}
        ctrl._map_buttons_to_actions()
        assert ctrl.speed_multiplier == 1.5  # level 2 (fast)
```

Update `test_speed_decrease` — D-pad down no longer decreases speed:

```python
    def test_speed_decrease_removed(self):
        """D-pad down no longer changes speed (now handled as reset_to_center meta control)."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl.buttons = {}
        ctrl._map_buttons_to_actions()  # initial
        ctrl.buttons = {"down": True}
        ctrl._map_buttons_to_actions()
        assert ctrl.speed_multiplier == 1.0  # unchanged
```

Update `test_speed_max_bound` and `test_speed_min_bound` — replace with level cycling test:

```python
    def test_speed_cycles_through_levels(self):
        """D-pad up cycles: normal(1.0) → fast(1.5) → fine(0.5) → normal(1.0)."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        assert ctrl.speed_level_index == 1  # normal
        # Press up → fast
        ctrl.buttons = {}
        ctrl._map_buttons_to_actions()
        ctrl.buttons = {"up": True}
        ctrl._map_buttons_to_actions()
        assert ctrl.speed_level_index == 2
        assert ctrl.speed_multiplier == 1.5
        # Press up → fine (wraps)
        ctrl.buttons = {}
        ctrl._map_buttons_to_actions()
        ctrl.buttons = {"up": True}
        ctrl._map_buttons_to_actions()
        assert ctrl.speed_level_index == 0
        assert ctrl.speed_multiplier == 0.5
        # Press up → normal (wraps back)
        ctrl.buttons = {}
        ctrl._map_buttons_to_actions()
        ctrl.buttons = {"up": True}
        ctrl._map_buttons_to_actions()
        assert ctrl.speed_level_index == 1
        assert ctrl.speed_multiplier == 1.0
```

- [ ] **Step 9: Run full test suite to check no regressions**

Run: `uv run pytest tests/teleoperators/test_joycon.py -xvs`
Expected: All tests PASS (updated speed tests + new IMU tests + all existing tests)

- [ ] **Step 10: Commit**

```bash
git add src/lerobot/teleoperators/joycon/joycon_utils.py tests/teleoperators/test_joycon.py
git commit -m "feat(joycon): add gyroscope parsing, complementary filter, and 3-level speed"
```

---

### Task 2: Config — Add `alt_mapping_path` and `speed_levels`

**Files:**
- Modify: `src/lerobot/teleoperators/joycon/configuration_joycon.py:54-66`
- Test: `tests/teleoperators/test_joycon.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `JoyConTeleopConfig.alt_mapping_path: str | None`, `JoyConTeleopConfig.speed_levels: list[float]`

- [ ] **Step 1: Write failing test**

Add to `tests/teleoperators/test_joycon.py` in the `TestJoyConConfig` class:

```python
    def test_config_alt_mapping_path(self):
        config = JoyConTeleopConfig(
            mapping_path="gyro.yaml",
            alt_mapping_path="stick.yaml",
        )
        assert config.mapping_path == "gyro.yaml"
        assert config.alt_mapping_path == "stick.yaml"

    def test_config_alt_mapping_path_default_none(self):
        config = JoyConTeleopConfig()
        assert config.alt_mapping_path is None

    def test_config_speed_levels(self):
        config = JoyConTeleopConfig(speed_levels=[0.3, 1.0, 2.0])
        assert config.speed_levels == [0.3, 1.0, 2.0]

    def test_config_speed_levels_default(self):
        config = JoyConTeleopConfig()
        assert config.speed_levels == [0.5, 1.0, 1.5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/teleoperators/test_joycon.py::TestJoyConConfig::test_config_alt_mapping_path -xvs`
Expected: FAIL — `TypeError: unexpected keyword argument 'alt_mapping_path'`

- [ ] **Step 3: Add new fields to `JoyConTeleopConfig`**

In `configuration_joycon.py`, add two fields after `mapping_path` (line 66):

```python
    alt_mapping_path: str | None = None
    speed_levels: list[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
```

Add `field` to the imports at the top of the file (line 17):

```python
from dataclasses import dataclass, field
```

Update the docstring to document the new fields:

```python
        alt_mapping_path: Path to alternate YAML mapping (stick-only mode).
            When set, SL button toggles between mapping_path and alt_mapping_path.
        speed_levels: Discrete speed multiplier levels cycled by D-pad up.
            Default: [0.5, 1.0, 1.5] (fine, normal, fast).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/teleoperators/test_joycon.py::TestJoyConConfig -xvs`
Expected: All `TestJoyConConfig` tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/lerobot/teleoperators/joycon/configuration_joycon.py tests/teleoperators/test_joycon.py
git commit -m "feat(joycon): add alt_mapping_path and speed_levels config fields"
```

---

### Task 3: MappingEngine — Add Gyro Inputs, Expanded MetaControls, Presets

**Files:**
- Modify: `src/lerobot/teleoperators/joycon/mapping_engine.py`
- Test: `tests/teleoperators/test_joycon.py`

**Interfaces:**
- Consumes: new VALID_INPUTS `"gyro_pitch_delta"`, `"gyro_roll_delta"`, `"imu_pitch"`, `"imu_roll"`
- Produces: expanded `MetaControls` with `reset_to_center`, `recalibrate_imu`, `pose_lock`, `mode_switch`, `filter_toggle`
- Produces: `MappingEngine.presets: dict[str, dict[str, float]]` — loaded from YAML `presets:` section

- [ ] **Step 1: Write failing tests**

Add to `tests/teleoperators/test_joycon.py`:

```python
class TestGyroInputs:
    """Test gyro inputs flow through MappingEngine."""

    def test_gyro_pitch_delta_in_valid_inputs(self):
        assert "gyro_pitch_delta" in VALID_INPUTS
        assert "gyro_roll_delta" in VALID_INPUTS
        assert "imu_pitch" in VALID_INPUTS
        assert "imu_roll" in VALID_INPUTS
        assert "sl_right" in VALID_INPUTS
        assert "sl_left" in VALID_INPUTS
        assert "sr_right" in VALID_INPUTS
        assert "sr_left" in VALID_INPUTS

    def test_gyro_delta_incremental(self):
        """gyro_pitch_delta flows through incremental mode with per-axis gain."""
        engine = MappingEngine(
            [MappingEntry("gyro_pitch_delta", "shoulder_lift", "incremental", speed=0.55)],
            MetaControls(),
            SO101_JOINT_LIMITS,
        )
        result = engine.compute_targets(
            {"gyro_pitch_delta": 2.0},
            {"shoulder_lift": 10.0},
        )
        assert result["shoulder_lift"] == pytest.approx(11.1)  # 10 + 2.0*0.55

    def test_gyro_delta_with_invert(self):
        """Inverted gyro delta (forward tilt → arm down)."""
        engine = MappingEngine(
            [MappingEntry("gyro_pitch_delta", "shoulder_lift", "incremental", speed=0.55, invert=True)],
            MetaControls(),
            SO101_JOINT_LIMITS,
        )
        result = engine.compute_targets(
            {"gyro_pitch_delta": 2.0},
            {"shoulder_lift": 10.0},
        )
        assert result["shoulder_lift"] == pytest.approx(8.9)  # 10 - 2.0*0.55

    def test_gyro_delta_zero_no_movement(self):
        """Zero gyro delta → no change."""
        engine = MappingEngine(
            [MappingEntry("gyro_pitch_delta", "shoulder_lift", "incremental", speed=0.55)],
            MetaControls(),
            SO101_JOINT_LIMITS,
        )
        result = engine.compute_targets(
            {"gyro_pitch_delta": 0.0},
            {"shoulder_lift": 10.0},
        )
        assert result["shoulder_lift"] == pytest.approx(10.0)

    def test_gyro_and_stick_same_motor(self):
        """Both gyro delta and stick map to same motor — deltas accumulate."""
        engine = MappingEngine(
            [
                MappingEntry("gyro_pitch_delta", "shoulder_lift", "incremental", speed=0.55),
                MappingEntry("left_stick_y", "shoulder_lift", "incremental", speed=0.8),
            ],
            MetaControls(),
            SO101_JOINT_LIMITS,
        )
        result = engine.compute_targets(
            {"gyro_pitch_delta": 1.0, "left_stick_y": 0.5},
            {"shoulder_lift": 0.0},
        )
        # 0 + 1.0*0.55 + 0.5*0.8 = 0.95
        assert result["shoulder_lift"] == pytest.approx(0.95)


class TestExpandedMetaControls:
    """Test expanded MetaControls fields."""

    def test_new_defaults(self):
        m = MetaControls()
        assert m.reset_to_center == "dpad_down"
        assert m.recalibrate_imu == "dpad_left"
        assert m.pose_lock == "dpad_right"
        assert m.mode_switch == "sl_right"
        assert m.filter_toggle == "sr_right"

    def test_speed_down_removed(self):
        """MetaControls no longer has speed_down field."""
        m = MetaControls()
        assert not hasattr(m, "speed_down")

    def test_custom_mode_switch(self):
        m = MetaControls(mode_switch="sl_left")
        assert m.mode_switch == "sl_left"

    def test_invalid_meta_input_raises(self):
        with pytest.raises(ValueError, match="unknown input"):
            MetaControls(mode_switch="fake_button")


class TestPresets:
    """Test preset loading and access."""

    def test_presets_loaded_from_yaml(self, tmp_path):
        yaml_content = """\
mappings:
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
presets:
  pickup:
    shoulder_lift: -30.0
    elbow_flex: 45.0
  place:
    shoulder_lift: 30.0
"""
        yaml_file = tmp_path / "test_presets.yaml"
        yaml_file.write_text(yaml_content)
        engine = MappingEngine.from_yaml(yaml_file, SO101_JOINT_LIMITS)
        assert "pickup" in engine.presets
        assert engine.presets["pickup"]["shoulder_lift"] == -30.0
        assert engine.presets["pickup"]["elbow_flex"] == 45.0
        assert "place" in engine.presets

    def test_presets_empty_when_not_in_yaml(self, tmp_path):
        yaml_content = """\
mappings:
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
"""
        yaml_file = tmp_path / "test_no_presets.yaml"
        yaml_file.write_text(yaml_content)
        engine = MappingEngine.from_yaml(yaml_file, SO101_JOINT_LIMITS)
        assert engine.presets == {}

    def test_default_mapping_has_no_presets(self):
        engine = MappingEngine.default(SO101_JOINT_LIMITS)
        assert engine.presets == {}

    def test_apply_preset(self):
        """apply_preset updates only specified motors, clamps to limits."""
        engine = MappingEngine(
            [MappingEntry("left_stick_x", "shoulder_pan", "incremental")],
            MetaControls(),
            SO101_JOINT_LIMITS,
        )
        engine.presets = {
            "pickup": {"shoulder_lift": -30.0, "elbow_flex": 45.0, "wrist_flex": -20.0},
        }
        current = {
            "shoulder_pan": 10.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        }
        result = engine.apply_preset("pickup", current)
        assert result["shoulder_lift"] == -30.0
        assert result["elbow_flex"] == 45.0
        assert result["wrist_flex"] == -20.0
        # Unchanged motors keep their values
        assert result["shoulder_pan"] == 10.0
        assert result["wrist_roll"] == 0.0
        assert result["gripper"] == 50.0

    def test_apply_preset_clamps_to_limits(self):
        """Preset values are clamped to joint limits."""
        engine = MappingEngine(
            [MappingEntry("left_stick_x", "shoulder_pan", "incremental")],
            MetaControls(),
            SO101_JOINT_LIMITS,
        )
        engine.presets = {"test": {"shoulder_lift": 200.0}}  # exceeds 100° limit
        result = engine.apply_preset("test", {"shoulder_lift": 0.0})
        assert result["shoulder_lift"] == 100.0  # clamped

    def test_apply_preset_unknown_name_returns_unchanged(self):
        """Unknown preset name returns targets unchanged."""
        engine = MappingEngine(
            [MappingEntry("left_stick_x", "shoulder_pan", "incremental")],
            MetaControls(),
            SO101_JOINT_LIMITS,
        )
        current = {"shoulder_pan": 10.0}
        result = engine.apply_preset("nonexistent", current)
        assert result == current
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/teleoperators/test_joycon.py::TestGyroInputs -xvs`
Expected: FAIL — `ValueError: Unknown input: 'gyro_pitch_delta'`

- [ ] **Step 3: Add gyro inputs to VALID_INPUTS**

In `mapping_engine.py`, add to the `VALID_INPUTS` set (after line 63):

```python
    "imu_pitch",
    "imu_roll",
    "gyro_pitch_delta",
    "gyro_roll_delta",
    # Side-specific SL/SR buttons (for mode switch and filter toggle)
    "sl_right",
    "sl_left",
    "sr_right",
    "sr_left",
```

- [ ] **Step 4: Expand MetaControls dataclass**

Replace the existing `MetaControls` class (lines 91-102) with:

```python
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
```

- [ ] **Step 5: Add `presets` to MappingEngine and `apply_preset` method**

Add `presets` parameter to `__init__` (line 125) and the `apply_preset` method:

```python
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
```

Add `apply_preset` method after `compute_targets`:

```python
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
```

- [ ] **Step 6: Update `from_yaml` to load presets and new meta controls**

Update `from_yaml` (lines 198-227) to load the new fields:

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
```

- [ ] **Step 7: Update `default()` to pass empty presets**

Update the `default()` classmethod (line 230):

```python
    @classmethod
    def default(cls, joint_limits: dict) -> MappingEngine:
        """Create engine with built-in default mapping."""
        return cls(list(_DEFAULT_MAPPINGS), MetaControls(), joint_limits, presets={})
```

- [ ] **Step 8: Fix existing test that references `speed_down`**

In `tests/teleoperators/test_joycon.py`, update `TestMetaControls::test_defaults`:

```python
    def test_defaults(self):
        m = MetaControls()
        assert m.speed_up == "dpad_up"
        assert m.fine_tune_toggle == "l_stick_press"
        assert m.reset_to_center == "dpad_down"
        assert m.mode_switch == "sl_right"
```

Update `TestMetaControls::test_custom_buttons`:

```python
    def test_custom_buttons(self):
        m = MetaControls(speed_up="r", mode_switch="sl_left", fine_tune_toggle="r_stick_press")
        assert m.speed_up == "r"
        assert m.mode_switch == "sl_left"
```

Update `TestMappingEngineFromYaml::test_load_yaml_without_meta_uses_defaults`:

```python
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
        assert engine.meta_controls.reset_to_center == "dpad_down"
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/teleoperators/test_joycon.py -k "TestGyroInputs or TestExpandedMetaControls or TestPresets or TestMetaControls or TestMappingEngineFromYaml" -xvs`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add src/lerobot/teleoperators/joycon/mapping_engine.py tests/teleoperators/test_joycon.py
git commit -m "feat(joycon): add gyro inputs, expanded MetaControls, and presets to MappingEngine"
```

---

### Task 4: Create YAML Mapping Files

**Files:**
- Create: `examples/joycon_to_so101/gyro_primary_mapping.yaml`
- Create: `examples/joycon_to_so101/stick_only_mapping.yaml`

- [ ] **Step 1: Create gyro primary mapping YAML**

Write `examples/joycon_to_so101/gyro_primary_mapping.yaml`:

```yaml
# SO101 Joy-Con 陀螺仪姿态主控映射
# 单只 Joy-Con（左或右），陀螺仪控制大臂和肘部，摇杆辅助微调

mappings:
  # 轴 1: 底座旋转 — 左摇杆 X
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
    speed: 1.5

  # 轴 2: 大臂升降 — 陀螺仪俯仰增量（主控）
  - input: gyro_pitch_delta
    motor: shoulder_lift
    control: incremental
    speed: 0.55
    invert: true              # 前倾 → 大臂下压

  # 轴 2 微调 — 左摇杆 Y
  - input: left_stick_y
    motor: shoulder_lift
    control: incremental
    speed: 0.8

  # 轴 3: 肘部屈伸 — 陀螺仪横滚增量（主控）
  - input: gyro_roll_delta
    motor: elbow_flex
    control: incremental
    speed: 0.95

  # 轴 4: 腕部俯仰 — 右摇杆 Y
  - input: right_stick_y
    motor: wrist_flex
    control: incremental
    speed: 1.15

  # 轴 5: 腕部自转 — ZL(逆时针) / ZR(顺时针)
  - input: zl
    motor: wrist_roll
    control: incremental
    speed: 1.15
    invert: true

  - input: zr
    motor: wrist_roll
    control: incremental
    speed: 1.15

  # 轴 6: 夹爪 — L(闭合) / R(张开)
  - input: l
    motor: gripper
    control: incremental
    speed: 3.0

  - input: r
    motor: gripper
    control: incremental
    speed: 3.0
    invert: true

meta_controls:
  speed_up: dpad_up
  reset_to_center: dpad_down
  recalibrate_imu: dpad_left
  pose_lock: dpad_right
  mode_switch: sl_right
  filter_toggle: sr_right
  fine_tune_toggle: l_stick_press

presets:
  pickup:
    shoulder_lift: -30.0
    elbow_flex: 45.0
    wrist_flex: -20.0
  place:
    shoulder_lift: 30.0
    elbow_flex: -20.0
    wrist_flex: 10.0
```

- [ ] **Step 2: Create stick-only mapping YAML**

Write `examples/joycon_to_so101/stick_only_mapping.yaml`:

```yaml
# SO101 Joy-Con 传统摇杆映射
# 纯摇杆控制，无陀螺仪，适合调试和机械限位检查

mappings:
  # 轴 1: 底座 — 左摇杆 X
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
    speed: 1.5

  # 轴 2: 大臂 — 左摇杆 Y
  - input: left_stick_y
    motor: shoulder_lift
    control: incremental
    speed: 1.5
    invert: true

  # 轴 3: 肘部 — 右摇杆 Y
  - input: right_stick_y
    motor: elbow_flex
    control: incremental
    speed: 1.5

  # 轴 4: 腕俯仰 — 右摇杆 X
  - input: right_stick_x
    motor: wrist_flex
    control: incremental
    speed: 1.15

  # 轴 5: 腕自转 — ZL / ZR
  - input: zl
    motor: wrist_roll
    control: incremental
    speed: 1.15
    invert: true

  - input: zr
    motor: wrist_roll
    control: incremental
    speed: 1.15

  # 轴 6: 夹爪 — L / R
  - input: l
    motor: gripper
    control: incremental
    speed: 3.0

  - input: r
    motor: gripper
    control: incremental
    speed: 3.0
    invert: true

meta_controls:
  speed_up: dpad_up
  reset_to_center: dpad_down
  recalibrate_imu: dpad_left
  pose_lock: dpad_right
  mode_switch: sl_right
  filter_toggle: sr_right
  fine_tune_toggle: l_stick_press

presets:
  pickup:
    shoulder_lift: -30.0
    elbow_flex: 45.0
    wrist_flex: -20.0
  place:
    shoulder_lift: 30.0
    elbow_flex: -20.0
    wrist_flex: 10.0
```

- [ ] **Step 3: Verify YAML files load correctly**

Run:
```bash
uv run python -c "
from lerobot.teleoperators.joycon.mapping_engine import MappingEngine
limits = {
    'shoulder_pan': (-110.0, 110.0),
    'shoulder_lift': (-100.0, 100.0),
    'elbow_flex': (-97.0, 97.0),
    'wrist_flex': (-95.0, 95.0),
    'wrist_roll': (-157.0, 163.0),
    'gripper': (0.0, 100.0),
}
e1 = MappingEngine.from_yaml('examples/joycon_to_so101/gyro_primary_mapping.yaml', limits)
print(f'Gyro: {len(e1.mappings)} mappings, {e1.motors}, {len(e1.presets)} presets')
e2 = MappingEngine.from_yaml('examples/joycon_to_so101/stick_only_mapping.yaml', limits)
print(f'Stick: {len(e2.mappings)} mappings, {e2.motors}, {len(e2.presets)} presets')
print('OK')
"
```
Expected: Both files load without errors, 6 motors each, 2 presets each.

- [ ] **Step 4: Commit**

```bash
git add examples/joycon_to_so101/gyro_primary_mapping.yaml examples/joycon_to_so101/stick_only_mapping.yaml
git commit -m "feat(joycon): add gyro-primary and stick-only YAML mapping files"
```

---

### Task 5: JoyConTeleop Integration — Dual Mode, Meta Controls, Gyro Inputs

**Files:**
- Modify: `src/lerobot/teleoperators/joycon/teleop_joycon.py`
- Test: `tests/teleoperators/test_joycon.py`

**Interfaces:**
- Consumes: `JoyConTeleopConfig.alt_mapping_path`, `JoyConTeleopConfig.speed_levels`
- Consumes: `JoyConHIDController.get_gyro_deltas()`, `.toggle_filter()`, `.recalibrate_imu()`, `.imu_pitch`, `.imu_roll`, `.gyro_pitch_delta`, `.gyro_roll_delta`, `.speed_levels`, `.speed_level_index`
- Consumes: `MappingEngine.apply_preset()`, `MappingEngine.presets`, expanded `MetaControls`
- Produces: `_active_mode: str` (`"gyro"` or `"stick"`), `_pose_locked: bool`, `_gripper_toggled: bool`

- [ ] **Step 1: Write failing tests for all new JoyConTeleop features**

Add to `tests/teleoperators/test_joycon.py` in `TestJoyConTeleopMappingIntegration`:

```python
    def test_build_input_state_includes_gyro(self):
        """_build_input_state includes gyro_pitch_delta and gyro_roll_delta."""
        teleop = self._make_teleop_with_mock_controller()
        teleop.controller.gyro_pitch_delta = 1.5
        teleop.controller.gyro_roll_delta = -0.8
        teleop.controller.imu_pitch = 10.0
        teleop.controller.imu_roll = -5.0
        state = teleop._build_input_state()
        assert state["gyro_pitch_delta"] == 1.5
        assert state["gyro_roll_delta"] == -0.8
        assert state["imu_pitch"] == 10.0
        assert state["imu_roll"] == -5.0

    def test_pose_lock_freezes_targets(self):
        """When pose_locked is True, get_action returns frozen targets regardless of inputs."""
        teleop = self._make_teleop_with_mock_controller()
        teleop.init_targets({
            "shoulder_pan.pos": 45.0, "shoulder_lift.pos": -30.0,
            "elbow_flex.pos": 10.0, "wrist_flex.pos": 5.0,
            "wrist_roll.pos": -20.0, "gripper.pos": 50.0,
        })
        teleop._pose_locked = True
        # Even with stick input, targets should not change
        teleop.controller.get_raw_left_stick.return_value = (1.0, 1.0)
        action = teleop.get_action()
        assert action["shoulder_pan.pos"] == 45.0  # frozen

    def test_pose_lock_toggle_via_meta_control(self):
        """D-pad right toggles pose lock (edge-triggered)."""
        teleop = self._make_teleop_with_mock_controller()
        assert teleop._pose_locked is False
        teleop.controller.buttons["right"] = True
        teleop.get_action()
        assert teleop._pose_locked is True
        # Toggle off
        teleop.controller.buttons = {k: False for k in teleop.controller.buttons}
        teleop.get_action()
        teleop.controller.buttons["right"] = True
        teleop.get_action()
        assert teleop._pose_locked is False

    def test_reset_to_center(self):
        """D-pad down resets all targets to midpoint of joint limits."""
        teleop = self._make_teleop_with_mock_controller()
        teleop.init_targets({
            "shoulder_pan.pos": 45.0, "shoulder_lift.pos": -30.0,
            "elbow_flex.pos": 10.0, "wrist_flex.pos": 5.0,
            "wrist_roll.pos": -20.0, "gripper.pos": 50.0,
        })
        teleop.controller.buttons["down"] = True
        teleop.get_action()
        # All targets should be at midpoint
        assert teleop._current_targets["shoulder_pan"] == 0.0
        assert teleop._current_targets["shoulder_lift"] == 0.0
        assert teleop._current_targets["elbow_flex"] == 0.0

    def test_preset_pickup_via_b_button(self):
        """B button applies 'pickup' preset."""
        teleop = self._make_teleop_with_mock_controller()
        teleop.mapping_engine.presets = {
            "pickup": {"shoulder_lift": -30.0, "elbow_flex": 45.0},
        }
        teleop.init_targets({
            "shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0, "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0, "gripper.pos": 50.0,
        })
        teleop.controller.buttons["b"] = True
        teleop.get_action()
        assert teleop._current_targets["shoulder_lift"] == -30.0
        assert teleop._current_targets["elbow_flex"] == 45.0
        # Unchanged motors keep values
        assert teleop._current_targets["shoulder_pan"] == 0.0

    def test_recalibrate_imu_via_meta_control(self):
        """D-pad left triggers IMU recalibration."""
        teleop = self._make_teleop_with_mock_controller()
        teleop.controller.buttons["left"] = True
        teleop.get_action()
        teleop.controller.recalibrate_imu.assert_called_once()

    def test_filter_toggle_via_meta_control(self):
        """SR toggles IMU filter mode."""
        teleop = self._make_teleop_with_mock_controller()
        teleop.controller.buttons["sr_right"] = True
        teleop.get_action()
        teleop.controller.toggle_filter.assert_called_once()

    def test_gripper_toggle_via_y_button(self):
        """Y button toggles gripper between fully open (0) and fully closed (100)."""
        teleop = self._make_teleop_with_mock_controller()
        teleop.init_targets({
            "shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0, "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0, "gripper.pos": 50.0,
        })
        # Press Y → gripper opens (100)
        teleop.controller.buttons["y"] = True
        teleop.get_action()
        assert teleop._current_targets["gripper"] == 100.0
        # Release Y
        teleop.controller.buttons["y"] = False
        teleop.get_action()
        # Press Y again → gripper closes (0)
        teleop.controller.buttons["y"] = True
        teleop.get_action()
        assert teleop._current_targets["gripper"] == 0.0

    def test_mode_switch_swaps_engines(self, tmp_path):
        """SL toggles between primary and alt mapping engines."""
        # Create two YAML files
        gyro_yaml = tmp_path / "gyro.yaml"
        gyro_yaml.write_text("""\
mappings:
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
    speed: 1.0
""")
        stick_yaml = tmp_path / "stick.yaml"
        stick_yaml.write_text("""\
mappings:
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
    speed: 2.0
""")
        config = JoyConTeleopConfig(
            mode=JoyConMode.SINGLE_LEFT,
            mapping_path=str(gyro_yaml),
            alt_mapping_path=str(stick_yaml),
        )
        teleop = JoyConTeleop(config)
        # Manually set up mock (skip connect)
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
            "sl_right": False, "sr_right": False,
        }
        ctrl.get_raw_left_stick.return_value = (0.0, 0.0)
        ctrl.get_raw_right_stick.return_value = (0.0, 0.0)
        ctrl.get_wrist_angles.return_value = (0.0, 0.0)
        ctrl.speed_multiplier = 1.0
        ctrl.fine_tune = False
        ctrl.update = MagicMock()
        teleop.controller = ctrl

        teleop.connect = lambda **kw: None  # skip real connect
        # Manually load engines
        from lerobot.teleoperators.joycon.mapping_engine import MappingEngine
        teleop.mapping_engine = MappingEngine.from_yaml(gyro_yaml, SO101_JOINT_LIMITS)
        teleop.alt_mapping_engine = MappingEngine.from_yaml(stick_yaml, SO101_JOINT_LIMITS)
        teleop._current_targets = {m: 0.0 for m in teleop.mapping_engine.motors}
        teleop._prev_mode_switch = False

        assert teleop._active_mode == "gyro"
        assert teleop.mapping_engine.mappings[0].speed == 1.0

        # Press SL → switch to stick mode
        ctrl.buttons["sl_right"] = True
        teleop.get_action()
        assert teleop._active_mode == "stick"
        # The active engine should now be the stick one (speed=2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/teleoperators/test_joycon.py::TestJoyConTeleopMappingIntegration::test_build_input_state_includes_gyro -xvs`
Expected: FAIL — `KeyError: 'gyro_pitch_delta'` or `AttributeError`

- [ ] **Step 3: Update `__init__` with new state variables**

In `teleop_joycon.py`, replace `__init__` (lines 74-82) with:

```python
    def __init__(self, config: JoyConTeleopConfig):
        super().__init__(config)
        self.config = config
        self.controller = None  # JoyConHIDController, set on connect()
        self.mapping_engine: MappingEngine | None = None
        self.alt_mapping_engine: MappingEngine | None = None
        self._active_mode: str = "gyro"  # "gyro" or "stick"
        self._current_targets: dict[str, float] = {}
        # Edge detection state
        self._prev_speed_up: bool = False
        self._prev_fine_tune: bool = False
        self._prev_reset_center: bool = False
        self._prev_recalibrate: bool = False
        self._prev_pose_lock: bool = False
        self._prev_mode_switch: bool = False
        self._prev_filter_toggle: bool = False
        self._prev_b_button: bool = False
        self._prev_x_button: bool = False
        self._prev_y_button: bool = False
        # State flags
        self._pose_locked: bool = False
        self._gripper_toggled: bool = False  # Y button gripper state
```

- [ ] **Step 4: Update `connect()` to load dual engines**

In `connect()` (lines 105-166), after loading the primary mapping engine (line 152), add alt engine loading:

```python
        # Load alternate mapping engine (stick mode)
        if self.config.alt_mapping_path:
            self.alt_mapping_engine = MappingEngine.from_yaml(
                self.config.alt_mapping_path, joint_limits
            )
            logger.info(
                "Alt mapping loaded from %s: %d motors.",
                self.config.alt_mapping_path,
                len(self.alt_mapping_engine.motors),
            )

        # Pass speed_levels to controller
        self.controller.speed_levels = list(self.config.speed_levels)
        self.controller.speed_level_index = 1  # normal
        self.controller.speed_multiplier = self.controller.speed_levels[1]

        # Initialize state
        self._active_mode = "gyro"
        self._current_targets = {}
        self._pose_locked = False
        self._gripper_toggled = False
        self._prev_speed_up = False
        self._prev_fine_tune = False
        self._prev_reset_center = False
        self._prev_recalibrate = False
        self._prev_pose_lock = False
        self._prev_mode_switch = False
        self._prev_filter_toggle = False
        self._prev_b_button = False
        self._prev_x_button = False
        self._prev_y_button = False
```

Remove the old state init lines (lines 156-159) that are now covered above.

- [ ] **Step 5: Update `_build_input_state()` with gyro inputs**

Replace `_build_input_state()` (lines 228-264) with:

```python
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
            "sl_right": buttons.get("sl_right", False),
            "sl_left": buttons.get("sl_left", False),
            "sr_right": buttons.get("sr_right", False),
            "sr_left": buttons.get("sr_left", False),
            # IMU (float, degrees)
            "imu_tilt": imu_tilt,
            "imu_roll": imu_roll,
            "imu_pitch": ctrl.imu_pitch,
            # Gyro deltas (float, degrees per frame)
            "gyro_pitch_delta": ctrl.gyro_pitch_delta,
            "gyro_roll_delta": ctrl.gyro_roll_delta,
        }
```

- [ ] **Step 6: Replace `_handle_meta_controls()` with expanded version**

Replace `_handle_meta_controls()` (lines 266-289) with:

```python
    def _handle_meta_controls(self, input_state: dict) -> None:
        """Process all meta-control buttons (edge-triggered).

        Handles: speed cycling, fine-tune, reset-to-center, IMU recalibrate,
        pose lock, mode switch, filter toggle, presets (B/X), gripper toggle (Y).
        """
        meta = self.mapping_engine.meta_controls

        # ── Speed cycling (D-pad up) ─────────────────────────────────
        speed_up = input_state.get(meta.speed_up, False)
        if speed_up and not self._prev_speed_up:
            ctrl = self.controller
            ctrl.speed_level_index = (ctrl.speed_level_index + 1) % len(ctrl.speed_levels)
            ctrl.speed_multiplier = ctrl.speed_levels[ctrl.speed_level_index]
        self._prev_speed_up = speed_up

        # ── Fine-tune toggle (stick press) ───────────────────────────
        fine_toggle = input_state.get(meta.fine_tune_toggle, False)
        if fine_toggle and not self._prev_fine_tune:
            self.controller.fine_tune = not self.controller.fine_tune
        self._prev_fine_tune = fine_toggle

        # ── Reset to center (D-pad down) ─────────────────────────────
        reset = input_state.get(meta.reset_to_center, False)
        if reset and not self._prev_reset_center:
            for motor in self._current_targets:
                self._current_targets[motor] = 0.0
            logger.info("Reset to center: all targets zeroed.")
        self._prev_reset_center = reset

        # ── Recalibrate IMU (D-pad left) ─────────────────────────────
        recal = input_state.get(meta.recalibrate_imu, False)
        if recal and not self._prev_recalibrate:
            self.controller.recalibrate_imu()
        self._prev_recalibrate = recal

        # ── Pose lock toggle (D-pad right) ───────────────────────────
        pose = input_state.get(meta.pose_lock, False)
        if pose and not self._prev_pose_lock:
            self._pose_locked = not self._pose_locked
            state = "LOCKED" if self._pose_locked else "UNLOCKED"
            logger.info("Pose: %s", state)
        self._prev_pose_lock = pose

        # ── Mode switch (SL) ─────────────────────────────────────────
        mode_btn = input_state.get(meta.mode_switch, False)
        if mode_btn and not self._prev_mode_switch and self.alt_mapping_engine is not None:
            if self._active_mode == "gyro":
                self._active_mode = "stick"
                self.mapping_engine, self.alt_mapping_engine = (
                    self.alt_mapping_engine, self.mapping_engine
                )
            else:
                self._active_mode = "gyro"
                self.mapping_engine, self.alt_mapping_engine = (
                    self.alt_mapping_engine, self.mapping_engine
                )
            logger.info("Mode switched to: %s", self._active_mode)
        self._prev_mode_switch = mode_btn

        # ── Filter toggle (SR) ───────────────────────────────────────
        filt = input_state.get(meta.filter_toggle, False)
        if filt and not self._prev_filter_toggle:
            self.controller.toggle_filter()
        self._prev_filter_toggle = filt

        # ── Preset pickup (B button) ─────────────────────────────────
        b_btn = input_state.get("b", False)
        if b_btn and not self._prev_b_button:
            self._current_targets = self.mapping_engine.apply_preset(
                "pickup", self._current_targets
            )
        self._prev_b_button = b_btn

        # ── Preset place (X button) ──────────────────────────────────
        x_btn = input_state.get("x", False)
        if x_btn and not self._prev_x_button:
            self._current_targets = self.mapping_engine.apply_preset(
                "place", self._current_targets
            )
        self._prev_x_button = x_btn

        # ── Gripper toggle (Y button) ────────────────────────────────
        y_btn = input_state.get("y", False)
        if y_btn and not self._prev_y_button:
            self._gripper_toggled = not self._gripper_toggled
            if "gripper" in self._current_targets:
                lo, hi = self.mapping_engine.joint_limits.get("gripper", (0.0, 100.0))
                self._current_targets["gripper"] = hi if self._gripper_toggled else lo
        self._prev_y_button = y_btn
```

- [ ] **Step 7: Update `get_action()` to respect pose lock**

Replace `get_action()` (lines 211-226) with:

```python
    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Read Joy-Con inputs and return motor position targets."""
        self.controller.update()

        # If pose is locked, return frozen targets without processing inputs
        if self._pose_locked:
            return {f"{motor}.pos": pos for motor, pos in self._current_targets.items()}

        input_state = self._build_input_state()
        self._handle_meta_controls(input_state)

        # After meta controls, check pose lock again (may have been toggled)
        if self._pose_locked:
            return {f"{motor}.pos": pos for motor, pos in self._current_targets.items()}

        targets = self.mapping_engine.compute_targets(
            input_state,
            self._current_targets,
            speed_multiplier=self.controller.speed_multiplier,
            fine_tune=self.controller.fine_tune,
        )
        self._current_targets = targets

        return {f"{motor}.pos": pos for motor, pos in targets.items()}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run pytest tests/teleoperators/test_joycon.py::TestJoyConTeleopMappingIntegration -xvs`
Expected: All integration tests PASS

- [ ] **Step 9: Run full test suite**

Run: `uv run pytest tests/teleoperators/test_joycon.py -xvs`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add src/lerobot/teleoperators/joycon/teleop_joycon.py tests/teleoperators/test_joycon.py
git commit -m "feat(joycon): integrate dual mode, pose lock, presets, and gyro meta controls"
```

---

### Task 6: Update Example Script

**Files:**
- Modify: `examples/joycon_to_so101/teleoperate.py`

**Interfaces:**
- Consumes: `JoyConTeleopConfig.alt_mapping_path`, `JoyConTeleop._active_mode`

- [ ] **Step 1: Add `--alt-mapping` argument and pass to config**

In `teleoperate.py`, update the `teleoperate()` function signature (line 49):

```python
def teleoperate(
    port: str,
    mode: str = "auto",
    mapping_path: str | None = None,
    alt_mapping_path: str | None = None,
    fps: int = 30,
):
```

Update the config creation (line 59):

```python
    teleop_config = JoyConTeleopConfig(
        mode=joycon_mode,
        mapping_path=mapping_path,
        alt_mapping_path=alt_mapping_path,
    )
```

- [ ] **Step 2: Update control summary printout**

Replace the print block (lines 94-107) with:

```python
    print("\n" + "=" * 50)
    print("  SO-101 Joy-Con Teleop (gyro-primary)")
    print("=" * 50)
    print(f"  Mapping:      {mapping_path or 'built-in default'}")
    print(f"  Alt mapping:  {alt_mapping_path or 'none'}")
    print(f"  Motors:       {', '.join(teleop.mapping_engine.motors)}")
    print(f"  FPS:          {fps}")
    print()
    if alt_mapping_path:
        print("  SL           → Toggle gyro/stick mode")
    print("  SR           → Toggle IMU filter (normal/stabilized)")
    print("  D-pad ↑      → Speed level cycle")
    print("  D-pad ↓      → Reset to center")
    print("  D-pad ←      → Recalibrate IMU")
    print("  D-pad →      → Pose lock toggle")
    print("  B / X        → Pickup / Place preset")
    print("  Y            → Gripper toggle")
    print("  A / Home     → Emergency stop")
    print("  Plus/Minus   → Episode success/failure")
    print("=" * 50 + "\n")
```

- [ ] **Step 3: Update speed display in main loop**

Replace the speed print block (lines 124-130) with:

```python
            # Print state changes
            cur_speed = events.get("speed_multiplier", 1.0)
            cur_fine = events.get("fine_tune", False)
            cur_mode = getattr(teleop, '_active_mode', 'gyro')
            if cur_speed != last_speed or cur_fine != last_fine_tune:
                ft_str = "ON" if cur_fine else "OFF"
                print(f"  Speed: {cur_speed:.1f}x | Fine: {ft_str} | Mode: {cur_mode}")
                last_speed = cur_speed
                last_fine_tune = cur_fine
```

- [ ] **Step 4: Add `--alt-mapping` CLI argument**

Add to the `main()` argparse (after line 163):

```python
    parser.add_argument(
        "--alt-mapping",
        default=None,
        help="Path to alternate mapping YAML (stick mode, toggled by SL)",
    )
```

Update the `teleoperate()` call (line 169):

```python
    teleoperate(
        port=args.port,
        mode=args.mode,
        mapping_path=args.mapping,
        alt_mapping_path=args.alt_mapping,
        fps=args.fps,
    )
```

- [ ] **Step 5: Verify the script parses correctly**

Run: `uv run python -c "import ast; ast.parse(open('examples/joycon_to_so101/teleoperate.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add examples/joycon_to_so101/teleoperate.py
git commit -m "feat(joycon): update teleoperate.py with gyro mode and dual mapping"
```

---

### Task 7: Integration Test + Pre-commit

**Files:**
- Test: `tests/teleoperators/test_joycon.py`

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/teleoperators/test_joycon.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 2: Run pre-commit on changed files**

Run: `uv run pre-commit run --files src/lerobot/teleoperators/joycon/joycon_utils.py src/lerobot/teleoperators/joycon/mapping_engine.py src/lerobot/teleoperators/joycon/teleop_joycon.py src/lerobot/teleoperators/joycon/configuration_joycon.py examples/joycon_to_so101/teleoperate.py`
Expected: All hooks PASS (ruff, typos, etc.)

- [ ] **Step 3: Fix any linting issues**

If pre-commit reports issues, fix them and re-run.

- [ ] **Step 4: Final commit if needed**

```bash
git add -A
git commit -m "chore: fix linting issues from gyro teleop implementation"
```
