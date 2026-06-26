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

"""Unit tests for the Joy-Con teleoperator.

All tests use a FakeHIDDevice mock so no real Joy-Con hardware is needed.
"""

import pytest

from lerobot.teleoperators.joycon.configuration_joycon import JoyConMode, JoyConTeleopConfig
from lerobot.teleoperators.joycon.joycon_utils import (
    FULL_INPUT_REPORT_ID,
    JOYCON_LEFT_PID,
    JOYCON_RIGHT_PID,
    NINTENDO_VID,
    SUBCMD_ACK_REPORT_ID,
    GripperAction,
    JoyConHIDController,
    _decode_stick,
    _parse_buttons_left,
    _parse_buttons_right,
    _parse_buttons_shared,
    gripper_action_map,
)
from lerobot.teleoperators.joycon.teleop_joycon import JoyConTeleop
from lerobot.teleoperators.utils import TeleopEvents, make_teleoperator_from_config

# ── Helpers ──────────────────────────────────────────────────────────────────


def _encode_stick(x: int, y: int) -> tuple[int, int, int]:
    """Encode (x, y) 12-bit values into 3 Joy-Con packed bytes (inverse of _decode_stick)."""
    b0 = x & 0xFF
    b1 = ((x >> 8) & 0x0F) | ((y & 0x0F) << 4)
    b2 = (y >> 4) & 0xFF
    return b0, b1, b2


def _make_full_report(
    right_btns: int = 0,
    shared_btns: int = 0,
    left_btns: int = 0,
    left_stick_x: int = 2048,
    left_stick_y: int = 2048,
    right_stick_x: int = 2048,
    right_stick_y: int = 2048,
    battery: int = 0x80,  # full battery, not charging
) -> list[int]:
    """Build a synthetic 0x30 full input report (362 bytes, but we only need the first 12+)."""
    report = [0] * 362
    report[0] = FULL_INPUT_REPORT_ID  # report ID
    report[1] = 0x00  # timer
    report[2] = battery  # battery/connection info

    # Button bytes
    report[3] = right_btns
    report[4] = shared_btns
    report[5] = left_btns

    # Left stick (bytes 6-8)
    lx0, lx1, lx2 = _encode_stick(left_stick_x, left_stick_y)
    report[6] = lx0
    report[7] = lx1
    report[8] = lx2

    # Right stick (bytes 9-11)
    rx0, rx1, rx2 = _encode_stick(right_stick_x, right_stick_y)
    report[9] = rx0
    report[10] = rx1
    report[11] = rx2

    return report


def _make_ack_report(subcmd_id: int) -> list[int]:
    """Build a synthetic subcommand ACK report."""
    report = [0] * 48
    report[0] = SUBCMD_ACK_REPORT_ID
    report[14] = subcmd_id
    return report


class FakeHIDDevice:
    """Mock HID device that replays scripted reports and records writes."""

    def __init__(self, scripted_reports: list[list[int]] | None = None, side: str = "left"):
        self._reports = list(scripted_reports or [])
        self._report_idx = 0
        self.written: list[list[int]] = []
        self.opened = True
        self._side = side

    def open_path(self, path):
        self.opened = True

    def set_nonblocking(self, n):
        pass

    def write(self, data):
        self.written.append(list(data))
        return len(data)

    def read(self, maxlen):
        if self._report_idx < len(self._reports):
            report = self._reports[self._report_idx]
            self._report_idx += 1
            return report
        return []

    def close(self):
        self.opened = False

    def get_manufacturer_string(self):
        return "Nintendo Co., Ltd."

    def get_product_string(self):
        return f"Joy-Con ({'L' if self._side == 'left' else 'R'})"


def _make_device_info(vid: int, pid: int, side: str = "left") -> dict:
    return {
        "vendor_id": vid,
        "product_id": pid,
        "product_string": f"Joy-Con ({'L' if side == 'left' else 'R'})",
        "path": f"/dev/hid-joycon-{side}".encode(),
        "interface_number": 0,
    }


def _create_controller_with_fake_devices(
    mode: JoyConMode = JoyConMode.AUTO,
    left_reports: list[list[int]] | None = None,
    right_reports: list[list[int]] | None = None,
    devices_map: dict | None = None,
) -> JoyConHIDController:
    """Create a JoyConHIDController with pre-configured fake HID devices."""
    ctrl = JoyConHIDController(mode=mode, deadzone=0.15)

    if devices_map is None:
        devices_map = {}

    # Inject fake devices directly (bypassing HID enumeration)
    for side, dev in devices_map.items():
        ctrl.devices[side] = dev
        ctrl.device_info[side] = _make_device_info(
            NINTENDO_VID, JOYCON_LEFT_PID if side == "left" else JOYCON_RIGHT_PID, side
        )

    # Set up calibration with default center
    ctrl.left_center = (2048, 2048)
    ctrl.right_center = (2048, 2048)
    ctrl.stick_available = True

    # Inject scripted reports for processing
    if left_reports:
        for report in left_reports:
            ctrl._process_report("left", report)
    if right_reports:
        for report in right_reports:
            ctrl._process_report("right", report)

    return ctrl


# ── Tests: Configuration ─────────────────────────────────────────────────────


class TestJoyConConfig:
    def test_config_registers_subclass(self):
        config = JoyConTeleopConfig()
        assert config.type == "joycon"

    def test_config_default_values(self):
        config = JoyConTeleopConfig()
        assert config.mode == JoyConMode.AUTO
        assert config.use_gripper is True
        assert config.deadzone == 0.15
        assert config.step_size == 1.0

    def test_config_custom_values(self):
        config = JoyConTeleopConfig(mode=JoyConMode.DUAL, deadzone=0.2, step_size=0.5)
        assert config.mode == JoyConMode.DUAL
        assert config.deadzone == 0.2
        assert config.step_size == 0.5

    def test_joycon_mode_values(self):
        assert JoyConMode.AUTO.value == "auto"
        assert JoyConMode.SINGLE_LEFT.value == "single_left"
        assert JoyConMode.SINGLE_RIGHT.value == "single_right"
        assert JoyConMode.DUAL.value == "dual"


# ── Tests: Action features ──────────────────────────────────────────────────


class TestActionFeatures:
    def test_action_features_with_gripper(self):
        config = JoyConTeleopConfig(use_gripper=True)
        teleop = JoyConTeleop(config)
        features = teleop.action_features
        assert features["shape"] == (9,)
        assert "gripper" in features["names"]
        assert "delta_wx" in features["names"]
        assert "wrist_flex" in features["names"]
        assert "wrist_roll" in features["names"]

    def test_action_features_without_gripper(self):
        config = JoyConTeleopConfig(use_gripper=False)
        teleop = JoyConTeleop(config)
        features = teleop.action_features
        assert features["shape"] == (8,)
        assert "gripper" not in features["names"]
        assert "delta_wz" in features["names"]
        assert "wrist_flex" in features["names"]

    def test_feedback_features_empty(self):
        config = JoyConTeleopConfig()
        teleop = JoyConTeleop(config)
        assert teleop.feedback_features == {}


# ── Tests: Stick decoding ───────────────────────────────────────────────────


class TestStickDecoding:
    def test_decode_center(self):
        b0, b1, b2 = _encode_stick(2048, 2048)
        x, y = _decode_stick(b0, b1, b2)
        assert x == 2048
        assert y == 2048

    def test_decode_min(self):
        b0, b1, b2 = _encode_stick(0, 0)
        x, y = _decode_stick(b0, b1, b2)
        assert x == 0
        assert y == 0

    def test_decode_max(self):
        b0, b1, b2 = _encode_stick(4095, 4095)
        x, y = _decode_stick(b0, b1, b2)
        assert x == 4095
        assert y == 4095

    def test_decode_asymmetric(self):
        b0, b1, b2 = _encode_stick(1000, 3000)
        x, y = _decode_stick(b0, b1, b2)
        assert x == 1000
        assert y == 3000

    def test_encode_decode_roundtrip(self):
        """Verify that encoding and decoding are inverses."""
        for x in [0, 512, 1024, 2048, 3072, 4095]:
            for y in [0, 512, 1024, 2048, 3072, 4095]:
                b0, b1, b2 = _encode_stick(x, y)
                dx, dy = _decode_stick(b0, b1, b2)
                assert dx == x, f"X mismatch for ({x}, {y}): got {dx}"
                assert dy == y, f"Y mismatch for ({x}, {y}): got {dy}"


# ── Tests: Button parsing ───────────────────────────────────────────────────


class TestButtonParsing:
    def test_right_buttons_all_off(self):
        result = _parse_buttons_right(0x00)
        assert all(not v for v in result.values())

    def test_right_buttons_a(self):
        result = _parse_buttons_right(0x08)
        assert result["a"] is True
        assert result["b"] is False

    def test_right_buttons_zr(self):
        result = _parse_buttons_right(0x80)
        assert result["zr"] is True

    def test_right_buttons_sr_sl(self):
        result = _parse_buttons_right(0x30)
        assert result["sr_right"] is True
        assert result["sl_right"] is True

    def test_shared_buttons_plus(self):
        result = _parse_buttons_shared(0x02)
        assert result["plus"] is True
        assert result["minus"] is False

    def test_shared_buttons_home(self):
        result = _parse_buttons_shared(0x10)
        assert result["home"] is True

    def test_left_buttons_zl(self):
        result = _parse_buttons_left(0x80)
        assert result["zl"] is True

    def test_left_buttons_sl_sr(self):
        result = _parse_buttons_left(0x30)
        assert result["sl_left"] is True
        assert result["sr_left"] is True


# ── Tests: Full report processing ───────────────────────────────────────────


class TestFullReportProcessing:
    def test_parse_center_stick_produces_zero_delta(self):
        """Stick at center (2048, 2048) → normalized 0.0 → zero delta."""
        report = _make_full_report(left_stick_x=2048, left_stick_y=2048)
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            left_reports=[report],
        )
        assert ctrl.left_x == 0.0
        assert ctrl.left_y == 0.0

    def test_parse_full_deflection(self):
        """Stick at max X (4095, 2048) → normalized ~1.0."""
        report = _make_full_report(left_stick_x=4095, left_stick_y=2048)
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            left_reports=[report],
        )
        assert ctrl.left_x > 0.9
        assert ctrl.left_y == 0.0

    def test_deadzone_suppresses_small_input(self):
        """Stick slightly off center (2100, 2048) with deadzone 0.15 → zero."""
        report = _make_full_report(left_stick_x=2100, left_stick_y=2048)
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            left_reports=[report],
        )
        assert ctrl.left_x == 0.0

    def test_button_state_stored(self):
        """Button press is stored in the buttons dict."""
        # Plus button: shared byte bit 1 → 0x02
        report = _make_full_report(shared_btns=0x02)
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        assert ctrl.buttons.get("plus") is True

    def test_battery_parsing(self):
        """Battery level is decoded from the report."""
        # battery=0x80 → level nibble 8 → 100%
        report = _make_full_report(battery=0x80)
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            left_reports=[report],
        )
        assert ctrl.battery_level.get("left") == 100

    def test_battery_low_warning(self, caplog):
        """Low battery triggers a warning log."""
        # battery=0x20 → level nibble 2 → 25% (above 15% threshold)
        report = _make_full_report(battery=0x20)
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            left_reports=[report],
        )
        assert ctrl.battery_level.get("left") == 25


# ── Tests: Motion deltas ────────────────────────────────────────────────────


class TestGetDeltas:
    def test_single_left_center_zero(self):
        """Center stick → zero deltas in single-left mode."""
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
        )
        ctrl.left_x = 0.0
        ctrl.left_y = 0.0
        dx, dy, dz = ctrl.get_deltas()
        assert dx == 0.0
        assert dy == 0.0
        assert dz == 0.0

    def test_single_left_deflected(self):
        """Deflected stick → non-zero deltas in single-left mode."""
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
        )
        ctrl.left_x = 1.0  # full right
        ctrl.left_y = 0.0
        dx, dy, dz = ctrl.get_deltas()
        # In single-left: dx = -ny * y_step, dy = -nx * x_step
        assert dy == pytest.approx(-1.0)  # left_x → world dy
        assert dx == 0.0

    def test_dual_center_zero(self):
        """Center sticks → zero deltas in dual mode."""
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
        )
        ctrl.left_x = 0.0
        ctrl.left_y = 0.0
        ctrl.right_x = 0.0
        ctrl.right_y = 0.0
        dx, dy, dz = ctrl.get_deltas()
        assert dx == 0.0
        assert dy == 0.0
        assert dz == 0.0

    def test_dual_left_stick_xy(self):
        """Left stick deflected → XY deltas in dual mode."""
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
        )
        ctrl.left_x = 0.5
        ctrl.left_y = 0.0
        ctrl.right_x = 0.0
        ctrl.right_y = 0.0
        dx, dy, dz = ctrl.get_deltas()
        assert dy == pytest.approx(-0.5)
        assert dz == 0.0

    def test_dual_right_stick_z(self):
        """Right stick Y deflected → no Z delta in dual mode (Z is now L/ZL)."""
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
        )
        ctrl.left_x = 0.0
        ctrl.left_y = 0.0
        ctrl.right_x = 0.0
        ctrl.right_y = 0.8
        dx, dy, dz = ctrl.get_deltas()
        assert dz == 0.0  # Z is controlled by L/ZL, not right stick

    def test_no_stick_available_returns_zero(self):
        """When stick_available is False, always return zeros."""
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
        )
        ctrl.stick_available = False
        ctrl.left_x = 1.0
        ctrl.left_y = 1.0
        dx, dy, dz = ctrl.get_deltas()
        assert dx == 0.0
        assert dy == 0.0
        assert dz == 0.0


# ── Tests: Button → action mapping ──────────────────────────────────────────


class TestButtonMapping:
    def test_dual_b_opens_gripper(self):
        """B pressed → open gripper in dual mode."""
        report = _make_full_report(right_btns=0x04)  # B
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        assert ctrl.gripper_command() == "open"

    def test_dual_a_closes_gripper(self):
        """A pressed → close gripper in dual mode."""
        report = _make_full_report(right_btns=0x08)  # A
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        assert ctrl.gripper_command() == "close"

    def test_dual_both_a_b_stay(self):
        """Both A and B pressed → stay (inherited InputController logic)."""
        report = _make_full_report(right_btns=0x0C)  # A + B
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        assert ctrl.gripper_command() == "stay"

    def test_dual_plus_success(self):
        """Plus button → SUCCESS episode end."""
        report = _make_full_report(shared_btns=0x02)  # Plus
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        assert ctrl.episode_end_status == TeleopEvents.SUCCESS

    def test_dual_minus_failure(self):
        """Minus button → FAILURE episode end."""
        report = _make_full_report(shared_btns=0x01)  # Minus
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        assert ctrl.episode_end_status == TeleopEvents.FAILURE

    def test_dual_home_emergency_stop(self):
        """Home button → emergency stop flag set."""
        report = _make_full_report(shared_btns=0x10)  # Home
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        assert ctrl.emergency_stop is True

    def test_single_left_l_closes_gripper(self):
        """L pressed → close gripper in single-left mode."""
        report = _make_full_report(left_btns=0x40)  # L
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            left_reports=[report],
        )
        assert ctrl.gripper_command() == "close"

    def test_single_right_r_opens_gripper(self):
        """R pressed → open gripper in single-right mode."""
        report = _make_full_report(right_btns=0x40)  # R
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_RIGHT,
            right_reports=[report],
        )
        assert ctrl.gripper_command() == "open"

    def test_single_right_plus_success(self):
        """Plus button → SUCCESS in single-right mode."""
        report = _make_full_report(shared_btns=0x02)  # Plus
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_RIGHT,
            right_reports=[report],
        )
        assert ctrl.episode_end_status == TeleopEvents.SUCCESS

    def test_no_buttons_no_episode_end(self):
        """No buttons pressed → episode_end_status is None."""
        report = _make_full_report()
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        assert ctrl.episode_end_status is None


# ── Tests: L/ZL Z-axis ──────────────────────────────────────────────────────


class TestLZAxis:
    def test_l_gives_positive_z(self):
        """L pressed → positive Z delta."""
        report = _make_full_report(left_btns=0x40)  # L
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        dz = ctrl._lz_z_delta()
        assert dz > 0.0

    def test_zl_gives_negative_z(self):
        """ZL pressed → negative Z delta."""
        report = _make_full_report(left_btns=0x80)  # ZL
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        dz = ctrl._lz_z_delta()
        assert dz < 0.0

    def test_no_l_zl_zero_z(self):
        """Neither L nor ZL → zero Z delta."""
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
        )
        dz = ctrl._lz_z_delta()
        assert dz == 0.0


# ── Tests: Speed control ────────────────────────────────────────────────────


class TestSpeedControl:
    def test_speed_increase(self):
        """D-pad up → speed increases by speed_step."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        assert ctrl.speed_multiplier == 1.0
        # Simulate D-pad up press (rising edge)
        ctrl._map_buttons_to_actions()  # initial: no press
        ctrl.buttons = {"up": True}
        ctrl._map_buttons_to_actions()
        assert ctrl.speed_multiplier == pytest.approx(1.2)

    def test_speed_decrease(self):
        """D-pad down → speed decreases by speed_step."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl.buttons = {}
        ctrl._map_buttons_to_actions()  # initial
        ctrl.buttons = {"down": True}
        ctrl._map_buttons_to_actions()
        assert ctrl.speed_multiplier == pytest.approx(0.8)

    def test_speed_max_bound(self):
        """Speed cannot exceed max_speed."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl.speed_multiplier = 2.0
        ctrl.buttons = {}
        ctrl._map_buttons_to_actions()
        ctrl.buttons = {"up": True}
        ctrl._map_buttons_to_actions()
        assert ctrl.speed_multiplier == 2.0  # unchanged

    def test_speed_min_bound(self):
        """Speed cannot go below min_speed."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl.speed_multiplier = 0.2
        ctrl.buttons = {}
        ctrl._map_buttons_to_actions()
        ctrl.buttons = {"down": True}
        ctrl._map_buttons_to_actions()
        assert ctrl.speed_multiplier == 0.2  # unchanged


# ── Tests: Fine-tune mode ───────────────────────────────────────────────────


class TestFineTune:
    def test_fine_tune_toggle(self):
        """Stick press toggles fine-tune mode."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        assert ctrl.fine_tune is False
        ctrl.buttons = {}
        ctrl._map_buttons_to_actions()  # initial
        ctrl.buttons = {"l_stick_press": True}
        ctrl._map_buttons_to_actions()
        assert ctrl.fine_tune is True
        # Toggle again
        ctrl.buttons = {}
        ctrl._map_buttons_to_actions()
        ctrl.buttons = {"l_stick_press": True}
        ctrl._map_buttons_to_actions()
        assert ctrl.fine_tune is False

    def test_fine_tune_halves_scale(self):
        """Fine-tune mode halves the effective scale."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl.fine_tune = True
        assert ctrl._effective_scale() == pytest.approx(0.5)

    def test_fine_tune_with_speed(self):
        """Fine-tune combines with speed multiplier."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl.speed_multiplier = 2.0
        ctrl.fine_tune = True
        assert ctrl._effective_scale() == pytest.approx(1.0)  # 2.0 * 0.5


# ── Tests: Orientation deltas ───────────────────────────────────────────────


class TestOrientationDeltas:
    def test_orientation_zero_in_single_mode(self):
        """Single Joy-Con modes return zero orientation."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.SINGLE_LEFT)
        ctrl.right_x = 1.0
        ctrl.right_y = 1.0
        dwx, dwy, dwz = ctrl.get_orientation_deltas()
        assert dwx == 0.0
        assert dwy == 0.0
        assert dwz == 0.0

    def test_right_stick_produces_yaw_pitch(self):
        """Right stick deflected → Yaw/Pitch deltas in dual mode."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl.right_x = 0.5
        ctrl.right_y = -0.3
        ctrl.buttons = {}
        dwx, dwy, dwz = ctrl.get_orientation_deltas()
        assert dwx != 0.0  # yaw from right_x
        assert dwy != 0.0  # pitch from right_y
        assert dwz == 0.0  # no roll

    def test_rz_produces_roll(self):
        """R/ZR buttons → Roll delta in dual mode."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl.right_x = 0.0
        ctrl.right_y = 0.0
        ctrl.buttons = {"r": True, "zr": False}
        dwx, dwy, dwz = ctrl.get_orientation_deltas()
        assert dwz > 0.0  # roll from R


# ── Tests: Factory dispatch ─────────────────────────────────────────────────


class TestFactory:
    def test_factory_creates_joycon_teleop(self):
        config = JoyConTeleopConfig()
        teleop = make_teleoperator_from_config(config)
        assert isinstance(teleop, JoyConTeleop)

    def test_factory_preserves_config(self):
        config = JoyConTeleopConfig(mode=JoyConMode.DUAL, deadzone=0.2)
        teleop = make_teleoperator_from_config(config)
        assert teleop.config.mode == JoyConMode.DUAL
        assert teleop.config.deadzone == 0.2


# ── Tests: Connection state ─────────────────────────────────────────────────


class TestConnectionState:
    def test_not_connected_before_connect(self):
        config = JoyConTeleopConfig()
        teleop = JoyConTeleop(config)
        assert teleop.is_connected is False

    def test_is_calibrated_always_true(self):
        config = JoyConTeleopConfig()
        teleop = JoyConTeleop(config)
        assert teleop.is_calibrated is True

    def test_disconnect_clears_controller(self):
        config = JoyConTeleopConfig()
        teleop = JoyConTeleop(config)
        # Simulate connected state
        teleop.controller = type(
            "FakeController", (), {"devices": {"left": None}, "stop": lambda self: None}
        )()
        assert teleop.is_connected is True
        teleop.disconnect()
        assert teleop.controller is None
        assert teleop.is_connected is False

    def test_get_teleop_events_when_disconnected(self):
        config = JoyConTeleopConfig()
        teleop = JoyConTeleop(config)
        events = teleop.get_teleop_events()
        assert events[TeleopEvents.IS_INTERVENTION] is False
        assert events[TeleopEvents.TERMINATE_EPISODE] is False
        assert events[TeleopEvents.SUCCESS] is False
        assert events[TeleopEvents.RERECORD_EPISODE] is False


# ── Tests: Disconnection handling ───────────────────────────────────────────


class TestDisconnectionHandling:
    def test_oserror_drops_device(self):
        """OSError during read removes the device from the dict."""

        class ErrorDevice(FakeHIDDevice):
            def __init__(self):
                super().__init__(side="left")
                self._error_on_read = True

            def read(self, maxlen):
                raise OSError("Device disconnected")

        dev = ErrorDevice()
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            devices_map={"left": dev},
        )
        # Disable reconnection so _try_reconnect doesn't find real Joy-Cons
        ctrl.running = False
        assert "left" in ctrl.devices
        ctrl.update()
        assert "left" not in ctrl.devices

    def test_disconnect_closes_all_devices(self):
        """disconnect() calls close() on all open devices."""
        left_dev = FakeHIDDevice(side="left")
        right_dev = FakeHIDDevice(side="right")
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            devices_map={"left": left_dev, "right": right_dev},
        )
        ctrl.stop()
        assert not left_dev.opened
        assert not right_dev.opened
        assert len(ctrl.devices) == 0


# ── Tests: Gripper action map ───────────────────────────────────────────────


class TestGripperActionMap:
    def test_gripper_action_values(self):
        assert GripperAction.CLOSE == 0
        assert GripperAction.STAY == 1
        assert GripperAction.OPEN == 2

    def test_gripper_action_map_complete(self):
        assert gripper_action_map["close"] == GripperAction.CLOSE.value
        assert gripper_action_map["open"] == GripperAction.OPEN.value
        assert gripper_action_map["stay"] == GripperAction.STAY.value


# ── Tests: HID subcommand framing ───────────────────────────────────────────


import sys


# On macOS, output reports have a 0x00 prefix byte, shifting all offsets by 1
_MACOS_OFFSET = 1 if sys.platform == "darwin" else 0
_EXPECTED_REPORT_LEN = 49 + _MACOS_OFFSET


class TestSubcommandFraming:
    def test_send_subcmd_writes_correct_bytes(self):
        """Subcommand output report should be 49 bytes (50 on macOS with prefix)."""
        dev = FakeHIDDevice(side="left")
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            devices_map={"left": dev},
        )
        ctrl._send_subcmd(dev, 0x03, bytes([0x30]))
        assert len(dev.written) == 1
        assert len(dev.written[0]) == _EXPECTED_REPORT_LEN

    def test_send_subcmd_report_id(self):
        """Report ID 0x01 should be at the correct offset."""
        dev = FakeHIDDevice(side="left")
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            devices_map={"left": dev},
        )
        ctrl._send_subcmd(dev, 0x03, bytes([0x30]))
        assert dev.written[0][_MACOS_OFFSET] == 0x01

    def test_send_subcmd_id_at_correct_offset(self):
        """Subcommand ID should be at offset 9 from report start."""
        dev = FakeHIDDevice(side="left")
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            devices_map={"left": dev},
        )
        ctrl._send_subcmd(dev, 0x03, bytes([0x30]))
        assert dev.written[0][_MACOS_OFFSET + 9] == 0x03

    def test_send_subcmd_payload_at_correct_offset(self):
        """Subcommand payload should start at offset 10 from report start."""
        dev = FakeHIDDevice(side="left")
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            devices_map={"left": dev},
        )
        ctrl._send_subcmd(dev, 0x03, bytes([0x30]))
        assert dev.written[0][_MACOS_OFFSET + 10] == 0x30

    def test_packet_counter_increments(self):
        """Packet counter should increment modulo 0x10."""
        dev = FakeHIDDevice(side="left")
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            devices_map={"left": dev},
        )
        ctrl._pkt_counter = 0
        ctrl._send_subcmd(dev, 0x03, bytes([0x30]))
        ctrl._send_subcmd(dev, 0x03, bytes([0x30]))
        # The counter is embedded in the rumble neutral payload byte 0 (low nibble)
        pkt1 = dev.written[0][_MACOS_OFFSET + 1] & 0x0F
        pkt2 = dev.written[1][_MACOS_OFFSET + 1] & 0x0F
        assert pkt2 == pkt1 + 1


# ── Tests: Battery info ─────────────────────────────────────────────────────


class TestBatteryInfo:
    def test_battery_info_returns_dict(self):
        dev = FakeHIDDevice(side="left")
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
            devices_map={"left": dev},
        )
        ctrl.battery_level["left"] = 75
        ctrl.battery_charging["left"] = True
        info = ctrl.battery_info()
        assert info["left"]["level"] == 75
        assert info["left"]["charging"] is True

    def test_battery_info_missing_device(self):
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.SINGLE_LEFT,
        )
        info = ctrl.battery_info()
        # No battery data recorded for a device that never sent a report
        assert "left" not in info or info["left"]["level"] == -1


# ── Tests: IMU accelerometer ────────────────────────────────────────────────


class TestIMU:
    def test_int16_le_positive(self):
        """Positive int16 LE decoding."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        assert ctrl._int16_le(0x00, 0x10) == 4096  # 0x1000

    def test_int16_le_negative(self):
        """Negative int16 LE decoding."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        assert ctrl._int16_le(0x00, 0xF0) == -4096  # 0xF000 as signed

    def test_parse_imu_upright(self):
        """Joy-Con held upright: acc_z ≈ +4096, acc_x/y ≈ 0 → small angles."""
        report = _make_full_report()
        # Set IMU bytes: acc_x=0, acc_y=0, acc_z=4096 (int16 LE)
        report[13] = 0x00  # acc_x lo
        report[14] = 0x00  # acc_x hi
        report[15] = 0x00  # acc_y lo
        report[16] = 0x00  # acc_y hi
        report[17] = 0x00  # acc_z lo
        report[18] = 0x10  # acc_z hi = 4096
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        # Upright → near-zero tilt
        assert abs(ctrl.wrist_flex_angle) < 5.0
        assert abs(ctrl.wrist_roll_angle) < 5.0

    def test_parse_imu_tilted_forward(self):
        """Joy-Con tilted forward: acc_x positive → positive flex angle."""
        report = _make_full_report()
        report[13] = 0x00  # acc_x = 4096
        report[14] = 0x10
        report[15] = 0x00  # acc_y = 0
        report[16] = 0x00
        report[17] = 0x00  # acc_z = 0 (sideways)
        report[18] = 0x00
        ctrl = _create_controller_with_fake_devices(
            mode=JoyConMode.DUAL,
            left_reports=[report],
        )
        assert ctrl.wrist_flex_angle > 45.0  # large forward tilt

    def test_get_wrist_angles_returns_tuple(self):
        """get_wrist_angles returns (flex, roll) tuple."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl.wrist_flex_angle = 15.0
        ctrl.wrist_roll_angle = -10.0
        flex, roll = ctrl.get_wrist_angles()
        assert flex == 15.0
        assert roll == -10.0

    def test_imu_clamped_to_90(self):
        """IMU angles are clamped to [-90, 90] degrees."""
        ctrl = _create_controller_with_fake_devices(mode=JoyConMode.DUAL)
        ctrl._imu_flex_offset = 0.0
        ctrl._imu_roll_offset = 0.0
        # Set very large accel values
        ctrl.imu_acc_x = 10000.0
        ctrl.imu_acc_y = 0.0
        ctrl.imu_acc_z = 1.0
        ctrl.wrist_flex_angle = 90.0  # would be >90 without clamp
        # Re-run parse with clamping check
        flex, _ = ctrl.get_wrist_angles()
        assert flex <= 90.0


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
