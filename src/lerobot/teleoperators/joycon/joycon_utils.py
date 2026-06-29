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

"""Nintendo Switch Joy-Con HID protocol and controller implementation.

This module handles the low-level HID communication with Joy-Con controllers,
including initialization, report parsing, and input mapping to motion deltas.

Joy-Con controllers use a Nintendo-specific HID protocol that requires an
explicit handshake to enable full input reports (which include analog stick
data). In the default "simple" mode (report 0x3F), only buttons are reported.
"""

import contextlib
import logging
import math
import time
from enum import IntEnum

from lerobot.utils.import_utils import require_package

from ..gamepad.gamepad_utils import InputController
from ..utils import TeleopEvents
from .configuration_joycon import JoyConMode

logger = logging.getLogger(__name__)


# ── Gripper action constants (shared with gamepad) ──────────────────────────


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}


# ── Nintendo HID identifiers ────────────────────────────────────────────────

NINTENDO_VID = 0x057E
JOYCON_LEFT_PID = 0x2006
JOYCON_RIGHT_PID = 0x2007
JOYCON_COMBINED_PID = 0x200E

SUPPORTED_PIDS = {JOYCON_LEFT_PID, JOYCON_RIGHT_PID, JOYCON_COMBINED_PID}

# Map PID → side label
_PID_TO_SIDE = {
    JOYCON_LEFT_PID: "left",
    JOYCON_RIGHT_PID: "right",
    JOYCON_COMBINED_PID: "combined",
}


# ── Output report constants (host → Joy-Con) ────────────────────────────────

_OUTPUT_REPORT_SUBCMD = 0x01  # report ID for subcommand writes
_SUBCMD_SET_REPORT_MODE = 0x03  # switch input report mode
_SUBCMD_PLAYER_LIGHTS = 0x30  # set player LEDs

# Rumble neutral payload (8 bytes) — must be sent with every subcommand
# or the Joy-Con will vibrate unexpectedly.
_RUMBLE_NEUTRAL = bytes([0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40])

# ── Input report IDs (Joy-Con → host) ───────────────────────────────────────

FULL_INPUT_REPORT_ID = 0x30  # 362-byte report with stick data
SIMPLE_INPUT_REPORT_ID = 0x3F  # 12-byte button-only report (default)
SUBCMD_ACK_REPORT_ID = 0x21  # subcommand acknowledgement


# ── Stick decoding constants ────────────────────────────────────────────────

_STICK_CENTER_DEFAULT = 2048  # nominal center for 12-bit stick values
_STICK_MAX = 4095


def _decode_stick(b0: int, b1: int, b2: int) -> tuple[int, int]:
    """Decode a Joy-Con 12-bit packed stick value into (x, y) raw integers.

    Each stick axis is encoded across 3 bytes:
        x_raw = b0 | ((b1 & 0x0F) << 8)   → 12 bits
        y_raw = (b1 >> 4) | (b2 << 4)      → 12 bits
    """
    x = (b0 & 0xFF) | ((b1 & 0x0F) << 8)
    y = ((b1 >> 4) & 0x0F) | ((b2 & 0xFF) << 4)
    return x, y


def _parse_buttons_right(byte0: int) -> dict[str, bool]:
    """Parse right Joy-Con button byte."""
    return {
        "y": bool(byte0 & 0x01),
        "x": bool(byte0 & 0x02),
        "b": bool(byte0 & 0x04),
        "a": bool(byte0 & 0x08),
        "sr_right": bool(byte0 & 0x10),
        "sl_right": bool(byte0 & 0x20),
        "r": bool(byte0 & 0x40),
        "zr": bool(byte0 & 0x80),
    }


def _parse_buttons_shared(byte1: int) -> dict[str, bool]:
    """Parse shared button byte."""
    return {
        "plus": bool(byte1 & 0x02),
        "minus": bool(byte1 & 0x01),
        "home": bool(byte1 & 0x10),
        "capture": bool(byte1 & 0x20),
        "l_stick_press": bool(byte1 & 0x80),
    }


def _parse_buttons_left(byte2: int) -> dict[str, bool]:
    """Parse left Joy-Con button byte."""
    return {
        "down": bool(byte2 & 0x01),
        "up": bool(byte2 & 0x02),
        "right": bool(byte2 & 0x04),
        "left": bool(byte2 & 0x08),
        "sr_left": bool(byte2 & 0x10),
        "sl_left": bool(byte2 & 0x20),
        "l": bool(byte2 & 0x40),
        "zl": bool(byte2 & 0x80),
    }


class JoyConHIDController(InputController):
    """Joy-Con HID controller using hidapi.

    Handles the Nintendo-specific HID handshake, full report mode switching,
    stick center calibration, and input parsing for Joy-Con controllers.
    """

    def __init__(
        self,
        mode: JoyConMode = JoyConMode.AUTO,
        deadzone: float = 0.15,
        x_step_size: float = 1.0,
        y_step_size: float = 1.0,
        z_step_size: float = 1.0,
        rotation_step: float = 1.0,
        min_speed: float = 0.2,
        max_speed: float = 2.0,
        speed_step: float = 0.2,
        fine_tune_multiplier: float = 0.5,
    ):
        require_package("hidapi", extra="joycon", import_name="hid")
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.mode = mode
        self.deadzone = deadzone

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

        # Edge detection for toggle buttons (previous frame state)
        self._prev_dpad_up = False
        self._prev_dpad_down = False
        self._prev_l_stick_press = False

        # Emergency stop flag
        self.emergency_stop: bool = False

        # Connected HID devices: side ("left"/"right") → hid.device
        self.devices: dict = {}
        self.device_info: dict[str, dict] = {}

        # Stick raw values and calibrated centers
        self.left_x_raw = _STICK_CENTER_DEFAULT
        self.left_y_raw = _STICK_CENTER_DEFAULT
        self.right_x_raw = _STICK_CENTER_DEFAULT
        self.right_y_raw = _STICK_CENTER_DEFAULT
        self.left_center = (_STICK_CENTER_DEFAULT, _STICK_CENTER_DEFAULT)
        self.right_center = (_STICK_CENTER_DEFAULT, _STICK_CENTER_DEFAULT)

        # Normalized stick values (after deadzone, clamped to [-1, 1])
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # Button states (logical names → bool)
        self.buttons: dict[str, bool] = {}

        # State flags
        self.stick_available = False
        self._pkt_counter = 0

        # Battery tracking
        self.battery_level: dict[str, int] = {}  # side → percentage
        self.battery_charging: dict[str, bool] = {}
        self._battery_warned: dict[str, bool] = {}

        # IMU accelerometer (for wrist tilt control)
        # Raw accel values from the left Joy-Con (or whichever is available)
        self.imu_acc_x: float = 0.0
        self.imu_acc_y: float = 0.0
        self.imu_acc_z: float = 0.0
        # Computed tilt angles in degrees
        self.wrist_flex_angle: float = 0.0  # forward/backward tilt
        self.wrist_roll_angle: float = 0.0  # rotation around grip axis
        # IMU calibration offsets (set on connect, when Joy-Con is held upright)
        self._imu_flex_offset: float = 0.0
        self._imu_roll_offset: float = 0.0
        # EMA smoothing
        self._imu_alpha: float = 0.3  # smoothing factor (lower = smoother)

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

        # Mode switch retry counter (for background retry in update())
        self._mode_switch_retries: int = 0
        self._max_mode_switch_retries: int = 10

        # Reconnection
        self._last_reconnect_attempt = 0.0
        self._reconnect_interval = 1.0  # seconds

    # ── InputController interface ────────────────────────────────────────────

    def start(self):
        """Enumerate, connect, and initialize Joy-Con device(s)."""

        found = self._enumerate_joycons()
        if not any(found.values()):
            logger.error(
                "No Joy-Con detected. Make sure a Joy-Con is paired via Bluetooth "
                "and that you have HID device permissions."
            )
            self.running = False
            return

        # Resolve mode
        resolved = self._resolve_mode(found)
        self.mode = resolved
        logger.info("Joy-Con mode resolved to: %s", self.mode.value)

        # Open device(s) based on resolved mode
        sides_to_open = self._sides_for_mode(resolved, found)
        for side, info_list in sides_to_open.items():
            if not info_list:
                continue
            info = info_list[0]  # take first if multiple
            dev = self._open_device(info)
            if dev is not None:
                self.devices[side] = dev
                self.device_info[side] = info
                self._initialize_device(side, dev)

        if not self.devices:
            logger.error("Failed to open any Joy-Con device.")
            self.running = False
            return

        # Print control summary
        self._print_controls()

    def stop(self):
        """Close all HID device connections."""
        for side, dev in self.devices.items():
            try:
                dev.close()
                logger.info("Joy-Con (%s) disconnected.", side)
            except Exception:
                pass
        self.devices.clear()
        self.device_info.clear()
        self.stick_available = False

    def update(self):
        """Read all connected devices and update state.

        Also attempts reconnection for any disconnected devices.
        """
        # Read from all open devices
        sides_to_remove = []
        for side, dev in list(self.devices.items()):
            try:
                # Read multiple times to flush stale buffers (same pattern as gamepad HID)
                for _ in range(10):
                    data = dev.read(362)
                    if data:
                        self._process_report(side, data)
            except OSError as e:
                logger.warning("Joy-Con (%s) read error: %s — marking disconnected.", side, e)
                with contextlib.suppress(Exception):
                    dev.close()
                sides_to_remove.append(side)

        for side in sides_to_remove:
            self.devices.pop(side, None)
            self.device_info.pop(side, None)

        # Attempt reconnection if we've lost devices (only when still active)
        if self.running and (sides_to_remove or not self.devices):
            self._try_reconnect()

        # Background retry: try to enable full report mode if still in simple mode
        if not self.stick_available and self.running and self.devices:
            if self._mode_switch_retries < self._max_mode_switch_retries:
                self._mode_switch_retries += 1
                for side, dev in self.devices.items():
                    ok = self._send_subcmd(dev, _SUBCMD_SET_REPORT_MODE, bytes([FULL_INPUT_REPORT_ID]))
                    if ok:
                        # Check if 0x30 reports start arriving
                        for _ in range(5):
                            try:
                                data = dev.read(362)
                            except OSError:
                                break
                            if data and len(data) > 0 and data[0] == FULL_INPUT_REPORT_ID:
                                self.stick_available = True
                                logger.info(
                                    "Joy-Con (%s): full report mode enabled on retry %d.",
                                    side,
                                    self._mode_switch_retries,
                                )
                                self._calibrate_center(side, dev)
                                break
                            time.sleep(0.01)
                    if self.stick_available:
                        break

        # Update intervention flag: active when any stick is outside deadzone
        any_stick_active = (
            abs(self.left_x) > 0.0
            or abs(self.left_y) > 0.0
            or abs(self.right_x) > 0.0
            or abs(self.right_y) > 0.0
        )
        self.intervention_flag = any_stick_active

    def get_deltas(self) -> tuple[float, float, float]:
        """Get position movement deltas based on the current Joy-Con mode.

        Returns:
            (delta_x, delta_y, delta_z) with speed and fine-tune applied.
        """
        if not self.stick_available:
            return 0.0, 0.0, 0.0

        scale = self._effective_scale()

        if self.mode == JoyConMode.SINGLE_LEFT:
            nx = self.left_x
            ny = self.left_y
            dz = self._lz_z_delta()
        elif self.mode == JoyConMode.SINGLE_RIGHT:
            nx = self.right_x
            ny = self.right_y
            dz = self._rz_z_delta()
        else:  # DUAL
            # Left stick → XY plane; L/ZL → Z axis
            nx = self.left_x
            ny = self.left_y
            dz = self._lz_z_delta()
            dx = -ny * self.y_step_size * scale
            dy = -nx * self.x_step_size * scale
            dz = dz * scale
            return dx, dy, dz

        # Single Joy-Con: stick is held sideways, so X→world Y, Y→world X
        dx = -ny * self.y_step_size * scale
        dy = -nx * self.x_step_size * scale
        dz = dz * scale
        return dx, dy, dz

    def get_left_stick(self) -> tuple[float, float]:
        """Get raw normalized left stick values (after deadzone).

        Returns:
            (stick_x, stick_y) in range [-1.0, 1.0], with speed/fine-tune applied.
        """
        if not self.stick_available:
            return 0.0, 0.0
        scale = self._effective_scale()
        return self.left_x * scale, self.left_y * scale

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

    def get_orientation_deltas(self) -> tuple[float, float, float]:
        """Get orientation rotation deltas from right Joy-Con.

        In dual mode: right stick → Yaw (X axis) / Pitch (Y axis), R/ZR → Roll.
        In single modes: returns (0, 0, 0).

        Returns:
            (delta_wx, delta_wy, delta_wz) with speed and fine-tune applied.
        """
        if not self.stick_available or self.mode != JoyConMode.DUAL:
            return 0.0, 0.0, 0.0

        scale = self._effective_scale()

        # Right stick X → Yaw (rotation around Z axis)
        yaw = self.right_x * self.rotation_step * scale
        # Right stick Y → Pitch (rotation around Y axis)
        pitch = -self.right_y * self.rotation_step * scale
        # R/ZR → Roll (rotation around X axis)
        roll = 0.0
        if self.buttons.get("r") and not self.buttons.get("zr"):
            roll = self.rotation_step * scale
        elif self.buttons.get("zr") and not self.buttons.get("r"):
            roll = -self.rotation_step * scale

        return yaw, pitch, roll

    def _effective_scale(self) -> float:
        """Compute effective scale = speed_multiplier × fine_tune_multiplier (if active)."""
        scale = self.speed_multiplier
        if self.fine_tune:
            scale *= self.fine_tune_multiplier
        return scale

    # ── HID enumeration and mode resolution ──────────────────────────────────

    def _enumerate_joycons(self) -> dict[str, list[dict]]:
        """Find all connected Joy-Con devices via HID.

        Returns:
            Dict with keys "left", "right", "combined" → list of hid device info dicts.
        """
        import hid

        result: dict[str, list[dict]] = {"left": [], "right": [], "combined": []}
        for dev_info in hid.enumerate():
            vid = dev_info.get("vendor_id", 0)
            pid = dev_info.get("product_id", 0)
            if vid != NINTENDO_VID or pid not in SUPPORTED_PIDS:
                continue
            side = _PID_TO_SIDE.get(pid, "unknown")
            if side in result:
                result[side].append(dev_info)
                logger.info(
                    "Found Joy-Con (%s): %s [%04x:%04x]",
                    side,
                    dev_info.get("product_string", "?"),
                    vid,
                    pid,
                )
        return result

    def _resolve_mode(self, found: dict[str, list[dict]]) -> JoyConMode:
        """Resolve AUTO mode to a concrete mode, or validate a user-pinned mode."""
        has_left = bool(found["left"])
        has_right = bool(found["right"])
        has_combined = bool(found["combined"])

        if self.mode != JoyConMode.AUTO:
            # User pinned a mode — validate it
            if self.mode == JoyConMode.DUAL:
                if not (has_left and has_right) and not has_combined:
                    logger.warning(
                        "Dual mode requested but only %s found. Falling back to AUTO.",
                        "left" if has_left else "right" if has_right else "nothing",
                    )
                else:
                    return JoyConMode.DUAL
            elif self.mode == JoyConMode.SINGLE_LEFT and not has_left:
                logger.warning("Single-left requested but no left Joy-Con found. Falling back to AUTO.")
            elif self.mode == JoyConMode.SINGLE_RIGHT and not has_right:
                logger.warning("Single-right requested but no right Joy-Con found. Falling back to AUTO.")
            else:
                return self.mode

        # AUTO resolution
        if has_combined:
            return JoyConMode.DUAL
        if has_left and has_right:
            return JoyConMode.DUAL
        if has_left:
            return JoyConMode.SINGLE_LEFT
        if has_right:
            return JoyConMode.SINGLE_RIGHT

        raise ConnectionError("No Joy-Con detected.")

    def _sides_for_mode(self, mode: JoyConMode, found: dict[str, list[dict]]) -> dict[str, list[dict]]:
        """Select which device(s) to open based on the resolved mode."""
        if mode == JoyConMode.DUAL:
            if found["combined"]:
                return {"combined": found["combined"]}
            return {"left": found["left"], "right": found["right"]}
        elif mode == JoyConMode.SINGLE_LEFT:
            return {"left": found["left"]}
        elif mode == JoyConMode.SINGLE_RIGHT:
            return {"right": found["right"]}
        return {}

    # ── Device initialization ────────────────────────────────────────────────

    def _open_device(self, info: dict):
        """Open an HID device from its info dict."""
        import hid

        try:
            dev = hid.device()
            dev.open_path(info["path"])
            dev.set_nonblocking(1)
            product = info.get("product_string", "?")
            logger.info("Opened Joy-Con: %s at %s", product, info["path"])
            return dev
        except OSError as e:
            logger.error(
                "Failed to open Joy-Con at %s: %s. "
                "You may need to run with elevated permissions or configure udev rules.",
                info["path"],
                e,
            )
            return None

    def _initialize_device(self, side: str, dev) -> None:
        """Run the Joy-Con initialization handshake on an open device."""
        # Flush any stale reports from the HID buffer
        for _ in range(20):
            try:
                dev.read(362)
            except OSError:
                break

        # 1. Set player LED (also wakes the device)
        self._send_subcmd(dev, _SUBCMD_PLAYER_LIGHTS, bytes([0x01]))
        time.sleep(0.3)

        # Read and discard any reports generated during LED setup
        for _ in range(10):
            try:
                dev.read(362)
            except OSError:
                break

        # 2. Switch to full report mode (0x30) for stick data.
        # On some BT stacks (especially macOS), the subcmd ACK (0x21) may not
        # arrive reliably. We also probe for full reports (0x30) as a secondary
        # confirmation signal.
        for attempt in range(8):
            self._send_subcmd(dev, _SUBCMD_SET_REPORT_MODE, bytes([FULL_INPUT_REPORT_ID]))

            # Wait for ACK or 0x30 reports with longer timeout
            if self._wait_for_ack(dev, _SUBCMD_SET_REPORT_MODE, timeout_s=2.0):
                self.stick_available = True
                logger.info("Joy-Con (%s): full report mode enabled (ACK received).", side)
                break

            # Secondary check: see if full reports (0x30) are arriving
            found_030 = False
            for _ in range(20):
                try:
                    data = dev.read(362)
                except OSError:
                    break
                if data and len(data) > 0 and data[0] == FULL_INPUT_REPORT_ID:
                    found_030 = True
                    break
                time.sleep(0.01)
            if found_030:
                self.stick_available = True
                logger.info("Joy-Con (%s): full report mode enabled (0x30 detected).", side)
                break

            logger.info("Joy-Con (%s): mode switch attempt %d/8.", side, attempt + 1)
            time.sleep(0.2)

        if not self.stick_available:
            logger.warning(
                "Joy-Con (%s): could not enable full report mode after 8 attempts. "
                "Falling back to simple mode (buttons only, no stick data). "
                "Try re-pairing the Joy-Con via Bluetooth.",
                side,
            )

        # 3. Calibrate stick center from initial reports
        if self.stick_available:
            self._calibrate_center(side, dev)

    def _send_subcmd(self, dev, subcmd_id: int, payload: bytes) -> bool:
        """Send a subcommand output report to a Joy-Con.

        Returns True if write succeeded, False otherwise.
        """
        pkt_id = self._next_pkt_id()
        rumble = bytearray(_RUMBLE_NEUTRAL)
        rumble[0] = (rumble[0] & 0xF0) | pkt_id

        report = bytearray(49)
        report[0] = _OUTPUT_REPORT_SUBCMD
        report[1:9] = rumble
        report[9] = subcmd_id
        plen = min(len(payload), 39)
        report[10 : 10 + plen] = payload[:plen]

        # hidapi expects [report_id, data...] for numbered output reports.
        # Report ID 0x01 is at report[0]. No extra prefix needed on any platform.
        write_buf = bytes(report)

        try:
            written = dev.write(list(write_buf))
            expected = len(write_buf)
            if written != expected:
                logger.warning(
                    "Subcmd 0x%02x: wrote %d/%d bytes", subcmd_id, written, expected
                )
                return False
            return True
        except OSError as e:
            logger.warning("Failed to send subcmd 0x%02x: %s", subcmd_id, e)
            return False

    def _wait_for_ack(self, dev, expected_subcmd: int, timeout_s: float = 1.0) -> bool:
        """Wait for a subcommand ACK report matching the expected subcommand ID."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                data = dev.read(362)
            except OSError:
                time.sleep(0.01)
                continue
            if data and len(data) > 14 and data[0] == SUBCMD_ACK_REPORT_ID and data[14] == expected_subcmd:
                return True
            time.sleep(0.01)
        return False

    def _calibrate_center(self, side: str, dev, num_samples: int = 20) -> None:
        """Read several reports and average the stick values to find the center."""
        xs, ys = [], []
        collected = 0
        deadline = time.time() + 2.0  # 2 second timeout

        while collected < num_samples and time.time() < deadline:
            try:
                data = dev.read(362)
            except OSError:
                time.sleep(0.01)
                continue
            if data and len(data) >= 12 and data[0] == FULL_INPUT_REPORT_ID:
                # Determine which stick offset to read based on side
                if side in ("left", "combined"):
                    x, y = _decode_stick(data[6], data[7], data[8])
                    xs.append(x)
                    ys.append(y)
                    collected += 1
                elif side == "right":
                    x, y = _decode_stick(data[9], data[10], data[11])
                    xs.append(x)
                    ys.append(y)
                    collected += 1
                elif side == "combined":
                    # Read both sticks
                    lx, ly = _decode_stick(data[6], data[7], data[8])
                    rx, ry = _decode_stick(data[9], data[10], data[11])
                    xs.extend([lx, rx])
                    ys.extend([ly, ry])
                    collected += 2
            time.sleep(0.005)

        if xs:
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))
            if side in ("left", "combined"):
                self.left_center = (cx, cy)
                logger.info("Joy-Con (%s): left stick center calibrated to (%d, %d).", side, cx, cy)
            if side in ("right", "combined"):
                self.right_center = (cx, cy)
                logger.info("Joy-Con (%s): right stick center calibrated to (%d, %d).", side, cx, cy)
        else:
            logger.warning("Joy-Con (%s): stick calibration timed out, using defaults.", side)

    def _next_pkt_id(self) -> int:
        """Get the next packet counter value (0x0..0xF)."""
        val = self._pkt_counter
        self._pkt_counter = (self._pkt_counter + 1) & 0x0F
        return val

    # ── Report processing ────────────────────────────────────────────────────

    def _process_report(self, side: str, data: list[int] | bytes) -> None:
        """Process a single HID input report from a Joy-Con."""
        if not data:
            return

        report_id = data[0]

        if report_id == FULL_INPUT_REPORT_ID and len(data) >= 12:
            self._parse_full_report(side, data)
        elif report_id == SIMPLE_INPUT_REPORT_ID and len(data) >= 6:
            self._parse_simple_report(side, data)
        # 0x21 (subcmd ACK) is only relevant during init; ignore at runtime.

    def _parse_full_report(self, side: str, data) -> None:
        """Parse a full 0x30 input report: buttons + sticks + battery."""
        # Battery info (byte 2, bits 7-4 = level, bit 0 = charging)
        self._parse_battery(side, data[2])

        # Button bytes (bytes 3, 4, 5)
        right_btns = _parse_buttons_right(data[3])
        shared_btns = _parse_buttons_shared(data[4])
        left_btns = _parse_buttons_left(data[5])

        # Merge all buttons into a flat dict
        self.buttons.update(right_btns)
        self.buttons.update(shared_btns)
        self.buttons.update(left_btns)

        # Left stick (bytes 6-8)
        if side in ("left", "combined"):
            self.left_x_raw, self.left_y_raw = _decode_stick(data[6], data[7], data[8])
            self.left_x = self._normalize_axis(self.left_x_raw, self.left_center[0])
            self.left_y = self._normalize_axis(self.left_y_raw, self.left_center[1])

        # Right stick (bytes 9-11)
        if side in ("right", "combined"):
            self.right_x_raw, self.right_y_raw = _decode_stick(data[9], data[10], data[11])
            self.right_x = self._normalize_axis(self.right_x_raw, self.right_center[0])
            self.right_y = self._normalize_axis(self.right_y_raw, self.right_center[1])

        # For combined device, parse both sticks
        if side == "combined" and len(data) >= 12:
            self.left_x_raw, self.left_y_raw = _decode_stick(data[6], data[7], data[8])
            self.left_x = self._normalize_axis(self.left_x_raw, self.left_center[0])
            self.left_y = self._normalize_axis(self.left_y_raw, self.left_center[1])
            self.right_x_raw, self.right_y_raw = _decode_stick(data[9], data[10], data[11])
            self.right_x = self._normalize_axis(self.right_x_raw, self.right_center[0])
            self.right_y = self._normalize_axis(self.right_y_raw, self.right_center[1])

        # Map buttons to gripper and episode commands
        self._map_buttons_to_actions()

        # Parse IMU accelerometer data (bytes 13-24, first sample)
        if len(data) >= 25:
            self._parse_imu(data)

    def _parse_simple_report(self, side: str, data) -> None:
        """Parse a simple 0x3F report (buttons only, limited stick data).

        Simple mode 0x3F layout for a single Joy-Con over Bluetooth:
            Byte 0: Report ID (0x3F)
            Byte 1: Side-specific button byte (same bit layout as full mode)
                - Right Joy-Con: same as full mode byte 3
                - Left Joy-Con: same as full mode byte 5
            Byte 2: Shared buttons (plus, minus, home, capture, l_stick_press)
            Bytes 3+: Stick data (side-specific, 2 bytes per axis)

        When two Joy-Cons are connected separately, each sends its own 0x3F.
        When combined (grip), byte 1=right, byte 2=left, byte 3=shared.
        """
        if len(data) < 3:
            return

        if side == "right":
            # Single right Joy-Con: byte 1 = right buttons, byte 2 = shared
            right_btns = _parse_buttons_right(data[1])
            shared_btns = _parse_buttons_shared(data[2])
            self.buttons.update(right_btns)
            self.buttons.update(shared_btns)
        elif side == "left":
            # Single left Joy-Con: byte 1 = left buttons, byte 2 = shared
            left_btns = _parse_buttons_left(data[1])
            shared_btns = _parse_buttons_shared(data[2])
            self.buttons.update(left_btns)
            self.buttons.update(shared_btns)
        elif side == "combined":
            # Combined: byte 1 = right, byte 2 = left, byte 3 = shared
            if len(data) >= 4:
                right_btns = _parse_buttons_right(data[1])
                left_btns = _parse_buttons_left(data[2])
                shared_btns = _parse_buttons_shared(data[3])
                self.buttons.update(right_btns)
                self.buttons.update(left_btns)
                self.buttons.update(shared_btns)

        # Map buttons to actions (gripper, episode controls, etc.)
        self._map_buttons_to_actions()

    def _parse_battery(self, side: str, info_byte: int) -> None:
        """Parse battery level from the connection info byte.

        Byte 2 layout:
            bits 7-5: battery level (4=full, 3=medium, 2=low, 1=critical, 0=empty)
            bit 4:    charging
            bits 3-0: connection info
        """
        level_nibble = (info_byte >> 5) & 0x07
        charging = bool(info_byte & 0x10)

        # Map level value to percentage
        # 4=full(100%), 3=medium(75%), 2=low(50%), 1=critical(25%), 0=empty(0%)
        level_map = {4: 100, 3: 75, 2: 50, 1: 25, 0: 0}
        pct = level_map.get(level_nibble, 50)

        self.battery_level[side] = pct
        self.battery_charging[side] = charging

        # One-shot warnings
        if pct <= 15 and not self._battery_warned.get(f"{side}_low"):
            logger.warning("Joy-Con (%s): battery low (%d%%).", side, pct)
            self._battery_warned[f"{side}_low"] = True
        if pct <= 5 and not self._battery_warned.get(f"{side}_critical"):
            logger.error("Joy-Con (%s): battery critical (%d%%)!", side, pct)
            self._battery_warned[f"{side}_critical"] = True

    # ── Axis normalization ───────────────────────────────────────────────────

    def _normalize_axis(self, raw: int, center: int) -> float:
        """Normalize a raw 12-bit stick value to [-1.0, 1.0] with deadzone."""
        offset = raw - center
        normalized = offset / (_STICK_MAX - center)  # scale relative to half-range

        # Apply deadzone
        if abs(normalized) < self.deadzone:
            return 0.0

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, normalized))

    # ── Button → action mapping ──────────────────────────────────────────────

    def _map_buttons_to_actions(self) -> None:
        """Map Joy-Con button states to InputController action flags.

        Also handles speed adjustment (D-pad), fine-tune toggle (stick press),
        and emergency stop (Home button).
        """
        btns = self.buttons

        # ── Speed control (edge-triggered on D-pad) ─────────────────────
        self._handle_speed_buttons(btns)

        # ── Fine-tune toggle (edge-triggered on stick press) ────────────
        self._handle_fine_tune_toggle(btns)

        # ── Emergency stop (Home button on right Joy-Con) ───────────────
        if btns.get("home"):
            self.emergency_stop = True

        # ── Gripper control ─────────────────────────────────────────────
        if self.mode == JoyConMode.DUAL:
            # A = close gripper, B = open gripper
            self.close_gripper_command = btns.get("a", False)
            self.open_gripper_command = btns.get("b", False)
        elif self.mode == JoyConMode.SINGLE_LEFT:
            self.close_gripper_command = btns.get("l", False) or btns.get("zl", False)
            self.open_gripper_command = False
        elif self.mode == JoyConMode.SINGLE_RIGHT:
            self.open_gripper_command = btns.get("r", False) or btns.get("zr", False)
            self.close_gripper_command = False

        # ── Episode end events ──────────────────────────────────────────
        if btns.get("plus"):
            self.episode_end_status = TeleopEvents.SUCCESS
        elif btns.get("minus"):
            self.episode_end_status = TeleopEvents.FAILURE
        else:
            self.episode_end_status = None

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

    def _handle_fine_tune_toggle(self, btns: dict[str, bool]) -> None:
        """Toggle fine-tune mode via stick press (edge-triggered)."""
        stick_press = btns.get("l_stick_press", False)

        if stick_press and not self._prev_l_stick_press:
            self.fine_tune = not self.fine_tune
            state = "ON" if self.fine_tune else "OFF"
            logger.info("Fine-tune: %s", state)

        self._prev_l_stick_press = stick_press

    # ── Z-axis helpers ───────────────────────────────────────────────────────

    def _lz_z_delta(self) -> float:
        """Get Z-axis delta from L/ZL buttons (dual mode: L=up, ZL=down)."""
        l_btn = self.buttons.get("l", False)
        zl_btn = self.buttons.get("zl", False)
        if l_btn and not zl_btn:
            return self.z_step_size
        elif zl_btn and not l_btn:
            return -self.z_step_size
        return 0.0

    def _rz_z_delta(self) -> float:
        """Get Z-axis delta from R/ZR buttons (single-right mode: R=up, ZR=down)."""
        r_btn = self.buttons.get("r", False)
        zr_btn = self.buttons.get("zr", False)
        if r_btn and not zr_btn:
            return self.z_step_size
        elif zr_btn and not r_btn:
            return -self.z_step_size
        return 0.0

    # ── IMU: accelerometer + gyroscope (complementary filter) ────────────

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

    def _int16_le(self, lo: int, hi: int) -> int:
        """Decode a signed 16-bit little-endian value."""
        val = (hi << 8) | lo
        if val >= 0x8000:
            val -= 0x10000
        return val

    def calibrate_imu(self, num_samples: int = 30) -> None:
        """Calibrate IMU offsets by sampling the current orientation.

        Call this after connect, when the Joy-Con is held in the natural
        resting position. The sampled angles become the zero reference.
        """
        if not self.stick_available or not self.devices:
            return

        flex_sum = 0.0
        roll_sum = 0.0
        count = 0
        deadline = time.time() + 3.0

        for dev in self.devices.values():
            while count < num_samples and time.time() < deadline:
                try:
                    data = dev.read(362)
                except OSError:
                    break
                if data and len(data) >= 25 and data[0] == FULL_INPUT_REPORT_ID:
                    raw_x = self._int16_le(data[13], data[14])
                    raw_y = self._int16_le(data[15], data[16])
                    raw_z = self._int16_le(data[17], data[18])
                    if raw_z != 0:
                        flex_sum += math.degrees(math.atan2(raw_x, raw_z))
                        roll_sum += math.degrees(math.atan2(raw_y, raw_z))
                        count += 1
                time.sleep(0.01)

        if count > 0:
            self._imu_flex_offset = flex_sum / count
            self._imu_roll_offset = roll_sum / count
            logger.info(
                "IMU calibrated: flex_offset=%.1f°, roll_offset=%.1f° (%d samples)",
                self._imu_flex_offset,
                self._imu_roll_offset,
                count,
            )
        # Also reset gyro integration state
        self._gyro_pitch_angle = 0.0
        self._gyro_roll_angle = 0.0
        self._prev_imu_pitch = 0.0
        self._prev_imu_roll = 0.0

    def get_wrist_angles(self) -> tuple[float, float]:
        """Get wrist tilt angles from IMU accelerometer.

        Returns:
            (wrist_flex_degrees, wrist_roll_degrees)
            wrist_flex: forward/backward tilt (motor 4)
            wrist_roll: rotation around grip axis (motor 5)
        """
        return self.wrist_flex_angle, self.wrist_roll_angle

    # ── Reconnection ─────────────────────────────────────────────────────────

    def _try_reconnect(self) -> None:
        """Attempt to reconnect to any missing Joy-Con devices."""
        now = time.time()
        if now - self._last_reconnect_attempt < self._reconnect_interval:
            return
        self._last_reconnect_attempt = now

        found = self._enumerate_joycons()
        if not any(found.values()):
            return

        sides_needed = self._sides_for_mode(self.mode, found)
        for side, info_list in sides_needed.items():
            if side in self.devices:
                continue  # already connected
            if not info_list:
                continue
            info = info_list[0]
            dev = self._open_device(info)
            if dev is not None:
                self.devices[side] = dev
                self.device_info[side] = info
                self._initialize_device(side, dev)
                logger.info("Joy-Con (%s): reconnected successfully.", side)

    # ── Utilities ────────────────────────────────────────────────────────────

    def battery_info(self) -> dict[str, dict]:
        """Return battery status for all connected Joy-Cons."""
        result = {}
        for side in self.devices:
            result[side] = {
                "level": self.battery_level.get(side, -1),
                "charging": self.battery_charging.get(side, False),
            }
        return result

    def _print_controls(self) -> None:
        """Print the control mapping for the current mode."""
        mode_name = {
            JoyConMode.SINGLE_LEFT: "Single Joy-Con (Left)",
            JoyConMode.SINGLE_RIGHT: "Single Joy-Con (Right)",
            JoyConMode.DUAL: "Dual Joy-Con",
        }.get(self.mode, str(self.mode))

        print(f"\n{'=' * 50}")
        print(f"  Joy-Con Teleop — {mode_name}")
        print(f"{'=' * 50}")

        if self.mode == JoyConMode.DUAL:
            print("  [Left Joy-Con]")
            print("  Stick:          XY position")
            print("  L / ZL:         Z up / down")
            print("  D-pad ↑/↓:      Speed ±20%")
            print("  Stick press:    Fine-tune toggle")
            print()
            print("  [Right Joy-Con]")
            print("  Stick:          Yaw / Pitch")
            print("  R / ZR:         Roll CW / CCW")
            print("  A / B:          Close / Open gripper")
            print("  Plus (+):       End episode (SUCCESS)")
            print("  Minus (-):      End episode (FAILURE)")
            print("  Home:           Emergency stop")
        elif self.mode == JoyConMode.SINGLE_LEFT:
            print("  Stick:          XY position")
            print("  L / ZL:         Z up / down")
            print("  D-pad ↑/↓:      Speed ±20%")
            print("  Stick press:    Fine-tune toggle")
            print("  Minus (-):      End episode (FAILURE)")
        else:  # SINGLE_RIGHT
            print("  Stick:          XY position")
            print("  R / ZR:         Z up / down")
            print("  Plus (+):       End episode (SUCCESS)")

        print(f"{'=' * 50}\n")
