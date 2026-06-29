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

        # If pose is locked, return frozen targets (meta controls still processed)
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
