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
            "sl_right": buttons.get("sl_right", False),
            "sr_right": buttons.get("sr_right", False),
            "sl_left": buttons.get("sl_left", False),
            "sr_left": buttons.get("sr_left", False),
            "dpad_up": buttons.get("up", False) or buttons.get("dpad_up", False),
            "dpad_down": buttons.get("down", False) or buttons.get("dpad_down", False),
            "dpad_left": buttons.get("left", False) or buttons.get("dpad_left", False),
            "dpad_right": buttons.get("right", False) or buttons.get("dpad_right", False),
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
