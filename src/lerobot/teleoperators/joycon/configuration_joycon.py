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

from dataclasses import dataclass
from enum import StrEnum

from ..config import TeleoperatorConfig


class JoyConMode(StrEnum):
    """Which Joy-Con(s) to use.

    AUTO detects whatever is paired: dual if both sides are present,
    single-left or single-right otherwise.
    """

    AUTO = "auto"
    SINGLE_LEFT = "single_left"
    SINGLE_RIGHT = "single_right"
    DUAL = "dual"


@TeleoperatorConfig.register_subclass("joycon")
@dataclass
class JoyConTeleopConfig(TeleoperatorConfig):
    """Configuration for Nintendo Switch Joy-Con teleoperator.

    Attributes:
        mode: Which Joy-Con(s) to use. AUTO picks whatever is paired.
        use_gripper: Include gripper in the action dict.
        deadzone: Analog stick deadzone (0..1). Joy-Con sticks drift more
            than typical gamepads, so the default is 0.15.
        step_size: Base scale for position delta output (mm per full deflection).
        z_step_size: Separate Z scale for vertical movement.
        rotation_step: Base scale for rotation delta output (radians per full deflection).
        min_speed: Minimum speed multiplier (20%).
        max_speed: Maximum speed multiplier (200%).
        speed_step: Speed adjustment increment per D-pad press (20%).
        fine_tune_multiplier: Step size multiplier when fine-tune mode is active (50%).
        mapping_path: Path to YAML mapping file. None uses built-in default.
    """

    mode: JoyConMode = JoyConMode.AUTO
    use_gripper: bool = True
    deadzone: float = 0.15
    step_size: float = 1.0
    z_step_size: float = 1.0
    rotation_step: float = 1.0
    min_speed: float = 0.2
    max_speed: float = 2.0
    speed_step: float = 0.2
    fine_tune_multiplier: float = 0.5
    mapping_path: str | None = None
