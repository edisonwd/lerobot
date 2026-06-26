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
