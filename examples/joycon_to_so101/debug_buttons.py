#!/usr/bin/env python

"""Joy-Con 按键诊断脚本 — 实时显示所有按键名称和摇杆/IMU 数值。

用法:
  uv run python examples/joycon_to_so101/debug_buttons.py
"""

import signal
import sys
import time

from lerobot.teleoperators.joycon.joycon_utils import JoyConHIDController
from lerobot.teleoperators.joycon.configuration_joycon import JoyConMode


def main():
    shutdown = False

    def handle_signal(sig, frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)

    ctrl = JoyConHIDController(mode=JoyConMode.AUTO, deadzone=0.10)
    ctrl.start()

    if not ctrl.devices:
        print("❌ 未检测到 Joy-Con，请确认蓝牙已配对。")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  Joy-Con 按键诊断工具")
    print("=" * 60)
    print(f"  模式: {ctrl.mode.value}")
    print(f"  设备: {list(ctrl.devices.keys())}")
    print(f"  摇杆可用: {ctrl.stick_available}")
    print()
    print("  按下任意按键查看名称，Ctrl+C 退出")
    print("=" * 60)
    print()

    prev_buttons: dict[str, bool] = {}
    frame = 0

    while not shutdown:
        ctrl.update()
        frame += 1

        buttons = ctrl.buttons
        # 找出变化的按键
        changed = []
        for key in set(list(buttons.keys()) + list(prev_buttons.keys())):
            old = prev_buttons.get(key, False)
            new = buttons.get(key, False)
            if old != new:
                state = "🔴 按下" if new else "⚪ 松开"
                changed.append(f"  {state}  {key}")

        if changed:
            print(f"── 帧 {frame} ──")
            for line in changed:
                print(line)

            # 显示摇杆和 IMU 数值
            print(f"  左摇杆: X={ctrl.left_x:+.3f}  Y={ctrl.left_y:+.3f}")
            print(f"  右摇杆: X={ctrl.right_x:+.3f}  Y={ctrl.right_y:+.3f}")
            print(f"  IMU pitch={ctrl.imu_pitch:+.1f}°  roll={ctrl.imu_roll:+.1f}°")
            print(f"  gyro  Δpitch={ctrl.gyro_pitch_delta:+.2f}°  Δroll={ctrl.gyro_roll_delta:+.2f}°")
            print(f"  速度档位: {ctrl.speed_multiplier:.1f}x  精细: {'ON' if ctrl.fine_tune else 'OFF'}")
            print()

        prev_buttons = dict(buttons)
        time.sleep(0.02)

    print("\n退出。")
    ctrl.stop()


if __name__ == "__main__":
    main()
