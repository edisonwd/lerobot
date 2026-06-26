#!/usr/bin/env python
"""Joy-Con simple mode button diagnostic tool.

Press each physical button on the Joy-Con one at a time.
The script will show which logical name the code assigns to it.

Usage:
    uv run python examples/joycon_to_so101/diagnose_buttons.py
"""

from lerobot.teleoperators.joycon.joycon_utils import JoyConHIDController, JoyConMode

print("Connecting to Joy-Con (single right)...")
ctrl = JoyConHIDController(mode=JoyConMode.SINGLE_RIGHT, deadzone=0.15)
ctrl.start()

if not ctrl.devices:
    print("ERROR: No Joy-Con connected.")
    exit(1)

print("\n" + "=" * 50)
print("  Joy-Con Button Diagnostic")
print("=" * 50)
print("  逐个按下右 Joy-Con 的物理按键")
print("  观察逻辑名称是否与物理按键匹配")
print("  Ctrl+C 退出")
print("=" * 50 + "\n")

prev_buttons = {}

try:
    while True:
        ctrl.update()
        # Find newly pressed buttons
        for name, pressed in ctrl.buttons.items():
            if pressed and not prev_buttons.get(name, False):
                print(f"  物理按键 → 逻辑名称: '{name}'")
        prev_buttons = dict(ctrl.buttons)
except KeyboardInterrupt:
    pass

ctrl.stop()
print("\nDone.")
