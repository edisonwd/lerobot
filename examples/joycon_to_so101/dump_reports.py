#!/usr/bin/env python
"""Dump raw Joy-Con HID reports to understand the macOS Bluetooth format.

This tool shows every byte of each report as it arrives, so we can see
exactly what the Joy-Con sends in simple mode on macOS.

Usage:
    uv run python examples/joycon_to_so101/dump_reports.py
"""

import time
import hid

NINTENDO_VID = 0x057E
JOYCON_LEFT_PID = 0x2006
JOYCON_RIGHT_PID = 0x2007

print("Searching for Joy-Con devices...")
devices = hid.enumerate(NINTENDO_VID, 0)
joycons = [d for d in devices if d["product_id"] in (JOYCON_LEFT_PID, JOYCON_RIGHT_PID)]

if not joycons:
    print("ERROR: No Joy-Con found. Pair via Bluetooth first.")
    exit(1)

for info in joycons:
    side = "LEFT" if info["product_id"] == JOYCON_LEFT_PID else "RIGHT"
    print(f"  Found: {side} Joy-Con — {info['product_string']} (path: {info['path']})")

# Open the first Joy-Con
info = joycons[0]
side = "LEFT" if info["product_id"] == JOYCON_LEFT_PID else "RIGHT"
dev = hid.device()
dev.open_path(info["path"])
dev.set_nonblocking(1)
print(f"\nOpened {side} Joy-Con. Reading reports for 30 seconds...")
print("Press buttons and move sticks to see raw data.\n")

# Try to switch to full report mode (0x30)
# Output report 0x01: [report_id, rumble(8), subcmd_id, payload...]
rumble = [0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40]
subcmd_set_mode = [0x01] + rumble + [0x03, 0x30]  # subcmd 0x03, payload 0x30
subcmd_set_mode += [0x00] * (49 - len(subcmd_set_mode))

print("Attempting mode switch to 0x30...")
try:
    # Try without prefix
    written = dev.write(subcmd_set_mode)
    print(f"  write (no prefix): {written} bytes")
except Exception as e:
    print(f"  write (no prefix) failed: {e}")

try:
    # Try with 0x00 prefix
    written = dev.write([0x00] + subcmd_set_mode)
    print(f"  write (0x00 prefix): {written} bytes")
except Exception as e:
    print(f"  write (0x00 prefix) failed: {e}")

# Also try send_feature_report
try:
    feature_data = [0x00] + subcmd_set_mode[:7]
    written = dev.send_feature_report(feature_data)
    print(f"  send_feature_report: {written} bytes")
except Exception as e:
    print(f"  send_feature_report failed: {e}")

print("\nListening for reports (press buttons, move sticks)...\n")

report_count = 0
report_ids_seen = set()
start_time = time.time()

try:
    while time.time() - start_time < 30:
        data = dev.read(362, timeout_ms=100)
        if data:
            report_count += 1
            report_id = data[0]
            report_ids_seen.add(report_id)

            # Show first 16 bytes (buttons + start of stick data)
            hex_bytes = " ".join(f"{b:02x}" for b in data[:16])
            total_len = len(data)

            # For 0x3F reports, try to show button state
            if report_id == 0x3F and total_len >= 4:
                btn_byte = data[1]
                bits = f"{btn_byte:08b}"
                print(f"  [{report_count:4d}] 0x3F ({total_len:3d}B): {hex_bytes}  btn_byte={btn_byte:02x} bits={bits}")
            elif report_id == 0x30 and total_len >= 12:
                right_btn = data[3]
                shared_btn = data[4]
                left_btn = data[5]
                print(f"  [{report_count:4d}] 0x30 ({total_len:3d}B): {hex_bytes}  R={right_btn:02x} S={shared_btn:02x} L={left_btn:02x}")
            elif report_id == 0x21:
                print(f"  [{report_count:4d}] 0x21 ACK ({total_len:3d}B): {hex_bytes}")
            else:
                print(f"  [{report_count:4d}] 0x{report_id:02x} ({total_len:3d}B): {hex_bytes}")

except KeyboardInterrupt:
    pass

print(f"\n\nSummary: {report_count} reports received")
print(f"Report IDs seen: {sorted(hex(rid) for rid in report_ids_seen)}")

dev.close()
print("Done.")
