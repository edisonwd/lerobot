#!/usr/bin/env bash
# Repeat lerobot-replay a specified number of times.
# Usage: ./replay-repeat.sh [num_times]
# Default: 1

set -euo pipefail

NUM_REPEATS="${1:-1}"

for ((i = 1; i <= NUM_REPEATS; i++)); do
  echo "=== Replay round ${i}/${NUM_REPEATS} ==="
  lerobot-replay \
      --robot.type=so101_follower \
      --robot.port=/dev/tty.usbmodem5B7B0137181 \
      --robot.id=my_awesome_follower_arm \
      --dataset.root=/Users/edison/myprojects/lerobot/data/my_dataset2 \
      --dataset.repo_id=${HF_USER}/record-test \
      --dataset.episode=1
  echo "=== Round ${i} completed ==="
done

echo "All ${NUM_REPEATS} round(s) done."
