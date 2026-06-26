# Nintendo Switch Joy-Con 遥操作机械臂指南

本文档介绍如何使用 Nintendo Switch Joy-Con 手柄通过蓝牙控制 SO-101 机械臂，实现 6-DOF（位置 + 姿态）增量运动控制。

## 原理概述

Joy-Con 遥操作器采用**末端增量控制**模式：左摇杆控制 XY 位置，L/ZL 控制 Z 轴，右摇杆控制 Yaw/Pitch 姿态，R/ZR 控制 Roll。还支持**速度调节**（方向键 ±20%）和**微调模式**（摇杆按下切换）。

由于 SO-101 机械臂接收的是**关节空间命令**（如 `shoulder_pan.pos`），而 Joy-Con 输出的是**末端增量**（`delta_x/y/z/wx/wy/wz`），需要通过逆运动学（IK）管线进行转换：

```
Joy-Con 摇杆偏移
    ↓ 归一化 [-1, 1]，× step_size × speed_multiplier × fine_tune
delta_x/y/z + delta_wx/wy/wz + gripper  (7维)
    ↓ MapDeltaActionToRobotActionStep
enabled, target_x/y/z/wx/wy/wz, gripper_vel
    ↓ EEReferenceAndDelta (正向运动学 FK)
ee.x/y/z/wx/wy/wz, ee.gripper_vel  (绝对末端位姿)
    ↓ EEBoundsAndSafety (安全边界)
裁剪后的 ee.x/y/z/wx/wy/wz
    ↓ GripperVelocityToJoint (夹爪速度→位置)
ee.gripper_pos
    ↓ InverseKinematicsEEToJoints (逆运动学 IK)
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
wrist_flex.pos, wrist_roll.pos, gripper.pos
    ↓
SO-101 机械臂执行
```

## 前置条件

- 已安装 lerobot 及 SO-101 follower 机械臂（参考 [SO-ARM101 实践指南](./so_arm101_practice.md)）
- SO-101 已完成校准
- 一个或两个 Nintendo Switch Joy-Con（左/右均可）
- 电脑支持蓝牙（macOS / Linux / Windows）

## 安装

安装 Joy-Con 遥操作器和 IK 求解器的依赖：

```bash
uv pip install 'lerobot[joycon,feetech,kinematics]'
```

## 准备 URDF 文件

IK 求解器需要 SO-101 的 URDF 模型文件。我们提供一个去掉了视觉网格的精简版 URDF——IK 只需要关节/连杆结构，不需要视觉模型。

```bash
# 1. 下载原始 URDF
mkdir -p SO101
curl -L -o SO101/so101_new_calib.urdf \
  "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf"

# 2. 去掉 <visual> 和 <collision> 元素
python3 -c "
import re
with open('SO101/so101_new_calib.urdf') as f:
    content = f.read()
content = re.sub(r'\s*<visual>.*?</visual>', '', content, flags=re.DOTALL)
content = re.sub(r'\s*<collision>.*?</collision>', '', content, flags=re.DOTALL)
with open('SO101/so101_new_calib_no_mesh.urdf', 'w') as f:
    f.write(content)
print('Created SO101/so101_new_calib_no_mesh.urdf')
"
```

## 蓝牙配对 Joy-Con

### macOS

1. 打开 **系统设置 → 蓝牙**
2. 按住 Joy-Con 侧面的 **同步按钮**（小圆钮，位于 SL/SR 按钮之间）约 3 秒
3. 在蓝牙列表中找到 **Joy-Con (L)** 或 **Joy-Con (R)**，点击连接

### Linux (Ubuntu/Debian)

```bash
bluetoothctl
> scan on
> pair XX:XX:XX:XX:XX:XX
> trust XX:XX:XX:XX:XX:XX
> connect XX:XX:XX:XX:XX:XX
```

Linux 上可能需要 udev 规则：

```bash
sudo tee /etc/udev/rules.d/70-joycon.rules << 'EOF'
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="2006", MODE="0666"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="2007", MODE="0666"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="200e", MODE="0666"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## 支持的模式

| 模式 | 配置值 | 说明 |
|------|--------|------|
| **自动检测** | `auto` | 根据检测到的 Joy-Con 自动选择（默认） |
| **单个左 Joy-Con** | `single_left` | 仅使用左 Joy-Con |
| **单个右 Joy-Con** | `single_right` | 仅使用右 Joy-Con |
| **双 Joy-Con** | `dual` | 同时使用左右两个 Joy-Con（推荐） |

## 按键映射

### 双 Joy-Con 模式（推荐）

```
┌───────────────────────┐              ┌───────────────────────┐
│     左 Joy-Con         │              │     右 Joy-Con         │
│                       │              │                       │
│  摇杆 → XY 位置       │              │  摇杆 → Yaw / Pitch   │
│  L / ZL → Z 上/下     │              │  R / ZR → Roll CW/CCW │
│                       │              │                       │
│  方向键↑↓ → 速度±20%  │              │  A → 关闭夹爪 ✊       │
│  摇杆按下 → 微调切换   │              │  B → 打开夹爪 ✋       │
│                       │              │                       │
│  Minus → 失败         │              │  Plus → 成功          │
│                       │              │  Home → 紧急停止 🛑    │
└───────────────────────┘              └───────────────────────┘
```

| 操作 | 按键 | 说明 |
|------|------|------|
| X-Y 位置 | 左摇杆 | 推杆方向和幅度控制移动 |
| Z 轴（上下） | L（上）/ ZL（下） | 按钮式控制 |
| Yaw 偏航 | 右摇杆 X | 水平旋转 |
| Pitch 俯仰 | 右摇杆 Y | 前后倾斜 |
| Roll 翻滚 | R（顺）/ ZR（逆） | 绕 X 轴旋转 |
| 关闭夹爪 | A | 抓取 |
| 打开夹爪 | B | 释放 |
| 速度增加 | 方向键 ↑ | +20%，最高 200% |
| 速度降低 | 方向键 ↓ | -20%，最低 20% |
| 微调切换 | 摇杆按下 | 步长减半（50%） |
| 标记成功 | Plus (+) | 结束片段，标记成功 |
| 标记失败 | Minus (-) | 结束片段，标记失败 |
| 紧急停止 | Home | 立即停止所有运动 |

### 单个左 Joy-Con

| 操作 | 按键 |
|------|------|
| X-Y 位置 | 摇杆 |
| Z 轴 | L（上）/ ZL（下） |
| 速度调节 | 方向键 ↑/↓ |
| 微调切换 | 摇杆按下 |
| 关闭夹爪 | L / ZL |
| 标记失败 | Minus (-) |

> ⚠️ 单个左 Joy-Con 无姿态控制和打开夹爪功能。

### 单个右 Joy-Con

| 操作 | 按键 |
|------|------|
| X-Y 位置 | 摇杆 |
| Z 轴 | R（上）/ ZR（下） |
| 打开夹爪 | R / ZR |
| 标记成功 | Plus (+) |

> ⚠️ 单个右 Joy-Con 无姿态控制和关闭夹爪功能。

## 速度控制

### 速度调节

| 操作 | 功能 | 说明 |
|------|------|------|
| 方向键 ↑ | 增加速度 | 速度 +20% |
| 方向键 ↓ | 降低速度 | 速度 -20% |

- **最低速度**：20%（0.2 倍速）
- **默认速度**：100%（1.0 倍速）
- **最高速度**：200%（2.0 倍速）

速度变化会在终端实时打印：`Speed: 120%`

### 微调模式

- **激活**：按下左摇杆（L3）
- **效果**：所有步长减半（×0.5），适合精确定位
- **状态**：终端打印 `Fine-tune: ON/OFF`
- **叠加**：微调与速度倍率相乘（如 200% 速度 + 微调 = 100% 实际步长）

## 使用方法

### 遥操作（推荐方式）

使用专用示例脚本，它配置了完整的 IK 处理管线：

```bash
# 双 Joy-Con（推荐）
uv run python examples/joycon_to_so101/teleoperate.py \
    --port=/dev/tty.usbmodem5B7B0137181 \
    --urdf=SO101/so101_new_calib_no_mesh.urdf \
    --mode=dual

# 单个左 Joy-Con
uv run python examples/joycon_to_so101/teleoperate.py \
    --port=/dev/tty.usbmodem5B7B0137181 \
    --urdf=SO101/so101_new_calib_no_mesh.urdf \
    --mode=single_left
```

> ⚠️ **不要使用 `lerobot-teleoperate` CLI**。该命令使用恒等处理器，无法将 delta 动作转换为关节命令。必须使用示例脚本。

### 示例脚本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | (必填) | SO-101 串口路径 |
| `--urdf` | (必填) | URDF 文件路径 |
| `--mode` | `auto` | Joy-Con 模式 |
| `--step-size` | `1.0` | XY 方向灵敏度 |
| `--z-step-size` | `0.8` | Z 方向灵敏度 |
| `--deadzone` | `0.15` | 摇杆死区 |
| `--fps` | `30` | 控制循环频率 |

### Python API

```python
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline, robot_action_observation_to_transition, transition_to_robot_action
from lerobot.processor.delta_action_processor import MapDeltaActionToRobotActionStep
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety, EEReferenceAndDelta, GripperVelocityToJoint, InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.joycon import JoyConTeleop, JoyConTeleopConfig, JoyConMode

# 1. 创建并连接
teleop = JoyConTeleop(JoyConTeleopConfig(mode=JoyConMode.DUAL))
follower = SO100Follower(SOFollowerConfig(port="/dev/tty.usbmodem*"))
teleop.connect()
follower.connect(calibrate=False)

# 2. 构建 IK 管线
motor_names = list(follower.bus.motors.keys())
kinematics = RobotKinematics(
    urdf_path="SO101/so101_new_calib_no_mesh.urdf",
    target_frame_name="gripper_frame_link", joint_names=motor_names,
)
processor = RobotProcessorPipeline(
    steps=[
        MapDeltaActionToRobotActionStep(position_scale=1.0, noise_threshold=1e-3),
        EEReferenceAndDelta(kinematics=kinematics,
            end_effector_step_sizes={"x": 0.001, "y": 0.001, "z": 0.001},
            motor_names=motor_names, use_latched_reference=False),
        EEBoundsAndSafety(end_effector_bounds={"min":[-1,-1,-1],"max":[1,1,1]}, max_ee_step_m=0.5),
        GripperVelocityToJoint(speed_factor=20.0, clip_min=0.0, clip_max=100.0, discrete_gripper=True),
        InverseKinematicsEEToJoints(kinematics=kinematics, motor_names=motor_names,
            initial_guess_current_joints=True),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# 3. 控制循环
try:
    while teleop.is_connected:
        obs = follower.get_observation()
        raw_action = teleop.get_action()
        events = teleop.get_teleop_events()
        if events.get("emergency_stop"):
            print("Emergency stop!")
            break
        robot_action = processor((raw_action, obs))
        follower.send_action(robot_action)
finally:
    teleop.disconnect()
    follower.disconnect()
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | JoyConMode | `auto` | Joy-Con 使用模式 |
| `use_gripper` | bool | `True` | 是否包含夹爪控制 |
| `deadzone` | float | `0.15` | 摇杆死区（0~1） |
| `step_size` | float | `1.0` | XY 位置步进 |
| `z_step_size` | float | `1.0` | Z 轴步进 |
| `rotation_step` | float | `1.0` | 旋转步进（弧度） |
| `min_speed` | float | `0.2` | 最低速度倍率（20%） |
| `max_speed` | float | `2.0` | 最高速度倍率（200%） |
| `speed_step` | float | `0.2` | 速度调节步长（20%） |
| `fine_tune_multiplier` | float | `0.5` | 微调模式倍率（50%） |

### 调参建议

- **死区**：Joy-Con 摇杆容易漂移，推荐 0.15~0.25
- **步进**：IK 管线的 `end_effector_step_sizes` 已设为 0.001m/帧，配合 `step_size` 调节灵敏度
- **速度**：首次使用建议降到 50%，熟练后可提高到 150%~200%
- **微调**：精确对准时开启微调模式，步长自动减半

## 动作格式

Joy-Con 遥操作器输出 **7 维动作向量**：

```python
{
    "delta_x": float,   # X 位置增量
    "delta_y": float,   # Y 位置增量
    "delta_z": float,   # Z 位置增量
    "delta_wx": float,  # Yaw 旋转增量（绕 Z 轴）
    "delta_wy": float,  # Pitch 旋转增量（绕 Y 轴）
    "delta_wz": float,  # Roll 旋转增量（绕 X 轴）
    "gripper": int,     # 夹爪: 0=关闭, 1=保持, 2=打开
}
```

`get_teleop_events()` 返回的额外状态：

```python
{
    "emergency_stop": bool,      # Home 按钮紧急停止
    "speed_multiplier": float,   # 当前速度倍率 (0.2~2.0)
    "fine_tune": bool,           # 微调模式是否开启
    # ... 标准 TeleopEvents (SUCCESS, FAILURE 等)
}
```

## 技术细节

### IK 处理管线

| 步骤 | 处理器 | 输入 | 输出 |
|------|--------|------|------|
| 1 | `MapDeltaActionToRobotActionStep` | `delta_x/y/z/wx/wy/wz, gripper` | `enabled, target_x/y/z/wx/wy/wz, gripper_vel` |
| 2 | `EEReferenceAndDelta` | `target_*` + FK | `ee.x/y/z/wx/wy/wz, ee.gripper_vel` |
| 3 | `EEBoundsAndSafety` | `ee.x/y/z/wx/wy/wz` | 裁剪后的绝对位姿 |
| 4 | `GripperVelocityToJoint` | `ee.gripper_vel` (离散 0/1/2) | `ee.gripper_pos` (0~100) |
| 5 | `InverseKinematicsEEToJoints` | `ee.x/y/z/wx/wy/wz + ee.gripper_pos` | `shoulder_pan.pos, ...` |

### HID 协议

Joy-Con 使用 Nintendo 私有 HID 协议：
1. **简单模式 (0x3F)**：默认模式，仅按钮，**无摇杆数据**
2. **完整模式 (0x30)**：需初始化切换，包含摇杆 + 按钮 + IMU + 电池

连接时自动完成：刷新旧数据 → LED 唤醒 → subcmd `0x03` 切换模式（5 次重试，1.5s 超时）→ 备用 0x30 检测 → 20 次采样校准中心

### 摇杆数据格式

12-bit 打包（3 字节/轴），中心约 2048，归一化 [-1, 1]：
```
x_raw = b0 | ((b1 & 0x0F) << 8)    # 0~4095
y_raw = (b1 >> 4) | (b2 << 4)       # 0~4095
```

### 自动重连

蓝牙断开后以 1Hz 频率自动重连。双 Joy-Con 模式下支持优雅降级——一个断开时另一个继续工作。

### 电池监控

- 100%~25%：正常
- ≤15%：WARNING
- ≤5%：ERROR

## 故障排除

### EE jump 错误

```
ValueError: EE jump 0.192m > 0.05m
```

**原因**：`EEBoundsAndSafety` 的步长限制过小，或 `end_effector_step_sizes` 过大。

**解决**：确保 IK 管线中 `end_effector_step_sizes` 设为 `{"x": 0.001, "y": 0.001, "z": 0.001}`，`max_ee_step_m` 设为 `0.5`。

### 无法检测到 Joy-Con

1. 确认蓝牙已配对
2. 确认 `pip show hidapi` 已安装
3. Linux 检查 udev 规则
4. 重新配对

### Full report mode 初始化失败

摇杆不可用（仅按钮工作）。尝试：重新配对 → 检查电量 → 减少蓝牙干扰。

### StopIteration 错误

使用了 `lerobot-teleoperate` CLI。改用示例脚本。

### URDF 网格文件找不到

使用精简版 URDF（`so101_new_calib_no_mesh.urdf`）。

### IK 管线 KeyError

确保管线顺序正确：`MapDeltaAction → EEReferenceAndDelta → EEBoundsAndSafety → GripperVelocityToJoint → InverseKinematicsEEToJoints`。

### 摇杆漂移

增大 `--deadzone`（推荐 0.15~0.25）。

## 文件结构

```
src/lerobot/teleoperators/joycon/
├── __init__.py                    # 包导出
├── configuration_joycon.py        # JoyConTeleopConfig + JoyConMode
├── teleop_joycon.py               # JoyConTeleop(Teleoperator)
└── joycon_utils.py                # HID 协议、速度/微调/姿态控制

examples/joycon_to_so101/
└── teleoperate.py                 # 完整 IK 管线示例

tests/teleoperators/
└── test_joycon.py                 # 72 个单元测试

docs/
└── joycon_teleoperation.md        # 本文档
```
