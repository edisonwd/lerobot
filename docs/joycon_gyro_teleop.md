# Joy-Con 陀螺仪姿态主控遥控使用指南

> SO101 六轴机械臂 · 单只 Joy-Con · 陀螺仪主控 + 摇杆微调

## 目录

- [前提条件](#前提条件)
- [快速开始](#快速开始)
- [两种操控模式](#两种操控模式)
- [陀螺仪模式详细操作](#陀螺仪模式详细操作)
- [摇杆模式详细操作](#摇杆模式详细操作)
- [参数调优](#参数调优)
- [常见问题](#常见问题)

---

## 前提条件

### 1. 安装依赖

```bash
uv pip install 'lerobot[joycon,feetech]'
```

### 2. 校准 SO-101 机械臂

```bash
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem*
```

### 3. 蓝牙配对 Joy-Con

1. 按住 Joy-Con 侧面配对按钮（SL/SR 之间的小圆点）直到指示灯闪烁
2. 在系统蓝牙设置中配对 `Joy-Con (L)` 或 `Joy-Con (R)`
3. 左/右均可，单只即可

### 4. 确认 Joy-Con 连接

```bash
uv run python -c "
import hid
for d in hid.enumerate():
    if d['vendor_id'] == 0x057E:
        print(f\"Found: {d['product_string']} [{d['vendor_id']:04x}:{d['product_id']:04x}]\")
"
```

应输出类似 `Found: Joy-Con (L) [057e:2006]`。

---

## 快速开始

### 基础启动（内置默认映射）

```bash
uv run python examples/joycon_to_so101/teleoperate.py \
    --port=/dev/tty.usbmodem5B7B0137181
```

### 陀螺仪主控模式

```bash
uv run python examples/joycon_to_so101/teleoperate.py \
    --port=/dev/tty.usbmodem5B7B0137181 \
    --mapping=examples/joycon_to_so101/gyro_primary_mapping.yaml \
    --alt-mapping=examples/joycon_to_so101/stick_only_mapping.yaml
```

### 纯摇杆模式

```bash
uv run python examples/joycon_to_so101/teleoperate.py \
    --port=/dev/tty.usbmodem5B7B0137181 \
    --mapping=examples/joycon_to_so101/stick_only_mapping.yaml
```

---

## 两种操控模式

通过 `--mapping` 和 `--alt-mapping` 参数加载两套映射文件，按 **SL** 键一键切换。

| 模式 | 核心特点 | 适用场景 |
|------|---------|---------|
| **陀螺仪主控** | 手柄姿态直接控制大臂和肘部，直觉操控 | 日常操作、抓取任务 |
| **纯摇杆** | 传统双摇杆控制所有关节 | 调试、检查机械限位 |

---

## 陀螺仪模式详细操作

### 核心操控逻辑

> 拿手柄像拿着机械臂末端一样摆动 — 手柄整体运动 = 机械臂空间运动

| 你的动作 | 机械臂响应 | 灵敏度 |
|---------|-----------|--------|
| 手柄前后俯仰 | 大臂升降（轴2） | 低增益(0.55)，防抖 |
| 手柄左右侧翻 | 肘部屈伸（轴3） | 中增益(0.95) |
| 左摇杆左右 | 底座旋转（轴1） | 标准 |
| 左摇杆上下 | 大臂微调（轴2） | 低(0.8)，配合陀螺仪 |
| 右摇杆上下 | 腕部俯仰（轴4） | 高增益(1.15) |
| ZL / ZR | 腕部自转 逆/顺时针（轴5） | 高增益(1.15) |
| L / R | 夹爪 闭合/张开（轴6） | 快速(3.0) |

### 功能按键

| 按键 | 功能 | 说明 |
|------|------|------|
| **SL** | 模式切换 | 陀螺仪 ↔ 摇杆模式 |
| **SR** | IMU 滤波切换 | 灵敏(α=0.02) ↔ 稳定(α=0.005) |
| **D-pad ↑** | 速度档位 | 标准(1.0x) → 高速(1.5x) → 精细(0.5x) 循环 |
| **D-pad ↓** | 回中位 | 所有关节归零 |
| **D-pad ←** | IMU 校准 | 消除陀螺仪长时间漂移 |
| **D-pad →** | 姿态锁定 | 冻结当前所有关节角度 |
| **A** | 急停 | 所有舵机锁死 |
| **B** | 拾取预设 | 移动到大臂-30°/肘45°/腕-20° |
| **X** | 放置预设 | 移动到大臂30°/肘-20°/腕10° |
| **Y** | 夹爪切换 | 全开 ↔ 全闭（状态翻转） |
| **Home** | 急停 | 硬编码安全保护 |
| **Plus / Minus** | 片段标记 | 录制时标记成功/失败 |

### 操作技巧

**粗定位**：大幅度摆动手柄 → 机械臂快速大范围移动

**精对位**：轻微倾斜手柄 + 左摇杆微调 → 高精度对准（拾取小物件、插孔）

**防抖**：定点操作时按 SR 切换到稳定模式，抑制手部微抖

**长时间使用**：每隔几分钟按 D-pad ← 重新校准 IMU 零点

---

## 摇杆模式详细操作

纯摇杆模式，无陀螺仪参与，适合调试。

| 输入 | 关节 |
|------|------|
| 左摇杆 X | 底座旋转（轴1） |
| 左摇杆 Y | 大臂升降（轴2） |
| 右摇杆 Y | 肘部屈伸（轴3） |
| 右摇杆 X | 腕部俯仰（轴4） |
| ZL / ZR | 腕部自转（轴5） |
| L / R | 夹爪（轴6） |

功能按键（D-pad、ABXY、SL、SR、Home、Plus/Minus）与陀螺仪模式相同。

---

## 参数调优

### 调整每轴灵敏度

编辑 YAML 映射文件中的 `speed` 字段：

```yaml
mappings:
  # speed 值越大，该轴响应越灵敏
  # 根据你的减速比调整：
  # - 1:345（大臂）→ 建议 0.4~0.6
  # - 1:191（底座/肘部）→ 建议 0.8~1.2
  # - 1:147（腕部）→ 建议 1.0~1.3
  - input: gyro_pitch_delta
    motor: shoulder_lift
    control: incremental
    speed: 0.55       # ← 调这里
```

### 调整预设位置

```yaml
presets:
  pickup:
    shoulder_lift: -30.0   # ← 改成你需要的角度
    elbow_flex: 45.0
    wrist_flex: -20.0
  place:
    shoulder_lift: 30.0
    elbow_flex: -20.0
    wrist_flex: 10.0
```

### 调整速度档位

在 Python 代码中配置（或修改 YAML）：

```python
from lerobot.teleoperators.joycon import JoyConTeleopConfig

config = JoyConTeleopConfig(
    speed_levels=[0.3, 1.0, 2.0],  # 精细、标准、高速
)
```

### 调整陀螺仪死区

死区控制"多小的手柄动作会被忽略"。在 `joycon_utils.py` 中：

```python
self._gyro_delta_deadzone: float = 0.1  # 单位：度，增大可抑制更多微抖
```

### 调整互补滤波系数

```python
self._imu_alpha_normal: float = 0.02      # 正常模式（越大越跟手，越小越平滑）
self._imu_alpha_stabilized: float = 0.005  # 稳定模式（精密操作时用）
```

---

## 常见问题

### Q: 机械臂跟着手柄慢慢漂移？

陀螺仪积分会随时间漂移。**解决**：按 D-pad ← 重新校准 IMU 零点。建议每隔几分钟校准一次。

### Q: 手柄静止时机械臂还有微小动作？

陀螺仪有微噪声。**解决**：
1. 按 SR 切换到稳定滤波模式
2. 或增大死区值（见参数调优）

### Q: 大臂动作太猛/太抖？

大臂减速比 1:345 最大，需要最低增益。**解决**：在 YAML 中降低 `shoulder_lift` 的 `speed` 值（建议 0.4~0.55）。

### Q: 切换到摇杆模式后切不回来？

确保同时传了 `--mapping` 和 `--alt-mapping` 两个参数。只传一个则无法切换。

### Q: macOS 蓝牙连接不上？

1. 确保系统偏好设置 → 蓝牙中已完成配对
2. 检查 Joy-Con 电量（低电量可能导致连接不稳定）
3. 尝试删除配对后重新配对

### Q: Linux 权限不足？

添加 udev 规则：

```bash
sudo tee /etc/udev/rules.d/50-joycon.rules << 'EOF'
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="057e", MODE="0666"
EOF
sudo udevadm control --reload-rules
```

### Q: 预设位置不对？

编辑 YAML 文件中的 `presets` 部分，改成适合你工作台的角度。

---

## 命令速查

```bash
# 查看可用串口
ls /dev/tty.usbmodem*

# 陀螺仪模式（推荐）
uv run python examples/joycon_to_so101/teleoperate.py \
    --port=/dev/tty.usbmodem* \
    --mapping=examples/joycon_to_so101/gyro_primary_mapping.yaml \
    --alt-mapping=examples/joycon_to_so101/stick_only_mapping.yaml

# 纯摇杆模式
uv run python examples/joycon_to_so101/teleoperate.py \
    --port=/dev/tty.usbmodem* \
    --mapping=examples/joycon_to_so101/stick_only_mapping.yaml

# 自定义帧率
uv run python examples/joycon_to_so101/teleoperate.py \
    --port=/dev/tty.usbmodem* \
    --mapping=examples/joycon_to_so101/gyro_primary_mapping.yaml \
    --fps=60
```
