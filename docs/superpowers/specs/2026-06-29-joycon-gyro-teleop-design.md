# SO101 六轴机械臂 Joy-Con 陀螺仪姿态主控遥控方案设计

> 日期: 2026-06-29
> 状态: 设计阶段

## 概述

本设计在 LeRobot 现有 Joy-Con 遥控架构基础上，增加陀螺仪姿态主控功能。使用单只 Joy-Con（左或右均可），陀螺仪俯仰/横滚倾角作为核心姿态输入，摇杆做辅助微调，适配 SO101 六轴舵机配置。

## 设计决策记录

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 实现位置 | Python 端（宿主 PC） | 复用现有 hidapi + MappingEngine 架构，无需 ESP32 |
| 偏航(Yaw)来源 | 摇杆（非 IMU） | Joy-Con 无磁力计，陀螺仪积分偏航会漂移 |
| 传感器融合 | 互补滤波器 | 简单高效，加速度计修正俯仰/横滚长期漂移 |
| 扳机控制 | 布尔开关 | 现有 HID 层已支持，增量控制足够 |
| 模式切换 | 双 YAML 映射文件 | SL 键切换 MappingEngine 实例，与现有架构一致 |
| 速度系统 | 3 档离散 | 高速/标准/精细，替代原有连续 ±20% |

## 舵机配置

| 轴 | 关节名 | 减速比 | 增益系数 K |
|----|--------|--------|-----------|
| 1 | shoulder_pan（底座） | 1:191 | 1.0 |
| 2 | shoulder_lift（大臂） | 1:345 | 0.55 |
| 3 | elbow_flex（肘部） | 1:191 | 0.95 |
| 4 | wrist_flex（腕俯仰） | 1:147 | 1.15 |
| 5 | wrist_roll（腕自转） | 1:147 | 1.15 |
| 6 | gripper（夹爪） | 1:147 | 1.15 |

增益系数通过 YAML 映射文件的 `speed` 字段配置，用户可自行调整。

## 一、IMU 增强：陀螺仪解析 + 互补滤波

### 1.1 现有状态

当前 `joycon_utils.py` 仅解析加速度计（字节 13-18），计算 `imu_tilt` 和 `imu_roll` 两个绝对角度。陀螺仪数据（字节 19-24）未读取。

### 1.2 陀螺仪数据解析

Joy-Con 完整报告（0x30）包含 3 组 IMU 采样。解析第一组：

```
字节 19-20: gyro_x (int16 LE) — X 轴角速度
字节 21-22: gyro_y (int16 LE) — Y 轴角速度
字节 23-24: gyro_z (int16 LE) — Z 轴角速度
```

单位：约 0.0027°/s per LSB（出厂校准，±2000°/s 量程）。报告速率约 60Hz，每帧约 16.7ms。

### 1.3 互补滤波算法

```
加速度计角度 = atan2(accel, accel_z)    # 绝对值，但有噪声
陀螺仪角度 += gyro × dt                 # 平滑，但会漂移
融合角度 = α × 加速度计角度 + (1-α) × 陀螺仪角度
```

- **α = 0.02**（正常模式）：低值信任陀螺仪短期响应，加速度计修正长期漂移
- **α = 0.005**（稳定模式）：重度平滑，抑制微抖动，适合精密操作
- **dt**：使用 `time.monotonic()` 测量帧间时间差

两路输出：
- `imu_pitch`：融合俯仰角（加速度计 X + 陀螺仪 Y 融合）
- `imu_roll`：融合横滚角（加速度计 Y + 陀螺仪 X 融合）

### 1.4 角度增量计算

增量控制需要每帧的**角度变化量**，而非绝对角度：

```
gyro_pitch_delta = current_pitch - previous_pitch
gyro_roll_delta  = current_roll  - previous_roll
```

### 1.5 陀螺仪死区

在 `_parse_imu()` 中对增量添加 ±0.1° 死区，抑制静止时的微小抖动：

```
if abs(gyro_pitch_delta) < 0.1:
    gyro_pitch_delta = 0.0
```

### 1.6 新增 IMU 输出

| 输出名 | 类型 | 说明 |
|--------|------|------|
| `imu_pitch` | float（度） | 融合俯仰绝对角 |
| `imu_roll` | float（度） | 融合横滚绝对角 |
| `gyro_pitch_delta` | float（度） | 每帧俯仰变化量 |
| `gyro_roll_delta` | float（度） | 每帧横滚变化量 |
| `imu_filter_stabilized` | bool | 滤波模式状态 |

### 1.7 运行时校准

- 连接时自动校准（现有 `calibrate_imu()` 扩展支持陀螺仪积分归零）
- D-pad ← 一键重新校准（消除长时间积分漂移）

## 二、输入映射与增益

### 2.1 新增 VALID_INPUTS

在 `mapping_engine.py` 中添加：

```python
VALID_INPUTS |= {
    "imu_pitch",          # 融合俯仰绝对角
    "imu_roll",           # 融合横滚绝对角
    "gyro_pitch_delta",   # 俯仰每帧变化量
    "gyro_roll_delta",    # 横滚每帧变化量
}
```

### 2.2 增益与 YAML speed 字段的对应

MappingEngine 增量模式计算公式：

```python
delta = value × entry.speed × speed_multiplier × fine_scale
```

对于陀螺仪输入，`value` 是角度增量（度），`speed` 即增益系数 K：

- 轴 1（1:191）：`speed: 1.0`
- 轴 2（1:345）：`speed: 0.55`（减速比最大，增益最低，防抖核心）
- 轴 3（1:191）：`speed: 0.95`
- 轴 4/5/6（1:147）：`speed: 1.15`（减速小，更灵敏）

用户可在 YAML 文件中自由调整这些值。

### 2.3 陀螺仪模式映射文件

`examples/joycon_to_so101/gyro_primary_mapping.yaml`：

```yaml
mappings:
  # 轴 1: 底座旋转 — 左摇杆 X（摇杆控制偏航）
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
    speed: 1.5

  # 轴 2: 大臂升降 — 陀螺仪俯仰（主控）
  - input: gyro_pitch_delta
    motor: shoulder_lift
    control: incremental
    speed: 0.55
    invert: true              # 前倾 → 大臂下压

  # 轴 2 微调 — 左摇杆 Y
  - input: left_stick_y
    motor: shoulder_lift
    control: incremental
    speed: 0.8

  # 轴 3: 肘部屈伸 — 陀螺仪横滚（主控）
  - input: gyro_roll_delta
    motor: elbow_flex
    control: incremental
    speed: 0.95

  # 轴 4: 腕部俯仰 — 右摇杆 Y
  - input: right_stick_y
    motor: wrist_flex
    control: incremental
    speed: 1.15

  # 轴 5: 腕部自转 — ZL(逆时针) / ZR(顺时针)
  - input: zl
    motor: wrist_roll
    control: incremental
    speed: 1.15
    invert: true

  - input: zr
    motor: wrist_roll
    control: incremental
    speed: 1.15

  # 轴 6: 夹爪 — L(闭合) / R(张开)
  - input: l
    motor: gripper
    control: incremental
    speed: 3.0

  - input: r
    motor: gripper
    control: incremental
    speed: 3.0
    invert: true

meta_controls:
  speed_up: dpad_up
  reset_to_center: dpad_down
  recalibrate_imu: dpad_left
  pose_lock: dpad_right
  mode_switch: sl_right
  filter_toggle: sr_right
  fine_tune_toggle: l_stick_press

presets:
  pickup:
    shoulder_lift: -30.0
    elbow_flex: 45.0
    wrist_flex: -20.0
  place:
    shoulder_lift: 30.0
    elbow_flex: -20.0
    wrist_flex: 10.0
```

### 2.4 摇杆模式映射文件

`examples/joycon_to_so101/stick_only_mapping.yaml`：

```yaml
mappings:
  # 轴 1: 底座 — 左摇杆 X
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
    speed: 1.5

  # 轴 2: 大臂 — 左摇杆 Y
  - input: left_stick_y
    motor: shoulder_lift
    control: incremental
    speed: 1.5
    invert: true

  # 轴 3: 肘部 — 右摇杆 Y
  - input: right_stick_y
    motor: elbow_flex
    control: incremental
    speed: 1.5

  # 轴 4: 腕俯仰 — 右摇杆 X
  - input: right_stick_x
    motor: wrist_flex
    control: incremental
    speed: 1.15

  # 轴 5: 腕自转 — ZL / ZR
  - input: zl
    motor: wrist_roll
    control: incremental
    speed: 1.15
    invert: true

  - input: zr
    motor: wrist_roll
    control: incremental
    speed: 1.15

  # 轴 6: 夹爪 — L / R
  - input: l
    motor: gripper
    control: incremental
    speed: 3.0

  - input: r
    motor: gripper
    control: incremental
    speed: 3.0
    invert: true

meta_controls:
  speed_up: dpad_up
  reset_to_center: dpad_down
  recalibrate_imu: dpad_left
  pose_lock: dpad_right
  mode_switch: sl_right
  filter_toggle: sr_right
  fine_tune_toggle: l_stick_press

presets:
  pickup:
    shoulder_lift: -30.0
    elbow_flex: 45.0
    wrist_flex: -20.0
  place:
    shoulder_lift: 30.0
    elbow_flex: -20.0
    wrist_flex: 10.0
```

## 三、双模式系统与元控制

### 3.1 模式切换（SL 键）

**配置变更** — `configuration_joycon.py`：

```python
@dataclass
class JoyConTeleopConfig(TeleoperatorConfig):
    mapping_path: str | None = None          # 主映射（陀螺仪模式）
    alt_mapping_path: str | None = None      # 备用映射（摇杆模式）
```

**运行时行为** — `teleop_joycon.py`：

- `connect()` 时加载两个 MappingEngine：`self.mapping_engine`（陀螺仪）和 `self.alt_mapping_engine`（摇杆）
- `self._active_mode: str` 追踪当前激活模式（`"gyro"` 或 `"stick"`）
- `get_action()` 使用激活的引擎
- SL 键（边沿触发）交换激活引擎，从当前机器人位置重新初始化 `_current_targets`，避免跳变

### 3.2 扩展 MetaControls

**`mapping_engine.py` 变更**：

```python
@dataclass
class MetaControls:
    speed_up: str = "dpad_up"
    reset_to_center: str = "dpad_down"       # 替代原 speed_down
    fine_tune_toggle: str = "l_stick_press"
    # 新增：
    mode_switch: str = "sl_right"
    recalibrate_imu: str = "dpad_left"
    pose_lock: str = "dpad_right"
    filter_toggle: str = "sr_right"
```

MetaControls 验证逻辑扩展：所有新字段检查 `val in VALID_INPUTS`。

### 3.3 速度系统

替换连续速度倍率为 3 档离散：

```python
speed_levels: list[float] = [0.5, 1.0, 1.5]  # 精细、标准、高速
speed_level_index: int = 1                     # 默认标准档
```

- D-pad ↑ 循环切换：`index = (index + 1) % len(speed_levels)`
- 精细模式下在当前档位基础上减半（与档位叠加）
- 精细模式自动降低轴 2 大臂陀螺仪增益（适合精密摆放）

### 3.4 姿态锁定

D-pad → 按下（边沿触发）：

- 冻结 `_current_targets`，后续所有 `get_action()` 返回冻结的目标值
- 忽略所有摇杆/陀螺仪/按钮输入对电机的控制
- 再次按下解锁

### 3.5 预设位置（B/X 键）

在 YAML 中定义预设：

```yaml
presets:
  pickup:
    shoulder_lift: -30.0
    elbow_flex: 45.0
    wrist_flex: -20.0
  place:
    shoulder_lift: 30.0
    elbow_flex: -20.0
    wrist_flex: 10.0
```

- **B 键**：触发 `pickup` 预设，列出的电机设为目标角度，未列出的电机保持当前位置不变
- **X 键**：触发 `place` 预设，同上
- 预设角度受关节限位约束（clamp 到 min/max）
- 预设触发是瞬时覆盖：仅修改 `_current_targets`，后续帧恢复增量控制

### 3.6 硬编码按键（安全/录制相关，不可重映射）

| 按键 | 动作 | 原因 |
|------|------|------|
| Home | 急停 | 安全保护，必须始终有效 |
| A | 急停（冗余） | 安全冗余 |
| Plus | 片段标记成功 | 录制工作流 |
| Minus | 片段标记失败 | 录制工作流 |

### 3.7 完整按键分配表

#### 陀螺仪模式（模式 1）

| 输入 | 动作 |
|------|------|
| 陀螺仪俯仰增量 | 轴 2 大臂升降（主控） |
| 陀螺仪横滚增量 | 轴 3 肘部屈伸（主控） |
| 左摇杆 X | 轴 1 底座旋转 |
| 左摇杆 Y | 轴 2 大臂升降（微调） |
| 右摇杆 Y | 轴 4 腕部俯仰 |
| ZL / ZR | 轴 5 腕部自转（逆时针/顺时针） |
| L / R | 轴 6 夹爪（闭合/张开） |
| SL | 切换陀螺仪/摇杆模式 |
| SR | 切换 IMU 滤波模式（灵敏/稳定） |
| D-pad ↑ | 循环速度档位（高速/标准/精细） |
| D-pad ↓ | 所有关节回中位 |
| D-pad ← | 重新校准 IMU 零点 |
| D-pad → | 姿态锁定开关 |
| A | 急停 |
| B | 拾取预设位 |
| X | 放置预设位 |
| Y | 夹爪全开/全闭切换（状态翻转：首次按下→全开，再次→全闭） |
| Home | 急停（硬编码） |
| Plus / Minus | 片段成功/失败（硬编码） |

#### 摇杆模式（模式 2）

| 输入 | 动作 |
|------|------|
| 左摇杆 X / Y | 轴 1 底座 + 轴 2 大臂 |
| 右摇杆 X / Y | 轴 3 肘部 + 轴 4 腕俯仰 |
| ZL / ZR | 轴 5 腕部自转 |
| L / R | 轴 6 夹爪 |
| 其余按键 | 与模式 1 相同 |

## 四、数据流

```
Joy-Con HID 报告 (0x30)
  ├── 字节 3-5:   按键 → self.buttons 字典
  ├── 字节 6-8:   左摇杆 → left_x, left_y
  ├── 字节 9-11:  右摇杆 → right_x, right_y
  ├── 字节 13-18: 加速度计 → accel_x, accel_y, accel_z
  └── 字节 19-24: 陀螺仪 → gyro_x, gyro_y, gyro_z
                          │
                          ▼
              ┌─ 互补滤波器 ─────────────┐
              │ pitch = α·accel_angle    │
              │       + (1-α)·gyro_angle │
              │ roll  = α·accel_angle    │
              │       + (1-α)·gyro_angle │
              └──────────────────────────┘
                          │
                          ▼
              ┌─ 增量计算 ───────────────┐
              │ gyro_pitch_delta =       │
              │   pitch - prev_pitch     │
              │ gyro_roll_delta =        │
              │   roll - prev_roll       │
              └──────────────────────────┘
                          │
                          ▼
              _build_input_state()
              {
                "left_stick_x": 0.3,
                "gyro_pitch_delta": 0.8,
                "gyro_roll_delta": -0.2,
                "zl": False, "zr": True,
                ...
              }
                          │
                          ▼
              _handle_meta_controls()
              边沿检测: SL→模式切换, D-pad→姿态锁定 等
              修改: speed, pose_lock, filter, active_engine
                          │
                          ▼
              激活的 MappingEngine.compute_targets()
              input_state + current_targets → new_targets
              (陀螺仪增量 × 每轴增益 → 增量式位置)
                          │
                          ▼
              {"shoulder_pan.pos": 45.2,
               "shoulder_lift.pos": -12.3, ...}
                          │
                          ▼
              Robot.send_action()
```

## 五、文件变更清单

| 文件 | 变更内容 | 规模 |
|------|---------|------|
| `src/lerobot/teleoperators/joycon/joycon_utils.py` | 添加陀螺仪解析、互补滤波、增量计算、滤波模式切换、3 档速度 | 中等 |
| `src/lerobot/teleoperators/joycon/mapping_engine.py` | 添加 `gyro_pitch_delta`/`gyro_roll_delta` 到 VALID_INPUTS，扩展 MetaControls，添加预设支持 | 中等 |
| `src/lerobot/teleoperators/joycon/teleop_joycon.py` | 双引擎加载、SL 模式切换、姿态锁定、预设处理、扩展元控制处理 | 中等 |
| `src/lerobot/teleoperators/joycon/configuration_joycon.py` | 添加 `alt_mapping_path`、`speed_levels` 配置 | 小 |
| `examples/joycon_to_so101/gyro_primary_mapping.yaml` | 新建：陀螺仪模式映射 | 小 |
| `examples/joycon_to_so101/stick_only_mapping.yaml` | 新建：摇杆模式映射 | 小 |
| `examples/joycon_to_so101/teleoperate.py` | 更新以传递 `alt_mapping_path` | 小 |

## 六、测试计划

| 测试 | 覆盖内容 |
|------|---------|
| `test_complementary_filter` | 使用模拟加速度计+陀螺仪数据验证俯仰/横滚融合 |
| `test_gyro_delta_computation` | 增量 = 当前 - 上一帧，校准时重置 |
| `test_gyro_inputs_in_mapping_engine` | `gyro_pitch_delta` 通过增量模式正确流转 |
| `test_mode_switching` | SL 交换激活引擎，保持目标位置 |
| `test_pose_lock` | D-pad → 冻结目标，忽略输入 |
| `test_presets` | B/X 设置电机到预设角度 |
| `test_3_level_speed` | D-pad ↑ 在 [0.5, 1.0, 1.5] 间循环 |
| `test_filter_toggle` | SR 切换 α 在 0.02 和 0.005 之间 |
| `test_imu_recalibrate` | D-pad ← 归零陀螺仪积分角度 |
