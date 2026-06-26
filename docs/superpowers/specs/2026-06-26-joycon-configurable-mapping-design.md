# Joy-Con 可配置映射设计

## 概述

将 Joy-Con → SO-101 舵机的按键映射从硬编码改为 YAML 可配置，映射引擎集成在 `JoyConTeleop` 类内部，`get_action()` 直接返回舵机目标位置。

## 设计决策

| 决策项 | 结论 |
|--------|------|
| 配置格式 | YAML 文件 |
| 映射引擎位置 | 集成到 `JoyConTeleop`，`get_action()` 返回舵机目标 |
| 可映射输入 | 摇杆、按键、IMU、摇杆按下——全部可映射 |
| 系统功能 | 紧急停止/成功失败固定不变；速度调节/微调模式可配置按键 |
| 关节限位 | 从 robot 对象自动读取，不在 YAML 中配置 |

## SO-101 舵机参考

| 名称 | 舵机编号 | 减速比 | 关节限位（度） |
|------|---------|--------|--------------|
| Shoulder Pan | 1 | 1/191 | ±110 |
| Shoulder Lift | 2 | 1/345 | ±100 |
| Elbow Flex | 3 | 1/191 | ±97 |
| Wrist Flex | 4 | 1/147 | ±95 |
| Wrist Roll | 5 | 1/147 | ±163 / -157 |
| Gripper | 6 | 1/147 | 0 ~ 100 |

关节限位在运行时从 robot 的 calibration 数据读取，YAML 中不写。

---

## 1. YAML 配置文件格式

文件路径：用户通过 `--mapping=path.yaml` 指定，不指定则使用内置默认映射。

### 示例（完整默认映射）

```yaml
# joycon_mapping.yaml
# Joy-Con → SO-101 舵机映射配置
#
# 输入类型:
#   摇杆:  left_stick_x, left_stick_y, right_stick_x, right_stick_y
#   按键:  a, b, x, y, l, zl, r, zr
#          dpad_up, dpad_down, dpad_left, dpad_right
#          plus, minus, home, capture
#          l_stick_press, r_stick_press
#   体感:  imu_tilt (前后倾斜), imu_roll (左右翻滚)
#
# 控制模式:
#   incremental — 每帧增减角度 (摇杆/按键)
#   absolute    — 直接设定目标角度 (IMU)

mappings:
  # Motor 1: Shoulder Pan (左右转)
  - input: left_stick_x
    motor: shoulder_pan
    control: incremental
    speed: 1.5
    invert: false

  # Motor 2: Shoulder Lift (上下抬)
  - input: left_stick_y
    motor: shoulder_lift
    control: incremental
    speed: 1.5
    invert: true

  # Motor 3: Elbow Flex (肘部)
  - input: l
    motor: elbow_flex
    control: incremental
    speed: 1.5
    invert: false

  - input: zl
    motor: elbow_flex
    control: incremental
    speed: 1.5
    invert: true

  # Motor 4: Wrist Flex (手腕俯仰)
  - input: imu_tilt
    motor: wrist_flex
    control: absolute
    scale: 0.5

  # Motor 5: Wrist Roll (手腕旋转)
  - input: imu_roll
    motor: wrist_roll
    control: absolute
    scale: 0.5

  # Motor 6: Gripper (夹爪)
  - input: a
    motor: gripper
    control: incremental
    speed: 3.0
    invert: false

  - input: b
    motor: gripper
    control: incremental
    speed: 3.0
    invert: true

meta_controls:
  speed_up: dpad_up
  speed_down: dpad_down
  fine_tune_toggle: l_stick_press
```

### 字段说明

| 字段 | 类型 | 适用 | 说明 |
|------|------|------|------|
| `input` | string | 所有 | Joy-Con 输入标识符 |
| `motor` | string | 所有 | 目标舵机名称 |
| `control` | string | 所有 | `"incremental"` 或 `"absolute"` |
| `speed` | float | incremental | 每帧移动度数（摇杆满偏 / 按键按下时） |
| `scale` | float | absolute | IMU 角度乘数 |
| `invert` | bool | 两者 | 反转方向 |

### 规则

- 同一个舵机可以有多条映射（如 L 和 ZL 都映射 `elbow_flex`）
- 两个按键同时按同舵机正反向 → 互相抵消，舵机不动
- 未映射的舵机保持上一帧位置
- `meta_controls` 可省略，省略时使用默认值

### 系统功能（固定不可配置）

| 按键 | 功能 | 说明 |
|------|------|------|
| `home` | 紧急停止 | 安全功能，不可重新映射 |
| `plus` | 成功标记 | 数据采集标记，不可重新映射 |
| `minus` | 失败标记 | 数据采集标记，不可重新映射 |

---

## 2. MappingEngine 类

### 文件结构

```
src/lerobot/teleoperators/joycon/
├── configuration_joycon.py   # + mapping_path 字段
├── joycon_utils.py           # 不改动
├── mapping_engine.py         # ← 新增
└── teleop_joycon.py          # + 集成 MappingEngine
```

### 核心类

```python
# src/lerobot/teleoperators/joycon/mapping_engine.py

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ── 输入常量 ──────────────────────────────────────────

VALID_INPUTS = {
    # 摇杆
    "left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y",
    # 按键
    "a", "b", "x", "y", "l", "zl", "r", "zr",
    "dpad_up", "dpad_down", "dpad_left", "dpad_right",
    "plus", "minus", "home", "capture",
    "l_stick_press", "r_stick_press",
    # IMU
    "imu_tilt", "imu_roll",
}

STICK_INPUTS = {"left_stick_x", "left_stick_y", "right_stick_x", "right_stick_y"}
BUTTON_INPUTS = VALID_INPUTS - STICK_INPUTS - {"imu_tilt", "imu_roll"}
IMU_INPUTS = {"imu_tilt", "imu_roll"}


# ── 数据类 ────────────────────────────────────────────

@dataclass
class MappingEntry:
    """一条映射规则：一个 Joy-Con 输入 → 一个舵机。"""
    input: str
    motor: str
    control: str  # "incremental" | "absolute"
    speed: float = 1.5
    scale: float = 1.0
    invert: bool = False

    def __post_init__(self):
        if self.input not in VALID_INPUTS:
            raise ValueError(f"Unknown input: {self.input!r}. Valid: {sorted(VALID_INPUTS)}")
        if self.control not in ("incremental", "absolute"):
            raise ValueError(f"control must be 'incremental' or 'absolute', got {self.control!r}")


@dataclass
class MetaControls:
    """可配置的系统功能按键。"""
    speed_up: str = "dpad_up"
    speed_down: str = "dpad_down"
    fine_tune_toggle: str = "l_stick_press"

    def __post_init__(self):
        for name in ("speed_up", "speed_down", "fine_tune_toggle"):
            val = getattr(self, name)
            if val not in VALID_INPUTS:
                raise ValueError(f"meta_controls.{name}: unknown input {val!r}")


# ── 映射引擎 ──────────────────────────────────────────

class MappingEngine:
    """将 Joy-Con 原始输入转换为舵机目标位置。"""

    def __init__(
        self,
        mappings: list[MappingEntry],
        meta_controls: MetaControls,
        joint_limits: dict[str, tuple[float, float]],
    ):
        self.mappings = mappings
        self.meta_controls = meta_controls
        self.joint_limits = joint_limits

        # 校验：每个映射的 motor 必须存在于 joint_limits 中
        for entry in mappings:
            if entry.motor not in joint_limits:
                raise ValueError(
                    f"Motor {entry.motor!r} not found in joint_limits. "
                    f"Available: {sorted(joint_limits.keys())}"
                )

    @property
    def motors(self) -> list[str]:
        """返回所有被映射的舵机名称（去重、保持顺序）。"""
        seen = set()
        result = []
        for entry in self.mappings:
            if entry.motor not in seen:
                seen.add(entry.motor)
                result.append(entry.motor)
        return result

    def compute_targets(
        self,
        input_state: dict[str, float | bool],
        current_targets: dict[str, float],
        speed_multiplier: float = 1.0,
        fine_tune: bool = False,
    ) -> dict[str, float]:
        """
        根据 Joy-Con 输入计算新的舵机目标位置。

        Args:
            input_state: Joy-Con 当前所有输入值
                摇杆: float [-1, 1]
                按键: bool
                IMU:  float 度
            current_targets: 当前各舵机目标位置
            speed_multiplier: 速度倍率 (0.2 ~ 2.0)
            fine_tune: 微调模式是否开启

        Returns:
            更新后的舵机目标位置字典
        """
        targets = dict(current_targets)
        fine_scale = 0.5 if fine_tune else 1.0

        for entry in self.mappings:
            raw = input_state.get(entry.input)
            if raw is None:
                continue

            value = float(raw)

            if entry.control == "incremental":
                if entry.invert:
                    value = -value
                delta = value * entry.speed * speed_multiplier * fine_scale
                lo, hi = self.joint_limits[entry.motor]
                targets[entry.motor] = max(lo, min(hi, targets[entry.motor] + delta))

            elif entry.control == "absolute":
                target = value * entry.scale
                if entry.invert:
                    target = -target
                lo, hi = self.joint_limits[entry.motor]
                targets[entry.motor] = max(lo, min(hi, target))

        return targets

    # ── 加载 ──────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path, joint_limits: dict) -> MappingEngine:
        """从 YAML 文件加载映射配置。"""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        mappings = []
        for item in data.get("mappings", []):
            mappings.append(MappingEntry(
                input=item["input"],
                motor=item["motor"],
                control=item["control"],
                speed=item.get("speed", 1.5),
                scale=item.get("scale", 1.0),
                invert=item.get("invert", False),
            ))

        meta_data = data.get("meta_controls", {})
        meta = MetaControls(
            speed_up=meta_data.get("speed_up", "dpad_up"),
            speed_down=meta_data.get("speed_down", "dpad_down"),
            fine_tune_toggle=meta_data.get("fine_tune_toggle", "l_stick_press"),
        )

        logger.info("Loaded mapping from %s: %d entries, %d motors",
                     path, len(mappings), len(set(e.motor for e in mappings)))
        return cls(mappings, meta, joint_limits)

    @classmethod
    def default(cls, joint_limits: dict) -> MappingEngine:
        """使用内置默认映射（等价于当前硬编码行为）。"""
        return cls(_DEFAULT_MAPPINGS, MetaControls(), joint_limits)


# ── 内置默认映射 ──────────────────────────────────────

_DEFAULT_MAPPINGS = [
    MappingEntry("left_stick_x", "shoulder_pan", "incremental", speed=1.5),
    MappingEntry("left_stick_y", "shoulder_lift", "incremental", speed=1.5, invert=True),
    MappingEntry("l", "elbow_flex", "incremental", speed=1.5),
    MappingEntry("zl", "elbow_flex", "incremental", speed=1.5, invert=True),
    MappingEntry("imu_tilt", "wrist_flex", "absolute", scale=0.5),
    MappingEntry("imu_roll", "wrist_roll", "absolute", scale=0.5),
    MappingEntry("a", "gripper", "incremental", speed=3.0),
    MappingEntry("b", "gripper", "incremental", speed=3.0, invert=True),
]
```

### 计算流程图

```
每帧调用 compute_targets(input_state, current_targets):

  for each MappingEntry:
    ├── incremental:
    │     raw_value (float or bool)
    │     → × invert
    │     → × speed × speed_multiplier × fine_tune_scale
    │     → current_target + delta
    │     → clamp(lo, hi)
    │     → new target
    │
    └── absolute:
          raw_value (float, IMU degrees)
          → × scale
          → × invert
          → clamp(lo, hi)
          → new target (直接替换，不累加)
```

---

## 3. JoyConTeleop 集成

### 配置变更

```python
# configuration_joycon.py — 新增字段

@TeleoperatorConfig.register_subclass("joycon")
@dataclass
class JoyConTeleopConfig(TeleoperatorConfig):
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
    mapping_path: str | None = None   # ← 新增
```

### connect() 变更

```python
def connect(self, calibrate: bool = True,
            joint_limits: dict[str, tuple[float, float]] | None = None) -> None:
    """
    Args:
        calibrate: 是否校准 IMU
        joint_limits: 从 robot 读取的关节限位
            {"shoulder_pan": (-110, 110), ...}
    """
    # 1. 连接 Joy-Con 硬件（不变）
    # 2. IMU 校准（不变）

    # 3. 加载映射引擎（新增）
    if joint_limits is None:
        joint_limits = self._default_joint_limits()

    if self.config.mapping_path:
        self.mapping_engine = MappingEngine.from_yaml(
            self.config.mapping_path, joint_limits
        )
    else:
        self.mapping_engine = MappingEngine.default(joint_limits)

    # 初始化 meta 控制边沿检测状态
    self._prev_speed_up = False
    self._prev_speed_down = False
    self._prev_fine_tune = False
    self._current_targets = {}
```

### get_action() 变更

```python
@check_if_not_connected
def get_action(self) -> RobotAction:
    """读取 Joy-Con 输入，通过映射引擎返回舵机目标位置。"""
    self.controller.update()

    # 构建统一输入状态
    input_state = self._build_input_state()

    # 处理 meta 控制（速度/微调）
    self._handle_meta_controls(input_state)

    # 映射引擎计算目标位置
    targets = self.mapping_engine.compute_targets(
        input_state,
        self._current_targets,
        speed_multiplier=self.controller.speed_multiplier,
        fine_tune=self.controller.fine_tune,
    )
    self._current_targets = targets

    # 返回 {motor.pos: value} 格式
    return {f"{motor}.pos": pos for motor, pos in targets.items()}
```

### _build_input_state()

```python
def _build_input_state(self) -> dict[str, float | bool]:
    """从 JoyConHIDController 提取所有输入值。"""
    ctrl = self.controller
    lx, ly = ctrl.get_left_stick()
    rx, ry = ctrl.get_right_stick()
    imu_tilt, imu_roll = ctrl.get_wrist_angles()
    buttons = ctrl.buttons  # flat dict of button states

    return {
        # 摇杆 (float [-1, 1])
        "left_stick_x": lx,
        "left_stick_y": ly,
        "right_stick_x": rx,
        "right_stick_y": ry,
        # 按键 (bool)
        "a": buttons.get("a", False),
        "b": buttons.get("b", False),
        "x": buttons.get("x", False),
        "y": buttons.get("y", False),
        "l": buttons.get("l", False),
        "zl": buttons.get("zl", False),
        "r": buttons.get("r", False),
        "zr": buttons.get("zr", False),
        "dpad_up": buttons.get("up", False),
        "dpad_down": buttons.get("down", False),
        "dpad_left": buttons.get("left", False),
        "dpad_right": buttons.get("right", False),
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
```

### _handle_meta_controls()

```python
def _handle_meta_controls(self, input_state: dict) -> None:
    """处理速度调节和微调模式（边沿检测）。"""
    meta = self.mapping_engine.meta_controls

    speed_up = input_state.get(meta.speed_up, False)
    speed_down = input_state.get(meta.speed_down, False)
    fine_toggle = input_state.get(meta.fine_tune_toggle, False)

    # 边沿检测：只在 False → True 瞬间触发
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
```

### init_targets()

```python
def init_targets(self, observation: dict) -> None:
    """从 robot 当前观测初始化目标位置。connect 后、主循环前调用。"""
    self._current_targets = {}
    for motor in self.mapping_engine.motors:
        key = f"{motor}.pos"
        self._current_targets[motor] = float(observation.get(key, 0.0))
```

### action_features 变更

```python
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
```

---

## 4. Example 脚本

新脚本位于 `examples/joycon_to_so101/teleoperate.py`，约 80 行。

核心逻辑：

```python
def teleoperate(port, mode="auto", mapping_path=None, fps=30):
    # Setup
    follower = SO100Follower(SO100FollowerConfig(port=port, use_degrees=True))
    teleop = JoyConTeleop(JoyConTeleopConfig(mode=mode, mapping_path=mapping_path))

    # Connect
    teleop.connect(joint_limits=follower.get_joint_limits())
    follower.connect(calibrate=False)
    teleop.init_targets(follower.get_observation())

    # Main loop
    while not shutdown and teleop.is_connected:
        events = teleop.get_teleop_events()
        if events.get("emergency_stop"):
            break
        action = teleop.get_action()
        follower.send_action(action)
        precise_sleep(max(loop_period - dt, 0))

    # Cleanup
    teleop.disconnect()
    follower.disconnect()
```

CLI 参数：

```
--port     必填    SO-101 串口
--mode     可选    Joy-Con 模式 (auto/single_left/single_right/dual)
--mapping  可选    YAML 映射文件路径（默认使用内置映射）
--fps      可选    控制频率（默认 30）
```

去掉了旧版的 `--joint-speed` 和 `--deadzone` 参数（已在 YAML 和 config 中配置）。

---

## 5. Robot 侧接口

`SOFollower` 目前没有 `get_joint_limits()` 方法，需要新增。

参考 `rebot_b601_follower` 和 `openarm_follower` 的已有模式，在 `SOFollowerRobotConfig` 中添加 `joint_limits` 字段：

```python
# src/lerobot/robots/so_follower/config_so_follower.py

@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-110.0, 110.0),
            "shoulder_lift": (-100.0, 100.0),
            "elbow_flex": (-97.0, 97.0),
            "wrist_flex": (-95.0, 95.0),
            "wrist_roll": (-157.0, 163.0),
            "gripper": (0.0, 100.0),
        }
    )
```

同时在 `SOFollower` 类上暴露读取方法：

```python
# src/lerobot/robots/so_follower/so_follower.py

def get_joint_limits(self) -> dict[str, tuple[float, float]]:
    """返回 {motor_name: (min_deg, max_deg)} 字典。"""
    return self.config.joint_limits
```

用户可通过 YAML 或 CLI 覆盖默认限位值。

---

## 6. 测试

### MappingEngine 测试

| 测试 | 验证内容 |
|------|---------|
| `test_load_from_yaml` | YAML 解析正确 |
| `test_invalid_motor_name_raises` | 无效舵机名报错 |
| `test_invalid_input_name_raises` | 无效输入名报错 |
| `test_incremental_stick` | 摇杆增量计算正确 |
| `test_incremental_stick_inverted` | 反转方向正确 |
| `test_incremental_button_press` | 按键按下 → 增量 |
| `test_incremental_button_not_pressed` | 按键未按 → 不变 |
| `test_absolute_imu` | IMU 绝对值设定正确 |
| `test_absolute_imu_inverted` | IMU 反转正确 |
| `test_clamp_to_joint_limits` | 超限被 clamp |
| `test_multiple_inputs_same_motor` | 同舵机多输入累加 |
| `test_opposing_buttons_cancel` | L+ZL 同时按 → 不动 |
| `test_unmapped_motor_stays` | 未映射舵机保持原位 |
| `test_default_mapping` | 默认映射覆盖 6 个舵机 |
| `test_speed_multiplier` | 速度倍率生效 |
| `test_fine_tune_halves_speed` | 微调减半 |

### MetaControls 测试

| 测试 | 验证内容 |
|------|---------|
| `test_speed_up_button` | 速度 +0.2 |
| `test_speed_down_button` | 速度 -0.2 |
| `test_speed_bounds` | 不超出 [0.2, 2.0] |
| `test_speed_edge_detection` | 持续按住只触发一次 |
| `test_fine_tune_toggle` | 切换微调模式 |
| `test_custom_meta_buttons` | YAML 自定义按键生效 |

### JoyConTeleop 集成测试

| 测试 | 验证内容 |
|------|---------|
| `test_get_action_returns_motor_positions` | 返回 {motor.pos: value} |
| `test_init_targets_from_observation` | 从 observation 初始化 |
| `test_action_features_match_mapping` | features 匹配配置的舵机 |
| `test_connect_with_joint_limits` | limits 传给 MappingEngine |

现有 78 个测试保持兼容（`joycon_utils.py` 不改动）。

---

## 7. 数据流总览

```
┌─────────────────┐
│  Joy-Con 硬件   │
│  (Bluetooth HID)│
└────────┬────────┘
         │ HID 读取
         ▼
┌─────────────────────────┐
│  JoyConHIDController    │  解析原始报告
│  (joycon_utils.py)      │  → buttons, sticks, IMU
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  _build_input_state()   │  统一输入字典
│  (teleop_joycon.py)     │  {left_stick_x: 0.5, a: True, ...}
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  _handle_meta_controls() │  速度/微调 边沿检测
│  (teleop_joycon.py)     │  → 更新 controller.speed_multiplier
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  MappingEngine          │  YAML 映射规则
│  .compute_targets()     │  → incremental / absolute
│  (mapping_engine.py)    │  → clamp to joint_limits
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  JoyConTeleop           │
│  .get_action()          │  → {"shoulder_pan.pos": 45.2, ...}
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  example 脚本           │
│  follower.send_action() │  直接转发给 robot
└─────────────────────────┘
```

---

## 8. 默认映射 YAML 文件

在 `examples/joycon_to_so101/` 下提供 `default_mapping.yaml`，内容与内置 `DEFAULT_MAPPINGS` 一致。用户可以：

1. 直接使用（不指定 `--mapping`）
2. 复制 `default_mapping.yaml` → 修改后通过 `--mapping` 使用
3. 从零编写自己的 YAML

---

## 变更范围

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/lerobot/teleoperators/joycon/mapping_engine.py` | 新增 | 映射引擎核心 |
| `src/lerobot/teleoperators/joycon/configuration_joycon.py` | 修改 | +mapping_path 字段 |
| `src/lerobot/teleoperators/joycon/teleop_joycon.py` | 修改 | 集成 MappingEngine |
| `src/lerobot/teleoperators/joycon/__init__.py` | 修改 | 导出 MappingEngine |
| `examples/joycon_to_so101/teleoperate.py` | 重写 | 简化为 ~80 行 |
| `examples/joycon_to_so101/default_mapping.yaml` | 新增 | 默认映射 YAML |
| `tests/teleoperators/test_joycon.py` | 扩展 | +映射引擎/集成测试 |
| `src/lerobot/robots/so_follower/config_so_follower.py` | 修改 | +joint_limits 配置字段 |
| `src/lerobot/robots/so_follower/so_follower.py` | 修改 | +get_joint_limits() 方法 |
