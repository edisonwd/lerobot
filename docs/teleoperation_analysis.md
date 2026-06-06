# LeRobot 遥操作 (lerobot-teleoperate) 源码分析

## 整体架构

遥操作系统的核心是一个**闭环控制循环**，数据流如下：

```
主臂 (Leader) ──→ 读取关节位置 ──→ 处理管线 ──→ 从臂 (Follower) ──→ 执行目标位置
```

入口命令：`lerobot-teleoperate`，定义在 `src/lerobot/scripts/lerobot_teleoperate.py`。

---

## 1. 入口点与初始化

`teleoperate()` 函数初始化三个关键组件：

```python
teleop = make_teleoperator_from_config(cfg.teleop)   # 主臂（操作臂）— 读取操作者关节位置
robot  = make_robot_from_config(cfg.robot)           # 从臂（跟随臂）— 接收指令并运动
(teleop_action_processor,
 robot_action_processor,
 robot_observation_processor) = make_default_processors()  # 处理管线
```

然后连接设备并进入 `teleop_loop()`，以 `fps`（默认 60Hz）的频率循环运行。

---

## 2. 主循环 (teleop_loop, line 131–209)

每帧执行以下步骤：

```python
# 1. 获取从臂当前状态（观测）
obs = robot.get_observation()  # 读取从臂电机实际位置 + 摄像头图像

# 2. (可选) 向主臂发送反馈 — 仅 Unitree G1 使用
if robot.name == "unitree_g1":
    teleop.send_feedback(obs)

# 3. 读取主臂关节位置（核心！）
raw_action = teleop.get_action()

# 4. 处理主臂动作（默认不做任何变换）
teleop_action = teleop_action_processor((raw_action, obs))

# 5. 处理发送给从臂的动作（默认不做任何变换）
robot_action_to_send = robot_action_processor((teleop_action, obs))

# 6. 将目标位置发送给从臂电机
_ = robot.send_action(robot_action_to_send)

# 7. 精确等待，维持目标帧率
precise_sleep(max(1 / fps - dt_s, 0.0))
```

循环一直运行直到 `KeyboardInterrupt` 或达到 `teleop_time_s` 时限。

---

## 3. 主臂如何读取位置 (`so_leader.py:146–152`)

```python
@check_if_not_connected
def get_action(self) -> dict[str, float]:
    start = time.perf_counter()
    action = self.bus.sync_read("Present_Position")  # 从6个Feetech舵机同时读取当前位置
    action = {f"{motor}.pos": val for motor, val in action.items()}
    return action
```

关键点：
- 使用 Feetech 舵机的 **同步读取** 指令，一次性读取所有 6 个关节的当前位置
- 6 个关节：`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`
- 主臂在连接时**关闭扭矩** (`self.bus.disable_torque()`)，人可以自由拖动主臂
- 舵机设为位置模式 (`OperatingMode.POSITION`)，用于校准
- 返回格式：`{"shoulder_pan.pos": 150.5, "shoulder_lift.pos": -30.2, ...}`

---

## 4. 从臂如何执行 (`so_follower.py:196–221`)

```python
@check_if_not_connected
def send_action(self, action: RobotAction) -> RobotAction:
    # 提取目标位置
    goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

    # 可选：限制最大相对运动幅度（安全保护）
    if self.config.max_relative_target is not None:
        present_pos = self.bus.sync_read("Present_Position")
        goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

    # 通过同步写入将目标位置发送到所有电机
    self.bus.sync_write("Goal_Position", goal_pos)
    return {f"{motor}.pos": val for motor, val in goal_pos.items()}
```

关键点：
- 从臂舵机设置为**位置模式** (`OperatingMode.POSITION`)，接收到 `Goal_Position` 后自动运动到目标角度
- 可选的 `max_relative_target` 参数限制每次运动的幅度，防止突然的大幅跳动
- `ensure_safe_goal_position()` 会检查目标位置与当前位置的差异，超出阈值则截断

---

## 5. 硬件通信层 (FeetechMotorsBus)

所有舵机通过**串行总线** (UART/USB) 连接。关键方法：

| 方法 | 作用 |
|------|------|
| `sync_read("Present_Position")` | 向多个电机发送同步读指令，一次性获取所有关节当前位置 |
| `sync_write("Goal_Position", goals)` | 向多个电机发送同步写指令，一次性设置所有关节目标位置 |

`sync_write` 不需要等待每个电机单独响应，适合高频控制循环。底层依赖 `scservo_sdk`（Feetech SDK，基于 Dynamixel SDK）。

---

## 6. 处理管线 (Processor Pipeline)

默认的三个处理管线使用 `IdentityProcessorStep`，即不做任何变换，直接传递数据。

但也存在更复杂的处理步骤（可选启用），定义在 `src/lerobot/robots/so_follower/robot_kinematic_processor.py`：

| 处理步骤 | 作用 |
|----------|------|
| `EEReferenceAndDelta` | 将相对增量命令转换为末端执行器目标位姿 |
| `EEBoundsAndSafety` | 限制工作空间边界，防止大幅度跳跃 |
| `InverseKinematicsEEToJoints` | 逆运动学：从末端执行器 Cartesian 位姿计算关节角度 |
| `ForwardKinematicsJointsToEE` | 正运动学：从关节角度计算末端执行器位姿 |
| `GripperVelocityToJoint` | 将夹爪速度积分为目标位置 |

---

## 7. 总结：遥操作的本质

**核心机制极其直接**：主臂每个关节的当前位置 → 直接映射为从臂对应关节的目标位置。

```
主臂 shoulder_pan 当前位置 150.5°  ──→  从臂 shoulder_pan 目标位置 150.5°
主臂 shoulder_lift 当前位置 -30.2° ──→  从臂 shoulder_lift 目标位置 -30.2°
主臂 elbow_flex 当前位置 45.0°     ──→  从臂 elbow_flex 目标位置 45.0°
...
```

两个机械臂使用**相同型号的舵机**、**相同的关节命名**、**相同的坐标系统**，因此不需要任何坐标变换，关节位置可以直接 1:1 传递。

### 关键条件

1. **同构硬件**：主臂和从臂使用相同的舵机型号（sts3215），相同的机械结构
2. **校准对齐**：通过校准文件确保主臂和从臂的零位和运动范围一致
3. **位置模式**：所有舵机工作在位置伺服模式，主臂关闭扭矩可被拖动，从臂开启扭矩跟随运动
4. **高频控制**：以 60Hz 的频率循环读取-发送，保证低延迟响应

### 控制时序

```
每帧时序：
├─ 读取从臂观测 (get_observation)
├─ 读取主臂位置 (get_action)        ← 人手拖动主臂
├─ 处理管线 (默认 Identity)          ← 可选：运动学/安全限制
├─ 发送给从臂 (send_action)          ← 从臂运动到目标位置
└─ precise_sleep (维持 60Hz)
```
