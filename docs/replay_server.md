# LeRobot ReplayServer — 使用指南与原理

## 1. 概述

ReplayServer 是一个 gRPC 服务器，能够读取 `lerobot-record` 录制的数据集，并通过与 `PolicyServer` 完全相同的 gRPC 协议将录制的动作发送回 `RobotClient`，从而实现**运动数据回放**——无需策略模型，无需推理。

```
┌─────────────────┐         gRPC          ┌─────────────────┐
│   RobotClient    │  ────────────────→    │  ReplayServer   │
│  (本地机器人端)   │                       │  (数据集回放)    │
│                  │  ←────────────────    │                 │
└─────────────────┘                       └─────────────────┘
   发送观测 (Observation)                     读取数据集 action
   接收动作 (Action)                          返回录制好的 action
   控制真实机械臂
```

## 2. 启动方式

### 启动 ReplayServer

```shell
python -m lerobot.async_inference.replay_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --dataset.root=/path/to/dataset \
    --dataset.episode=0

# 回放 3 次后停止
python -m lerobot.async_inference.replay_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --dataset.root=/path/to/dataset \
    --dataset.episode=0 \
    --num_repeats=3
```

### 启动 RobotClient（与连接 PolicyServer 完全一致）

```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=my_awesome_follower_arm \
    --task="replay" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=dummy/path \
    --policy_device=cpu \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only
```

> **关键**：RobotClient 的配置与连接真实 PolicyServer 时完全相同，无需任何修改。客户端只认 gRPC 接口，不关心后端是策略推理还是数据集回放。

## 3. 配置参数

### ReplayServer 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `localhost` | 监听地址 |
| `--port` | `8080` | 监听端口 |
| `--fps` | `30` | 回放帧率 |
| `--inference_latency` | `0.033` | 目标延迟（秒），控制返回速度 |
| `--obs_queue_timeout` | `2` | 观测队列超时 |
| `--dataset.repo_id` | 必填 | 数据集标识符，如 `{hf_username}/{dataset_name}` |
| `--dataset.root` | `""` | 本地数据集路径（录制的目录） |
| `--dataset.episode` | `0` | 回放第几个 episode（从 0 开始） |
| `--num_repeats` | `0` | 回放次数。`0` = 无限循环，`N` = 回放 N 次后停在最后一帧 |
| `--loop` *(已弃用)* | `true` | 已弃用，请使用 `--num_repeats` |

### RobotClient 关键参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `--actions_per_chunk` | `50` | 每次从服务器获取的动作数量 |
| `--chunk_size_threshold` | `0.5` | action_queue 剩余低于此比例时才发送新观测 |
| `--aggregate_fn_name` | `latest_only` | 回放场景下建议用最新动作覆盖，避免新旧合并抖动 |
| `--policy_type` | 任意 | 回放模式下不实际加载模型，可填任意值 |
| `--pretrained_name_or_path` | 任意 | 同上 |

## 4. 原理

### 4.1 gRPC 接口兼容

ReplayServer 实现了与 PolicyServer 完全相同的四个 gRPC RPC：

| RPC | PolicyServer 行为 | ReplayServer 行为 |
|-----|-------------------|-------------------|
| `Ready()` | 清空状态，准备接收新客户端 | 清空回放指针到第 0 帧 |
| `SendPolicyInstructions()` | 加载策略模型到 GPU | 提取 `actions_per_chunk` 参数，不加载模型 |
| `SendObservations()` | 接收观测放入队列用于推理 | 接收观测仅记录 FPS/延迟日志 |
| `GetActions()` | 用策略模型推理，返回 action chunk | 从数据集读取录制动作，返回 action chunk |

### 4.2 数据读取流程

```
1. 客户端调用 GetActions()
2. ReplayServer 从 dataset[current_frame : current_frame + actions_per_chunk] 读取动作
3. 每个动作封装为 TimedAction(timestamp, timestep, action_tensor)
4. current_frame += actions_per_chunk
5. 如果超出 episode 长度：
   - num_repeats=0  → 回到第 0 帧循环（无限）
   - num_repeats=N  → 如果已完成 N 次回放，停在最后一帧；否则回到第 0 帧继续
6. pickle 序列化后通过 gRPC 返回
```

### 4.3 动作格式

数据集中存储的 action 是**扁平的 float32 向量**，形状为 `(N_motors,)`：

```
action = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
```

录制时（`lerobot-record`），主臂的关节位置通过 `build_dataset_frame()` 打包为平面向量存储。回放时直接按原样读取并发送。

### 4.4 客户端动作队列机制

RobotClient 维护一个本地 `action_queue`，工作流程：

```
[ReplayServer] ── action chunk (50个动作) ──→ [RobotClient]
                                                   │
                                           放入 action_queue
                                                   │
                                     ┌─────────────┘
                                     ▼
                              每帧从 queue 弹出 1 个动作
                              robot.send_action() 执行
                                     │
                          action_queue.qsize() / 50 ≤ 0.5 ?
                                     │
                              是 → 发送新观测到 ReplayServer
                              否 → 继续消费队列
```

### 4.5 完整时序图

```
ReplayServer                          RobotClient (主线程)           RobotClient (后台线程)
     │                                      │                              │
     │←──── Ready() ────────────────────────│                              │
     │──── Empty() ────────────────────────→│                              │
     │                                      │                              │
     │←──── SendPolicyInstructions() ───────│                              │
     │     (actions_per_chunk=50)           │                              │
     │──── Empty() ────────────────────────→│                              │
     │                                      │                              │
     │                                      │  ┌── 启动 action_receiver ───┤
     │                                      │  │                           │
     │                                      │  │  GetActions() ────────────→│
     │←─── Actions(50个TimedAction) ────────│                              │
     │                                      │  │  合并到 action_queue       │
     │                                      │  │                           │
     │  ┌─ 读取 dataset[frame:frame+50]     │  │                           │
     │  │  ┌─ 封装为 TimedAction 列表       │  │                           │
     │  │  └─ current_frame += 50           │  │  loop:                     │
     │  │                                   │  │    if actions_available(): │
     │  │  ┌─ GetActions() ────────────────────────────────────────────────→│
     │  └──│── Actions(...) ←──────────────────────────────────────────────│
     │     │                                                              │
     │     │  control_loop:                                               │
     │     │    1. robot.get_observation()                                │
     │     │    2. 如果 queue 不足 → SendObservations() ──────────────────→│
     │     │    3. 从 queue 弹出 action → robot.send_action()             │
     │     │    4. 精确等待 → 下一帧                                       │
     │     │                                                              │
```

## 5. 使用场景

### 5.1 验证录制数据

录制完成后，用 ReplayServer 回放数据可以验证录制是否正确、动作是否流畅。

### 5.2 调试机器人硬件

无需策略模型，直接用录制数据驱动从臂运动，验证机械臂通信和控制链路。

### 5.3 演示与展示

录制一段高质量的遥操作演示，通过 ReplayServer 自动循环回放，无需人工操作。

### 5.4 策略对比

同一数据集，对比 ReplayServer（原始录制动作）和 PolicyServer（策略推理输出）下机器人的实际运动差异。

## 6. 常见问题

### 6.1 如何确认回放的是正确的 episode？

启动 ReplayServer 后，日志中会打印以下信息：

```
INFO Loading episode 0 from repo_id=xxx/my_dataset root=/path/to/data
INFO Episode 0 loaded: 600 frames, fps=30, repeats=3 time(s)
```

确认：
- **episode 编号**：日志中显示的 episode 编号是否与你指定的一致
- **帧数**：与你录制时的帧数是否匹配
- **repeats 模式**：`infinite` = 无限循环，`N time(s)` = 回放 N 次

回放过程中，每次进入新的 repeat 会打印：
```
INFO Repeat 2/3 started (frame 600, total_limit=1800)
```

所有 repeats 完成后：
```
INFO All 3 repeat(s) completed (1800 frames). Clamping to last frame.
```

### 6.2 机器人没有运动或运动不自然

可能原因：
1. **动作值不匹配**：数据集中的 action 是电机位置（度），需要与 RobotClient 的 `action_features` 匹配。确保 `lerobot-record` 和 `RobotClient` 使用相同的机器人配置
2. **`num_repeats` 设置过小**：如果设置了 `--num_repeats=1`，回放一次后机器人会停在最后一帧
3. **`aggregate_fn_name` 设置不当**：回放场景下建议使用 `latest_only`，避免新旧动作合并导致抖动

### 6.3 日志显示 0 帧

如果日志显示 `Episode X loaded: 0 frames`，说明该 episode 没有数据。可能原因：
- 录制过程中没有采集到有效数据
- `--dataset.episode` 指定的 episode 编号不存在（数据集只有更少的 episodes）

## 7. 文件变更

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/lerobot/async_inference/replay_server.py` | 新建 | ReplayServer 主实现 |
| `src/lerobot/async_inference/configs.py` | 修改 | 新增 `ReplayServerConfig` |
| `src/lerobot/async_inference/__init__.py` | 修改 | 文档注释新增 replay_server 引用 |
