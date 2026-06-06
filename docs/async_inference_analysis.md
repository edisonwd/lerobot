# LeRobot Async Inference — PolicyServer 与 RobotClient 通信分析

## 1. 概述

`async_inference` 模块实现了**远程推理**架构：策略网络（Policy）运行在 GPU 服务器上，真实机器人运行在本地客户端，两者通过 **gRPC** 网络通信。

```
┌────────────────────┐         gRPC          ┌────────────────────┐
│   RobotClient       │  ────────────────→    │   PolicyServer     │
│  (本地机器人端)      │                       │  (GPU 推理服务器)   │
│                     │  ←────────────────    │                    │
└────────────────────┘                       └────────────────────┘
    发送观测 (Observation)                        返回动作 (Action)
    接收动作 (Action)                             加载策略模型
    控制真实机械臂
```

**关键设计目标**：低延迟、高频控制（默认 30Hz），支持网络环境下的推理部署。

---

## 2. gRPC 通信协议

定义在 `src/lerobot/transport/services.proto`：

```protobuf
service AsyncInference {
  // 初始化握手：客户端 → 服务器
  rpc Ready(Empty) returns (Empty);

  // 发送策略配置：客户端 → 服务器（模型路径、类型、设备等）
  rpc SendPolicyInstructions(PolicySetup) returns (Empty);

  // 流式发送观测：客户端 → 服务器（支持分块传输，超 4MB 大图）
  rpc SendObservations(stream Observation) returns (Empty);

  // 获取动作：客户端 ← 服务器（返回动作块）
  rpc GetActions(Empty) returns (Actions);
}
```

所有数据体均使用 **pickle 序列化**，通过 `bytes data` 字段传输。

---

## 3. 通信时序图

```
RobotClient (线程1: 控制循环)          PolicyServer                    RobotClient (线程2: 动作接收)
        │                                  │                                  │
        │──── Ready() ────────────────────→│                                  │
        │←──── Empty() ────────────────────│                                  │
        │                                  │                                  │
        │──── SendPolicyInstructions() ───→│                                  │
        │     (policy_type, path, device)  │                                  │
        │                                  │                                  │
        │                                  │   加载模型 from_pretrained()      │
        │                                  │   创建 pre/post processor         │
        │←──── Empty() ────────────────────│                                  │
        │                                  │                                  │
        │                                  │                                  │
  ┌─────┴─────────────────────────────────────────────────────────────────────┐
  │                         控制循环开始 (双线程并行)                           │
  │                                                                          │
  │  [控制循环线程]                         │  [动作接收线程]                  │
  │                                       │                                  │
  │  1. 读取机器人观测                     │                                  │
  │     robot.get_observation()           │                                  │
  │                                       │                                  │
  │  2. 封装为 TimedObservation            │                                  │
  │     (timestamp, timestep, must_go)    │                                  │
  │                                       │                                  │
  │  3. SendObservations() ───────────────│───────────────────────────────→  │
  │     (流式分块传输，支持大图)            │                                  │
  │                                       │  4. 接收观测，反序列化             │
  │                                       │     记录 FPS、延迟统计             │
  │                                       │                                  │
  │                                       │  5. 放入 observation_queue       │
  │                                       │     (带过滤：跳过重复/已预测的)    │
  │                                       │                                  │
  │                                       │  6. 运行策略推理                   │
  │                                       │     _predict_action_chunk()       │
  │                                       │     → 返回 Action Chunk           │
  │                                       │                                  │
  │  7. GetActions() ←────────────────────│───────────────────────────────   │
  │     接收 Actions (pickled bytes)      │                                  │
  │                                       │                                  │
  │  8. 反序列化 → list[TimedAction]      │                                  │
  │                                       │                                  │
  │  9. _aggregate_action_queues()        │                                  │
  │     合并新旧动作块（加权平均等策略）    │                                  │
  │                                       │                                  │
  │  10. control_loop_action()            │                                  │
  │      从 action_queue 弹出 action      │                                  │
  │      robot.send_action() 执行动作     │                                  │
  └─────┬─────────────────────────────────────────────────────────────────────┘
        │
        ↓  循环持续直到 shutdown_event 被设置
```

---

## 4. 关键组件详解

### 4.1 客户端架构 (RobotClient)

客户端运行**两个并行线程**：

| 线程 | 职责 |
|------|------|
| **控制循环线程** (主线程) | 读取机器人观测 → 发送到服务器 → 从 action_queue 弹出动作 → 执行 |
| **动作接收线程** (`action_receiver_thread`) | 持续调用 `GetActions()` → 接收服务器返回的动作块 → 合并到 action_queue |

两个线程通过 `threading.Barrier(2)` 同步启动，确保在开始控制循环前已经建立了与服务器的连接。

#### 核心数据结构

```python
# 每个动作附带时间戳和时间步，用于跨网络对齐
@dataclass
class TimedAction(TimedData):
    timestamp: float    # Unix 时间戳 (time.time())
    timestep: int       # 离散时间步
    action: torch.Tensor

# 观测同样附带时间信息
@dataclass
class TimedObservation(TimedData):
    timestamp: float
    timestep: int
    observation: dict   # 原始观测数据
    must_go: bool       # 是否强制送入推理（action_queue 为空时设为 True）
```

#### 动作队列机制 (`action_queue`)

客户端维护一个 `Queue` 缓存未来若干步的动作。关键机制：

1. **Chunk 缓冲**：服务器一次性返回 K 个动作（action chunk），客户端将其放入队列。
2. **动作聚合** (`_aggregate_action_queues`)：当新的动作块到达时，对于相同的 timestep，按聚合函数合并新旧动作。默认使用**加权平均**：`0.3 * old + 0.7 * new`（更信任新动作）。
3. **可用聚合函数**：
   - `weighted_average`: `0.3*old + 0.7*new` (默认)
   - `latest_only`: 只用最新动作
   - `average`: `0.5*old + 0.5*new`
   - `conservative`: `0.7*old + 0.3*new` (更保守)

#### 观测发送策略 (`chunk_size_threshold`)

客户端并非每帧都发送观测，而是根据 `chunk_size_threshold` 控制：

```python
def _ready_to_send_observation(self):
    # 当 action_queue 剩余量 / 块大小 ≤ threshold 时才发送新观测
    return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold
```

默认 `threshold=0.5`，即当 action_queue 剩余不到一半时才发送新观测。这避免了频繁调用推理，让网络带宽集中在需要的时刻。

此外，当 `action_queue` 完全为空时，`must_go=True` 强制发送观测，确保不会因为队列耗尽而停滞。

---

### 4.2 服务器架构 (PolicyServer)

服务器是 gRPC servicer，接收多个 RPC 请求，核心逻辑在 `GetActions` 中。

#### 观测队列 (`observation_queue`)

```python
self.observation_queue = Queue(maxsize=1)  # 只保留最新一帧
```

只保留**最新的观测**，如果队列满了则丢弃旧的。这样确保推理始终基于最新状态，而不是过时的观测。

#### 观测过滤 (`_obs_sanity_checks`)

服务器会跳过以下观测：
1. **已预测过的 timestep**：避免对同一帧重复推理
2. **与上次预测的观测过于相似**：如果关节位置差异很小（范数 < 1），跳过推理，节省计算资源

```python
def _enqueue_observation(self, obs: TimedObservation) -> bool:
    # 只有 must_go=True 或观测足够新、足够不同时才入队
    if obs.must_go or self.last_processed_obs is None or self._obs_sanity_checks(obs, self.last_processed_obs):
        # 入队...
        return True
    return False
```

#### 推理管线 (`_predict_action_chunk`)

完整的推理流程：

```
1. raw_observation_to_observation()
   将机器人原始观测 {motor.pos: value, camera: numpy} 
   转为 LeRobot 格式 {observation.state: tensor, observation.images.<cam>: tensor}

2. preprocessor(observation)
   - 图像缩放：resize 到策略模型要求的分辨率
   - tokenization / normalization
   - 添加 batch 维度
   - 移动到指定设备 (GPU/CPU)

3. policy.predict_action_chunk(observation)
   运行神经网络推理，返回 (B, chunk_size, action_dim)

4. postprocessor 逐帧处理
   - 反归一化 (unnormalization)
   - 从 GPU 移回 CPU
   - detach() 避免保留梯度

5. _time_action_chunk()
   将 tensor chunk 转为 list[TimedAction]，附加时间戳和时间步
```

#### 延迟控制

`GetActions` 返回前会 sleep 以控制推理延迟：

```python
time.sleep(max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts)))
```

这确保从接收请求到返回动作的时间**不低于**目标 `inference_latency`（默认 33ms ≈ 30Hz），防止推理返回过快导致客户端节奏紊乱。

---

## 5. 观测数据转换

机器人与策略模型使用**不同的观测格式**，转换发生在服务器端：

### 机器人格式 (RawObservation)
```python
{
    "shoulder_pan.pos": 150.5,
    "shoulder_lift.pos": -30.2,
    "elbow_flex.pos": 45.0,
    "wrist_flex.pos": ...,
    "wrist_roll.pos": ...,
    "gripper.pos": ...,
    "front": np.ndarray,          # 摄像头图像 (H, W, C) uint8
    "task": "fold my tshirt",     # 自然语言指令 (VLA 策略需要)
}
```

### LeRobot 格式 (Observation)
```python
{
    "observation.state": torch.tensor([150.5, -30.2, 45.0, ...]),
    "observation.images.front": torch.tensor(...),  # (B, C, H, W) float32 [0,1]
    "task": "fold my tshirt",
}
```

转换由 `prepare_raw_observation()` 完成，包括：
- 将分散的电机位置合并为 `observation.state` 向量
- 将图像从 `(H, W, C)` 转为 `(B, C, H, W)` 并缩放到模型要求的分辨率
- 添加 `task` 字段给 VLA 策略使用

---

## 6. 使用方式

### 启动 PolicyServer

```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```

### 启动 RobotClient

```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

---

## 7. 对比遥操作模式

| 维度 | 遥操作 (`lerobot-teleoperate`) | 异步推理 (`async_inference`) |
|------|-------------------------------|-----------------------------|
| 控制来源 | 人类拖动主臂 | 策略网络推理 |
| 通信方式 | 直接 USB 串口 | gRPC 网络 |
| 架构 | 单机 | 客户端-服务器分离 |
| 动作生成 | 主臂关节位置 → 从臂 | 观测 → 神经网络 → 动作块 |
| 线程模型 | 单线程循环 | 双线程并行（控制循环 + 动作接收） |
| 延迟敏感 | 本地延迟（ms 级） | 网络延迟 + 推理延迟 |
| 动作缓冲 | 无 | action_queue 缓存未来动作 |
| 适用场景 | 数据采集、手动控制 | 部署训练好的策略模型 |

---

## 8. 总结

Async Inference 的核心设计思想：

1. **动作块缓冲**：服务器一次性返回 K 步动作，客户端本地消费，减少网络往返对延迟的影响
2. **双线程解耦**：控制循环负责读观测、执行动作；动作接收线程负责从服务器获取动作。两者通过 `action_queue` 解耦
3. **按需推理**：通过 `chunk_size_threshold` 和 `must_go` 机制，只在 action_queue 不足时发送观测，避免浪费网络带宽和 GPU 算力
4. **动作聚合**：新旧动作块按 timestep 加权合并，平滑网络抖动带来的影响
5. **观测去重**：服务器端跳过相似观测和已预测时间步，减少冗余推理
