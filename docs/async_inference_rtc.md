# 异步推理中的 RTC（Real-Time Chunking）实现

## 背景

Pi0/Pi0.5 策略支持 RTC 以获得更好的动作连续性——当推理较慢时，模型使用前一个动作块的剩余动作作为 flow-matching 去噪过程中的前缀引导。目前 RTC 仅在单进程 `lerobot-rollout --inference.type=rtc` 中工作。异步 PolicyServer 调用 `predict_action_chunk(observation)` 时**未传递任何 RTC 参数**，因此 RTC 处于休眠状态。

目标：在异步模式中启用 RTC，**零协议变更**——所有状态在服务器端跟踪。

---

## 架构设计

服务器维护两个动作缓冲区（镜像 `RTCInferenceEngine`）：

- **`original_queue`**（模型空间，后处理前）→ `get_left_over()` → 成为下一次推理的 `prev_chunk_left_over`
- **`queue`**（执行空间，后处理后）→ 发送给客户端 + 用于相对动作重新锚定

每次推理周期：

```
1. 从 original_queue[last_index:] 捕获 prev_chunk_left_over
2. 计算 inference_delay = ceil(latency_tracker.max() / environment_dt)
3. 运行 predict_action_chunk(obs, inference_delay, prev_chunk_left_over)
4. 将输出拆分为 original（模型空间）和 processed（执行空间）
5. ActionQueue.merge(original, processed, actual_delay, idx_before) — 跳过过时的 delay 个动作
6. 将剩余 processed 动作发送给客户端
```

---

## 文件修改

### 1. `src/lerobot/async_inference/helpers.py`

**新增导入：**
```python
from lerobot.policies.rtc.configuration_rtc import RTCConfig
```

**修改 `RemotePolicyConfig` 数据类：**
```python
@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)
    rtc_config: RTCConfig | None = None  # 新增：可选的 RTC 覆盖
```

向后兼容——现有调用者不传 `rtc_config`，默认为 `None`。

---

### 2. `src/lerobot/async_inference/configs.py`

**在 `RobotClientConfig` 中添加 RTC CLI 字段：**
```python
rtc_enabled: bool = field(default=False, metadata={"help": "Enable RTC on server"})
rtc_execution_horizon: int = field(default=10, metadata={"help": "RTC execution horizon"})
rtc_max_guidance_weight: float = field(default=10.0, metadata={"help": "RTC max guidance weight"})
```

同时在 `to_dict()` 方法中添加对应字段。

---

### 3. `src/lerobot/async_inference/robot_client.py`

**在 `__init__` 中，从 CLI 标志构建 `RTCConfig` 并传递给 `RemotePolicyConfig`：**

```python
rtc_config = None
if config.rtc_enabled:
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=config.rtc_execution_horizon,
        max_guidance_weight=config.rtc_max_guidance_weight,
    )

self.policy_config = RemotePolicyConfig(
    config.policy_type,
    config.pretrained_name_or_path,
    lerobot_features,
    config.actions_per_chunk,
    config.policy_device,
    rtc_config=rtc_config,
)
```

**客户端无需其他更改。** `_aggregate_action_queues()` 继续处理重叠的动作块。RTC 提高预测质量，聚合平滑重叠块——两者互补。

---

### 4. `src/lerobot/async_inference/policy_server.py`（核心修改）

#### 4a. 新增导入

```python
import math
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep
from lerobot.rollout.inference.rtc import (
    _normalize_prev_actions_length,
    reanchor_relative_rtc_prefix,
)
```

#### 4b. 新增实例属性（`__init__` 中）

```python
# RTC 状态（在 SendPolicyInstructions 加载策略后初始化）
self._rtc_enabled = False
self._rtc_queue: ActionQueue | None = None
self._rtc_latency_tracker: LatencyTracker | None = None
self._rtc_relative_step: RelativeActionsProcessorStep | None = None
self._rtc_normalizer_step: NormalizerProcessorStep | None = None
```

#### 4c. 新方法：`_init_rtc_state()`

在 `SendPolicyInstructions` 加载策略后调用。检查策略配置和预处理器流水线：

```python
def _init_rtc_state(self) -> None:
    """初始化服务器端 RTC 状态。"""
    rtc_config = getattr(self.policy.config, "rtc_config", None)
    if rtc_config is None or not rtc_config.enabled:
        self._rtc_enabled = False
        self.logger.info("RTC: disabled")
        return

    self._rtc_enabled = True
    self._rtc_queue = ActionQueue(rtc_config)
    self._rtc_latency_tracker = LatencyTracker()

    # 内省预处理器以获取相对动作支持
    self._rtc_relative_step = next(
        (s for s in self.preprocessor.steps
         if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
        None,
    )
    self._rtc_normalizer_step = next(
        (s for s in self.preprocessor.steps
         if isinstance(s, NormalizerProcessorStep)),
        None,
    )

    self.logger.info(
        f"RTC: enabled | execution_horizon={rtc_config.execution_horizon} | "
        f"relative_actions={self._rtc_relative_step is not None}"
    )
```

#### 4d. 新方法：`_reset_rtc_state()`

在客户端重连时从 `_reset_server()` 调用：

```python
def _reset_rtc_state(self) -> None:
    """重连时重置 RTC 状态。"""
    if self._rtc_queue is not None:
        self._rtc_queue.clear()
    if self._rtc_latency_tracker is not None:
        self._rtc_latency_tracker.reset()
```

#### 4e. 修改 `_get_action_chunk()`

接受 RTC 参数并转发给策略：

```python
def _get_action_chunk(
    self,
    observation: dict[str, torch.Tensor],
    inference_delay: int = 0,
    prev_chunk_left_over: torch.Tensor | None = None,
    execution_horizon: int | None = None,
) -> torch.Tensor:
    """获取动作块，可选带 RTC 引导。"""
    rtc_kwargs = {}
    if self._rtc_enabled:
        rtc_kwargs["inference_delay"] = inference_delay
        rtc_kwargs["prev_chunk_left_over"] = prev_chunk_left_over
        if execution_horizon is not None:
            rtc_kwargs["execution_horizon"] = execution_horizon

    chunk = self.policy.predict_action_chunk(observation, **rtc_kwargs)
    if chunk.ndim != 3:
        chunk = chunk.unsqueeze(0)
    return chunk[:, : self.actions_per_chunk, :]
```

#### 4f. 重写 `_predict_action_chunk()` — 完整 RTC 流水线

```python
def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
    """基于观测预测动作块。

    流水线：
    1. 将原始观测转换为 LeRobot 格式
    2. 捕获 RTC 状态（prev_chunk_left_over, action_index）
    3. 应用预处理器（tokenization、归一化、批处理、设备放置）
    4. 重新锚定相对动作 RTC 前缀（如适用）
    5. 将 prev_actions 长度归一化到 execution_horizon
    6. 运行带 RTC 参数的策略推理
    7. 保存模型空间动作（original_actions）
    8. 应用后处理 → 执行空间动作（processed_actions）
    9. RTC: 合并到队列，跳过 delay 个动作，提取要发送的动作
    10. 转换为带修正时间戳的 TimedAction 列表
    """
```

关键改动：
- **步骤 2**：推理前捕获 `idx_before` 和 `get_left_over()`
- **步骤 4-5**：重新锚定相对动作前缀（如使用相对动作），归一化前缀长度
- **步骤 6**：从 `latency_tracker.max()` 计算 `rtc_delay`，传递给 `_get_action_chunk()`
- **步骤 7-8**：将推理输出拆分为模型空间（`original_actions`）和执行空间（`processed_actions`）
- **步骤 9**：`ActionQueue.merge()` 存储两个缓冲区，跳过 `delay` 个过时动作
- **步骤 10**：用跳过的动作数偏移时间步和时间戳

#### 4g. 在 `SendPolicyInstructions` 中接线

```python
# 可选：从客户端覆盖策略的 rtc_config
if policy_specs.rtc_config is not None:
    self.policy.config.rtc_config = policy_specs.rtc_config
    if hasattr(self.policy, "init_rtc_processor"):
        self.policy.init_rtc_processor()

self._init_rtc_state()
```

#### 4h. 在 `_reset_server()` 中接线

```python
def _reset_server(self) -> None:
    ...
    self._reset_rtc_state()  # 新增
```

---

### 5. `tests/async_inference/test_policy_server.py`

新增 5 个 RTC 测试：

| 测试 | 验证内容 |
|------|----------|
| `test_rtc_disabled_by_default` | RTC 默认关闭 |
| `test_rtc_init_with_rtc_policy` | 策略有 rtc_config 时正确初始化 |
| `test_rtc_queue_tracks_leftover` | ActionQueue 正确跟踪剩余动作 |
| `test_rtc_delay_skip` | delay 正确跳过动作数 |
| `test_predict_action_chunk_with_rtc` | 端到端验证 RTC 参数传递 |

---

## 使用方式

### 启动服务器（无需更改）

```bash
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8088 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=0.1
```

### 启动客户端（启用 RTC）

```bash
python -m lerobot.async_inference.robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=my_awesome_follower_arm \
    --task="grasp orange" \
    --server_address=127.0.0.1:8088 \
    --policy_type=pi0 \
    --pretrained_name_or_path=/root/gpufree-data/outputs/pi0_training/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --rtc_enabled=True \
    --rtc_execution_horizon=10 \
    --rtc_max_guidance_weight=10.0
```

### 验证

1. 服务器日志应显示：`RTC: enabled | execution_horizon=10 | relative_actions=False/True`
2. 推理期间日志应显示：`RTC: delay=N | leftover=M | send=K`
3. 客户端正常接收并执行动作

### 回归测试

不使用 RTC 时（`--rtc_enabled=False` 或不传），行为与之前完全相同。

---

## 关键设计不变量

1. **向后兼容**：当 `rtc_config` 为 None 或 `enabled=False` 时，行为与原代码字节级相同
2. **双缓冲区跟踪**：服务器维护模型空间动作（`original_queue`）用于 `prev_chunk_left_over`，和执行空间动作（`queue`）用于发送给客户端和重新锚定
3. **服务器不调用 `queue.get()`**：`ActionQueue` 仅用作状态跟踪，不作为消费队列
4. **客户端无感知 RTC**：客户端接收 `list[TimedAction]` 与之前相同，只是块可能更短（delay 个动作被跳过）
5. **零协议变更**：`services.proto` 未修改，RTC 状态不跨越网络边界
6. **相对动作重新锚定**：使用与 `RTCInferenceEngine` 完全相同的 `reanchor_relative_rtc_prefix` 辅助函数
