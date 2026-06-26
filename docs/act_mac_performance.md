# ACT 模型在 Mac 上推理速度慢（6.7 Hz）问题分析

## 现象

```
WARNING ... Record loop is running slower (6.7 Hz) than the target FPS (30.0 Hz).
Dataset frames might be dropped and robot control might be unstable.
```

## 根因分析

### 核心原因：`temporal_ensemble_coeff` 导致逐帧全量推理

当 `temporal_ensemble_coeff` 不为 `None` 时（ACT 默认训练配置通常为 `0.01`），**每个 control tick 都会执行完整的 ResNet18 + Transformer 前向传播**，而不是利用 action queue 缓存。

**文件位置：** `src/lerobot/policies/act/modeling_act.py:110-113`

```python
if self.config.temporal_ensemble_coeff is not None:
    actions = self.predict_action_chunk(batch)  # 每个 tick 都执行完整前向传播
    action = self.temporal_ensembler.update(actions)
    return action
```

**正常行为（ensemble 关闭时）：**
- 每 `n_action_steps=100` 个 tick 推理一次完整 action chunk
- 其余 99 个 tick 直接从 `deque` 队列弹出缓存动作，**零推理开销**
- 实际推理频率：30 Hz / 100 = 0.3 Hz

**异常行为（ensemble 开启时）：**
- 每个 tick 都运行完整前向传播
- 实际推理频率：30 Hz（30 次全量推理/秒）
- Mac MPS 上单次 ResNet18+Transformer 推理约 150ms → 6.7 Hz

### 次要瓶颈

| # | 瓶颈 | 文件位置 | 影响 |
|---|------|----------|------|
| 1 | `torch.compile` 在 MPS 上无效 | `context.py:210-219` | Triton 内核不支持 MPS，编译回退到 eager 模式 |
| 2 | 每帧冗余预处理 | `sync.py:97-122` | 即使从队列弹缓存，numpy→tensor→device 转换仍每帧执行 |
| 3 | MPS 传输非阻塞仅支持 CUDA | `device_processor.py:72` | MPS 上的 `.to(device)` 总是阻塞式 |
| 4 | 图像 `.contiguous()` 强制内存拷贝 | `utils.py:128-130` | 1920×1080 图像每帧 permute + contiguous 开销大 |
| 5 | 高分辨率图像 | ResNet18 输入 | 1920×1080 → resize 后仍比 640×480 多 6 倍像素 |

## 优化命令

### 步骤 1：检查当前策略配置

```bash
cat /Users/edison/myprojects/lerobot/outputs/outputs/train/my_first_train/checkpoints/last/pretrained_model/config.json | grep temporal_ensemble_coeff
```

- 如果输出是 `null` → ensemble 已关闭，看其他优化
- 如果输出是数字（如 `0.01`）→ 这是根因，用方案 2

### 方案 2：关闭 temporal ensembling（推荐，效果最大）

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=/Users/edison/myprojects/lerobot/outputs/outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --policy.temporal_ensemble_coeff=null \
    --policy.n_action_steps=100 \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=my_awesome_follower_arm \
    --task="grasp orange" \
    --duration=600
```

预期效果：推理频率从 30 Hz 降到 ~0.3 Hz，整体循环恢复到 30 Hz。

### 方案 3：插值降频推理（平衡推理频率与控制平滑度）

```bash
lerobot-rollout \
    ... \
    --interpolation_multiplier=2 \
    ...
```

| `interpolation_multiplier` | 推理频率 | 控制频率 | 说明 |
|---|---|---|---|
| 1（默认） | 30 Hz | 30 Hz | 每帧都推理 |
| 2 | 15 Hz | 30 Hz | 每隔一帧推理，中间帧线性插值 |
| 3 | 10 Hz | 30 Hz | 适合 Mac MPS |
| 4 | 7.5 Hz | 30 Hz | 最低推荐值 |

### 方案 4：降低图像分辨率

```bash
--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}"
```

| 分辨率 | 像素数 | 相对于 1920×1080 | 预期推理加速 |
|---|---|---|---|
| 1920×1080 | 2,073,600 | 1× | 基准 |
| 1280×720 | 921,600 | 0.44× | ~1.5× |
| 640×480 | 307,200 | 0.15× | ~3× |
| 320×240 | 76,800 | 0.037× | ~5× |

### 方案 5：组合优化（推荐用于 Mac MPS）

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=/Users/edison/myprojects/lerobot/outputs/outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --policy.temporal_ensemble_coeff=null \
    --policy.n_action_steps=100 \
    --interpolation_multiplier=2 \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=my_awesome_follower_arm \
    --task="grasp orange" \
    --duration=600
```

**预期效果：** 关闭 ensemble（100× 推理减少）+ 降低分辨率（~3× 加速）+ 插值（2× 冗余保护）→ 目标 30 Hz 可达成。

## 代码路径参考

```
lerobot-rollout entry point:  src/lerobot/scripts/lerobot_rollout.py:233
  → rollout() at line 198
    → build_rollout_context()
      → create_strategy().run()

Control loop:  src/lerobot/rollout/strategies/base.py:55-77
  → send_next_action() at core.py:269-304
    → engine.get_action(obs_frame)
      → sync.py:115 → policy.select_action()
        → modeling_act.py:110-113 → temporal_ensembler 分支
        → modeling_act.py:117-118 → action queue 分支（正常缓存路径）
```

## 配置参数速查

### ACT 策略参数（`configuration_act.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 100 | 每次推理输出的动作数量 |
| `n_action_steps` | 100 | 每次使用 chunk 中的动作数，决定缓存长度 |
| `temporal_ensemble_coeff` | None | 非 None 时强制每帧推理（性能杀手） |
| `n_encoder_layers` | 4 | Transformer 编码器层数 |
| `n_decoder_layers` | 1 | Transformer 解码器层数 |
| `dim_model` | 512 | Transformer 模型维度 |

### Rollout 参数（`configs.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fps` | 30.0 | 目标控制频率 |
| `device` | auto | 设备选择（Mac 上自动选 mps） |
| `interpolation_multiplier` | 1 | 插值倍率，>1 减少推理频率 |
| `use_torch_compile` | False | torch.compile（MPS 上无效） |
