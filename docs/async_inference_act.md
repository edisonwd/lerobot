# LeRobot 异步推理（ACT 模型）常见问题

## PolicyServer + RobotClient 无法控制机械臂

### 症状

- `lerobot-rollout --strategy.type=base` 能正常控制机械臂
- 但使用 PolicyServer（远程服务器）+ RobotClient（本地 mac）时，机械臂不动
- Client 和 Server 都启动成功，无报错

### 典型命令

**远程服务器（PolicyServer）：**
```bash
uv run python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=8088 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=2
```

**本地 mac（RobotClient）—— 有问题的版本：**
```bash
uv run python -m lerobot.async_inference.robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.id=my_awesome_follower_arm \
    --task="graph orange into plate" \
    --server_address=127.0.0.1:8088 \
    --policy_type=act \
    --pretrained_name_or_path=/root/lerobot/outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```

### 原因分析

`lerobot-rollout` 是**单进程本地推理**，策略和机器人在同一进程中直接通信，不需要网络传输和特征序列化。

PolicyServer + RobotClient 是**分离式架构**，观测特征必须通过 `RemotePolicyConfig.lerobot_features` 在两端对齐。以下三个问题会导致推理失败但无明显报错：

#### 1. Client 未配置 `--robot.cameras`

如果 ACT 模型训练时使用了图像数据，Client 端机器人必须配置摄像头。否则：
- `map_robot_keys_to_lerobot_features(robot)` 提取的特征中**没有图像键**
- Client 将不含图像的特征列表发送给 Server
- Server 端 ACT 模型的 `config.image_features` 声明了需要图像
- 预处理器找不到对应的图像键，推理静默失败（或动作队列为空）

**对比：**
```
# lerobot-rollout 有摄像头配置（能工作）
--robot.cameras="{ front: {type: opencv, index_or_path: 0, ...} }"

# robot_client 没有摄像头配置（不工作）
# ← 缺少 --robot.cameras
```

#### 2. `--pretrained_name_or_path` 路径错误

模型在 **Server 端加载**，路径必须是 Server 文件系统上的路径：
```bash
# 正确：远程服务器路径（Client 只是把这个字符串发给 Server）
--pretrained_name_or_path=/root/lerobot/outputs/train/my_first_train/checkpoints/last/pretrained_model

# 错误：本地 mac 路径（Server 上不存在这个路径）
--pretrained_name_or_path=/Users/edison/myprojects/lerobot/outputs/...
```

#### 3. SSH 隧道未建立

Server 绑定 `--host=127.0.0.1`（仅本机可访问），本地 Client 必须通过 SSH 隧道：
```bash
ssh -L 8088:127.0.0.1:8088 user@remote_server
```
未建立隧道时，Client 连接 `127.0.0.1:8088` 实际连接的是本机，而本机没有运行 Server。

### 修复方案

**本地 mac（RobotClient）—— 修复后的版本：**
```bash
uv run python -m lerobot.async_inference.robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=my_awesome_follower_arm \
    --task="graph orange into plate" \
    --server_address=127.0.0.1:8088 \
    --policy_type=act \
    --pretrained_name_or_path=/root/lerobot/outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

### 排查清单

| 检查项 | 命令/方法 |
|--------|-----------|
| SSH 隧道 | `ssh -L 8088:127.0.0.1:8088 user@remote` 是否运行中 |
| Client 摄像头 | 命令中是否有 `--robot.cameras="{...}"` |
| 模型路径 | Server 上 `ls /root/lerobot/outputs/train/.../pretrained_model` 是否存在 |
| Server 日志 | 查看 Server 端终端是否有报错 |
| 特征对齐 | Client 日志中查看 `lerobot_features` 是否包含 `observation.image.front` |

### 机械臂运动非常慢，每动一下就停顿

**症状：** 机械臂能动，但每动一下就停一下，运动不连贯，整体速度远慢于预期。

**原因：** `actions_per_chunk` 太小 + `chunk_size_threshold` 太低，导致动作缓冲不足。

**问题机制：**

```
actions_per_chunk=20 → 每次只获取 20 个动作（20/30Hz ≈ 0.67s 缓冲）
  ↓
chunk_size_threshold=0.3 → 队列剩 30%（即只剩 6 个动作）时才请求下一次
  ↓
SSH 隧道延迟 50-200ms → 新动作块到达前，旧动作已用完
  ↓
动作队列空了 → Client 等待新动作 → 机械臂停顿
  ↓
新动作到了 → 动几下 → 又空了 → 又停顿
```

**修复：增大缓冲，提前请求**

```bash
# 修改前（慢）
--actions_per_chunk=20 \
--chunk_size_threshold=0.3

# 修改后（流畅）
--actions_per_chunk=50 \
--chunk_size_threshold=0.5
```

**原理：**
- `actions_per_chunk=50` → 50/30Hz ≈ 1.67s 的动作缓冲
- `chunk_size_threshold=0.5` → 队列剩 25 个动作时就发起下一次请求
- 即使 SSH 往返延迟 200ms，队列中仍有约 19 个动作可执行，机械臂不会停顿

**参数调优参考：**

| 参数 | 推荐值 | 缓冲时间 | 适用场景 |
|------|--------|----------|----------|
| `actions_per_chunk=50` | 默认推荐 | ~1.67s | SSH 隧道、一般网络延迟 |
| `actions_per_chunk=30` | 低延迟场景 | ~1.0s | 本地网络（无 SSH 隧道） |
| `actions_per_chunk=20` | 过小 | ~0.67s | 不推荐，容易卡顿 |
| `chunk_size_threshold=0.5` | 默认推荐 | 剩 50% 时请求 | 平衡流畅度和推理负载 |
| `chunk_size_threshold=0.3` | 过低 | 剩 30% 才请求 | 不推荐，容易在请求响应前耗尽 |
| `chunk_size_threshold=0.7` | 较高 | 剩 70% 就请求 | 更频繁推理，环境适应性更强，但增加 Server 负载 |

### `lerobot-rollout` vs PolicyServer + RobotClient

| 方面 | `lerobot-rollout` | PolicyServer + RobotClient |
|------|-------------------|---------------------------|
| 架构 | 单进程 | 双进程（gRPC） |
| 模型位置 | 本地 | Server 端 |
| 特征传输 | 内存直接传递 | 序列化 → 网络 → 反序列化 |
| 摄像头配置 | 必需 | 必需（且必须与训练时一致） |
| 网络依赖 | 无 | SSH 隧道或直连 |
| 适用场景 | 本地推理 | 远程 GPU 推理 + 本地机械臂 |
