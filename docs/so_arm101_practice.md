## 安装 lerobot

首先，从源码安装，克隆仓库并进入目录：
```
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```
1. 安装 uv
```
pip install uv
```
2. 使用 uv 创建虚拟环境
uv venv --python 3.12
输出如下：
```
Using CPython 3.12.13
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```
激活虚拟环境
```
source .venv/bin/activate
```

然后，将库安装为可编辑模式。如果你打算为代码做出贡献，或者使用大模型分析源码，这非常有用。
```
uv pip install -e .
```

3. 安装 ffmpeg

LeRobot默认使用TorchCodec进行视频解码，这需要ffmpeg。
从 PyTorch >= 2.10（TorchCodec ≥ 0.10）开始，TorchCodec 可以动态链接到系统范围的 ffmpeg 安装。这在使用UV或其他非conda环境管理器时非常有用：

复制
```
# Ubuntu/Debian

Sudo apt 安装 ffmpeg

# macOS（苹果硅）

brew 安装 ffmpeg
```
系统范围的ffmpeg仅支持PyTorch >= 2.10（TorchCodec ≥ 0.10）。对于较旧的 PyTorch 版本，你必须使用 conda install ffmpeg -c conda-forge。



## 使用机械臂

找到每个臂对应的USB接口
```
lerobot-find-port
```

从动臂

从动臂设置舵机的ID和波特率
```
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181
```


从动臂校准
```
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181  \
    --robot.id=my_awesome_follower_arm 
```

主动臂设置舵机的ID和波特率
```   
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5B7B0141781
```

主动臂校准
```
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5B7B0141781 \
    --teleop.id=my_awesome_leader_arm
```


遥操作
uv pip install -e ".[viz]"
```
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --teleop.port=/dev/tty.usbmodem5B7B0141781 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```

录制episode（没有使用摄像头）
```
hf auth whoami

export HF_USER="test"



lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5B7B0141781 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=false \
    --dataset.root=/Users/edison/myprojects/lerobot/data/my_dataset2 \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=20 \
    --dataset.num_episodes=2 \
    --dataset.single_task="Grab the black cube" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=False
```
录制episode（带有摄像头）

```
export HF_USER="edisonwd"

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5B7B0141781 \
    --teleop.id=my_awesome_leader_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --display_data=true \
    --dataset.root=/Users/edison/myprojects/lerobot/data/Grasp_OrangeToPlate_Dataset \
    --dataset.repo_id=${HF_USER}/Grasp_OrangeToPlate_Dataset \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=20 \
    --dataset.num_episodes=50 \
    --dataset.single_task="把橘子放到盘子里面" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false
```


收集数据的技巧

一旦你熟悉了数据记录，就可以创建更大的数据集用于训练。一个好的起始任务是抓住不同位置的物体并放入箱子里。我们建议录制至少50集，每个地点录制10集。保持摄像头固定，并在录制过程中保持一致的抓取行为。还要确保你操作的物体在摄像头上是可见的。一个不错的经验法则是：只看相机的画面，你应该能自己完成任务。

在接下来的章节中，你将训练你的神经网络。在获得稳定抓取性能后，你可以在数据收集过程中引入更多变化，比如增加抓握位置、不同的抓取技巧以及调整摄像机位置。

避免过快加入过多变化，否则可能会影响效果。



## 可视化整个数据集
- huggingface提供的可视化数据集工具：https://huggingface.co/spaces/lerobot/visualize_dataset

- 由艾欧智能开发的本地 LeRobot 数据可视化检查工具：http://io-ai.tech/lerobot


重放录制的episode
```
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.id=my_awesome_follower_arm \
    --dataset.root=/Users/edison/myprojects/lerobot/data/my_dataset \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.episode=0
```


上传数据集到 huggingface 

```
hf upload ${HF_USER}/Grasp_OrangeToPlate_Dataset /Users/edison/myprojects/lerobot/data/Grasp_OrangeToPlate_Dataset --repo-type dataset

```


## 模型训练

```
lerobot-train \
  --dataset.repo_id=username/hf_act_record \
  --policy.type=act \
  --output_dir=outputs/train/hf_act_record0 \
  --job_name=hf_act_training_job \
  --policy.device=cuda \
  --wandb.enable=False \
  --policy.repo_id=username/hf_act_recordpolicy0 \
  --batch_size=8 \
  --steps=20000
```


```
lerobot-train \
  --dataset.repo_id=/root/lerobot/Grasp_OrangeToPlate_Dataset \
  --policy.type=act \
  --output_dir=outputs/train/hf_act_record0 \
  --job_name=hf_act_training_job \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=/root/lerobot/act_policy \
  --batch_size=8 \
  --steps=20000
```

## 重放录制数据（ReplayServer + RobotClient）

适用于重放已录制的 episode，**不需要策略模型**。

启动服务器（ReplayServer）：
```

# 使用推理服务重放 3 次后停止

python -m lerobot.async_inference.replay_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --dataset.root=/Users/edison/myprojects/lerobot/data/my_dataset \
    --dataset.episode=0 \
    --num_repeats=3
```

启动客户端
```
python -m lerobot.async_inference.robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
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

```
python -m lerobot.async_inference.robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.id=my_awesome_follower_arm \
    --task="graph orange into plate" \
    --server_address=127.0.0.1:8088 \
    --policy_type=act \
    --pretrained_name_or_path=/root/lerobot/outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```


```
python -m lerobot.async_inference.robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.id=my_awesome_follower_arm \
    --task="graph orange into plate" \
    --server_address=127.0.0.1:8088 \
    --policy_type=pi0 \
    --pretrained_name_or_path=/root/.cache/modelscope/hub/models/lerobot/pi0_base \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```

## 模型推理（使用训练好的策略控制机械臂）

**注意：** `lerobot-record` **不支持** `--policy.path` 参数。用训练好的策略控制真实机械臂，应使用 **`lerobot-rollout`** 命令。

`lerobot-rollout` 是 LeRobot 统一的策略部署工具，支持多种运行模式（strategy）和推理后端（inference backend）。

### 基本模式（Base）— 自动执行策略，不录制

最简命令：

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=/Users/edison/myprojects/lerobot/outputs/outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=my_awesome_follower_arm \
    --task="grasp orange" \
    --duration=60
```

### 带摄像头可视化（Rerun）

```bash
uv pip install -e ".[viz]"

lerobot-rollout \
    --strategy.type=base \
    --policy.path=/Users/edison/myprojects/lerobot/outputs/outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=my_awesome_follower_arm \
    --task="grasp orange" \
    --duration=60 \
    --display_data=true
```

### 使用 RTC 推理（适合慢速 VLA 模型，如 Pi0、SmolVLA）

```bash
lerobot-rollout \
    --strategy.type=base \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=10 \
    --inference.rtc.max_guidance_weight=10.0 \
    --policy.path=/Users/edison/myprojects/lerobot/outputs/outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --task="grasp orange" \
    --duration=60
```

### 录制模式 — Sentry（持续录制+自动上传）

```bash
lerobot-rollout \
    --strategy.type=sentry \
    --strategy.upload_every_n_episodes=5 \
    --policy.path=/Users/edison/myprojects/lerobot/outputs/outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=${HF_USER}/rollout_sentry_data \
    --dataset.single_task="grasp orange" \
    --duration=3600
```

### 录制模式 — Episodic（按集录制+重置阶段）

```bash
lerobot-rollout \
    --strategy.type=episodic \
    --policy.path=/Users/edison/myprojects/lerobot/outputs/outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5B7B0141781 \
    --dataset.repo_id=${HF_USER}/rollout_episodic_data \
    --dataset.num_episodes=20 \
    --dataset.single_task="grasp orange"
```

### 远程服务器推理（SSH 隧道模式）

**远程服务器（Ubuntu）：**
```bash
cd /root/lerobot
lerobot-rollout \
    --strategy.type=base \
    --policy.path=/root/lerobot/outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --task="grasp orange" \
    --duration=60
```

**本地电脑（macOS）通过 SSH 隧道：**
```bash
# 先建立 SSH 隧道：ssh -L 8088:127.0.0.1:8088 user@remote_server
```

### 五种策略模式说明

| 模式 | 用途 | 需要遥操作臂 |
|------|------|:---:|
| `base` | 纯自动执行，不录制 | 否 |
| `sentry` | 持续录制+定期上传到 Hub | 否 |
| `highlight` | 环形缓冲区，按 's' 保存精彩片段 | 否 |
| `dagger` | 人在环路中（DAgger/RaC），人工介入修正 | 是 |
| `episodic` | 按集录制，每集之间有重置阶段 | 可选 |

### 两种推理后端

| 后端 | 适用场景 | 说明 |
|------|----------|------|
| `sync`（默认） | ACT、Diffusion 等轻量模型 | 每个控制周期调用一次策略 |
| `rtc` | Pi0、Pi0.5、SmolVLA 等慢速 VLA | Real-Time Chunking，用前一步结果引导当前推理 |

### 两个核心超参数中文详解
#### 1. actions_per_chunk（单块动作数）
**默认值**：50，常用范围：10~50
- 含义：策略模型**单次批量输出的动作数量**。
- 影响：
  1. 调大数值：队列里始终有充足待执行动作，**不容易出现无动作可执行的空档**；
  2. 弊端：一次性预测过长序列的动作，**误差会不断累积**，最终动作精准度下降。

#### 2. chunk_size_threshold（队列阈值）
**默认值**：0.7，取值范围：0~1
- 含义：动作队列剩余容量的触发阈值，原文补充说明：**队列占用率≤50%时，客户端就会发送新的观测数据**，向策略服务请求推理。
- 影响：
  1. 调大数值：客户端**频繁发送观测数据、频繁生成新动作块**，新旧动作大量重叠。模型环境适配能力更强，但会产生大量推理请求，**加重推理链路的负载**；极端情况下每一条观测都生成一组动作。
  2. 调至接近0：变为**同步执行模式**，只有当前整批动作全部执行完毕，才会发送新观测、生成新动作。

---
### 补充总结
论文实验中默认参数表现良好，你可根据自身场景调试：
- 追求**流畅不中断**：适当加大 `actions_per_chunk`；
- 追求**动作精准**：适当减小 `actions_per_chunk`；
- 追求**环境快速适配**：加大 `chunk_size_threshold`（接受推理压力上升）；
- 追求**低负载、同步运行**：把 `chunk_size_threshold` 调低至接近0。

## 问题记录

### Teleoperation 运行约 6 分钟后崩溃: ConnectionError: There is no status packet!

**错误信息:**
```
ConnectionError: Failed to sync read 'Present_Position' on ids=[1, 2, 3, 4, 5, 6] after 1 tries.
[TxRxResult] There is no status packet!
```

**现象:**
- 运行约 6 分钟后崩溃
- Loop 速度为 253ms (4 Hz)，远低于预期的 ~16ms (60 Hz)
- 崩溃发生在 `robot.get_observation()` → `sync_read("Present_Position")`

**原因分析:**

`sync_read` 默认 `num_retry=0`，即只尝试 1 次，失败直接抛 `ConnectionError`。USB/UART 通信在长时间运行后出现偶发丢包，单次读取超时即崩溃。253ms 的 loop 时间表明通信已经严重退化。

**修复方案 (三层防御):**

1. **增加默认重试次数**: `so_follower.py` 中 `sync_read("Present_Position")` 改为 `num_retry=3`，4 次尝试容忍偶发丢包
2. **teleop_loop 容错**: 捕获 `ConnectionError`，连续 5 次失败才停止，中间帧跳过并打印警告
3. **硬件检查**: 253ms loop 时间说明通信早已不稳定，建议检查:
   - USB 线缆是否松动或质量不佳
   - 舵机电源是否稳定（电压不足会导致舵机响应变慢）
   - UART 总线连接是否可靠
   - 尝试更换 USB 端口或使用带屏蔽的线缆

### 校准不显示 wrist_roll

使用 lerobot-calibrate 校准机械臂，没有显示 wrist_roll 的内容，显示的内容如下：
```
lerobot-calibrate \                                                                                                                                                 
      --robot.type=so101_follower \                                                                                                                                            
      --robot.port=/dev/tty.usbmodem5B7B0137181  \                                                                                                                             
      --robot.id=my_awesome_follower_arm 显示的结果如下：INFO 2026-06-05 19:51:38 calibrate.py:88 {'robot': {'calibration_dir': None,                                          
             'cameras': {},                                                                                                                                                    
             'disable_torque_on_disconnect': True,                                                                                                                             
             'id': 'my_awesome_follower_arm',                                                                                                                                  
             'max_relative_target': None,                                                                                                                                      
             'port': '/dev/tty.usbmodem5B7B0137181',                                                                                                                           
             'use_degrees': True},                                                                                                                                             
   'teleop': None}                                                                                                                                                             
  INFO 2026-06-05 19:51:38 follower.py:105 my_awesome_follower_arm SOFollower connected.                                                                                       
  INFO 2026-06-05 19:51:38 follower.py:122                                                                                                                                     
  Running calibration of my_awesome_follower_arm SOFollower                                                                                                                    
  Move my_awesome_follower_arm SOFollower to the middle of its range of motion and press ENTER....                                                                             
  Move all joints except 'wrist_roll' sequentially through their entire ranges of motion.                                                                                      
  Recording positions. Press ENTER to stop...                                                                                                                                  
                                                                                                                                                                               
  -------------------------------------------                                                                                                                                  
  -------------------------------------------                                                                                                                                  
  NAME            |    MIN |    POS |    MAX                                                                                                                                   
  shoulder_pan    |    706 |   2032 |   3183                                                                                                                                   
  shoulder_lift   |    935 |    956 |   3314                                                                                                                                   
  elbow_flex      |    961 |   3177 |   3184                                                                                                                                   
  wrist_flex      |    584 |   1989 |   2616 
```

**原因分析**

因为 wrist_roll 是一个360° 全旋转舵机，没有物理行程限制，不需要像其他关节那样记录最小/最大位置。

从源码可以看到（so_follower.py:130-133）：
```
full_turn_motor = "wrist_roll"
unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
# ...
range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
range_mins[full_turn_motor] = 0
range_maxes[full_turn_motor] = 4095
```
代码显式将 wrist_roll 排除在行程记录之外，并直接赋予完整范围 [0, 4095]（舵机的 12 位分辨率，对应 4096 个位置）。

### `lerobot-record` 不支持 `--policy.path` 参数

**报错信息:**
```
lerobot-record: error: unrecognized arguments: --policy.path=...
```

**原因:**

`lerobot-record` 的 `RecordConfig` 只定义了 `robot`、`dataset`、`teleop` 等字段，没有 `policy` 字段。`lerobot-record` 是用于**遥操作数据采集**的命令，不需要策略模型。

**正确做法:**

用训练好的策略控制真实机械臂，应使用 **`lerobot-rollout`** 命令（见上方"模型推理"章节）。`lerobot-rollout` 是 LeRobot 统一的策略部署工具，支持 `base`（纯自动执行）、`sentry`（持续录制）、`dagger`（人在环路）、`episodic`（按集录制）等多种模式，以及 `sync`/`rtc` 两种推理后端。

### `lerobot-record` 键盘控制失效

**现象:** 录制时按 `→`、`←`、`Esc` 键盘无法控制录制流程（提前结束、重录、停止）。

**原因:**

键盘控制依赖 `pynput` 库，通过 `is_headless()` 检测环境（`control_utils.py:46-70`）：

```python
def is_headless():
    try:
        import pynput
        return False
    except Exception:
        # 进入 headless 模式，键盘控制不可用
        return True
```

`init_keyboard_listener()` 中：
- 如果 `pynput` 未安装 → `is_headless()` 返回 `True` → 跳过键盘监听
- 如果 macOS 未授予终端辅助功能权限 → `pynput` 启动但收不到按键事件

**键盘控制的三个关键按键:**

| 按键 | 功能 |
|------|------|
| `→` 右箭头 | 提前结束当前 episode，进入重置阶段 |
| `←` 左箭头 | 提前结束当前 episode 并重录（丢弃本集数据） |
| `Esc` | 完全停止录制，保存并退出 |

**macOS 修复步骤:**

1. **确认 pynput 已安装:**
```bash
uv run python -c "import pynput; print('OK')"
```

2. **授予终端辅助功能权限:**
   - 打开 **系统设置 → 隐私与安全性 → 辅助功能**
   - 确保你运行 `lerobot-record` 的终端应用（Terminal / iTerm2 / VS Code 等）在列表中且开关已开启
   - 如果已在列表中但不工作，先移除再重新添加

3. **macOS Sequoia (15.x) 及以上额外要求:**
如果使用的是较新的 macOS 版本，可能还需要 **输入监听 (Input Monitoring)** 权限，在同一隐私设置页面中添加终端应用

4. **替代方案（不需要键盘控制）:**
录制时完全靠时间参数控制流程：
   - `--dataset.episode_time_s=30`: 每集录制 30 秒后自动进入重置阶段
   - `--dataset.reset_time_s=20`: 重置阶段持续 20 秒
   - `--dataset.num_episodes=2`: 录制指定数量后自动停止

### SSH 隧道 + 远程服务器（ReplayServer / PolicyServer）动作不连贯

**现象:** 本地通过 SSH 隧道连接远程 replay_server，执行 `robot_client` 时机器人动作不连贯，每动一下就停顿。

**原因:**

`actions_per_chunk` 控制服务器一次发送的动作数量。当设置为 `1` 时（`--actions_per_chunk=1`），客户端每执行 1 个动作就需要等待服务器发送下一个。

控制循环流程（`robot_client.py:467-479`）：
1. 队列有动作 → 取出并执行 1 个
2. 队列空了 → 发送观察到服务器
3. 通过 SSH 隧道往返等待（50-200ms）→ 机器人停顿

`chunk_size_threshold` 决定何时发送观察：
- `queue_size / actions_per_chunk <= threshold` 时发送
- `actions_per_chunk=1, threshold=0.1` → 只有队列完全为空时才发请求
- 机器人每执行 1 个动作就停顿等待网络

**修复方案:**

增加 `actions_per_chunk`，让本地队列有足够的缓冲抵消 SSH 延迟：

```bash
# 推荐参数
python -m lerobot.async_inference.robot_client \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --robot.id=my_awesome_follower_arm \
    --task="replay" \
    --server_address=127.0.0.1:8088 \
    --policy_type=act \
    --pretrained_name_or_path=dummy/path \
    --policy_device=cpu \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=latest_only
```

参数效果：
- 服务器一次发送 50 个动作（约 1.7 秒的缓冲，50/30≈1.67s）
- 队列剩下 25 个动作时（50% 阈值），客户端提前发送下一次请求
- SSH 往返期间本地仍有 25 个动作可以执行，机器人不会停顿
- 即使 SSH 延迟 200ms，队列中仍有足够的动作维持流畅运动

关键参数说明：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `actions_per_chunk` | 50 | 一次获取的动作数量，约 1-2 秒的缓冲 |
| `chunk_size_threshold` | 0.5 | 队列剩余 50% 时提前请求下一次 |
| `aggregate_fn_name` | latest_only | 回放场景取最新动作即可 |

### `lerobot-rollout` RTC 推理后端不支持 ACT 模型

**报错信息:**
```
ERROR: ACTPolicy.predict_action_chunk() got an unexpected keyword argument 'inference_delay'
```

**原因:**

RTC (Real-Time Chunking) 推理后端在调用策略的 `predict_action_chunk` 方法时，会传入 `inference_delay` 和 `prev_chunk_left_over` 两个额外参数（`rtc.py:307-308`）：

```python
self._policy.predict_action_chunk(
    preprocessed, inference_delay=delay, prev_chunk_left_over=prev_actions
)
```

ACT 模型的方法签名为 `def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor`，**不接受 `**kwargs`**，因此收到额外参数时直接抛出 `TypeError`。

根本原因在于架构差异：
- **ACT** 是确定性回归模型，一次前向传播直接输出完整 action chunk，没有去噪循环，不需要 RTC 的前一步引导机制
- **RTC** 专为 Pi0、Pi0.5、SmolVLA 等慢速 VLA 模型设计，利用扩散/流匹配的去噪循环特性，用前一步的部分结果引导当前推理

**修复方案:**

使用 ACT 或 Diffusion 等轻量模型时，**不要指定 `--inference.type=rtc`**，使用默认的 `sync` 推理后端即可：

```bash
# 正确：ACT 使用默认 sync 后端
lerobot-rollout \
    --strategy.type=base \
    --policy.path=outputs/train/my_first_train/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B7B0137181 \
    --task="grasp orange" \
    --duration=60
```

**推理后端兼容性:**

| 策略 | sync | rtc |
|------|------|-----|
| ACT | ✅ | ❌ |
| Diffusion | ✅ | ❌ |
| Pi0 | ✅ | ✅ |
| Pi0.5 | ✅ | ✅ |
| SmolVLA | ✅ | ✅ |

### ModuleNotFoundError: No module named 'lerobot') 报错

 python -m lerobot.async_inference.replay_server     --host=127.0.0.1     --port=8088     --fps=30     --dataset.repo_id=${HF_USER}/my_dataset                               
  --dataset.root=/root/gpufree-data/my_dataset3     --dataset.episode=0     --num_repeats=3                                                                                    
  /usr/bin/python: Error while finding module specification for 'lerobot.async_inference.replay_server' (ModuleNotFoundError: No module named 'lerobot') 分析报错原因          
                                                                                                                                                                             
⏺ 原因是你使用的 /usr/bin/python（系统自带 Python）中没有安装 lerobot 模块。路径 /root/gpufree-data/my_dataset3 看起来是在 Linux 服务器上运行，而 lerobot 需要先通过 uv        
  安装到虚拟环境中。                                                                                                                                                         
                                                                                                                                                                               
  解决方法：用 uv run 来执行，它会自动使用安装过 lerobot 的环境：                                                                                                              
   
  cd /root/lerobot                                                                                                                                                             
  uv run python -m lerobot.async_inference.replay_server \                                                                                                                     
      --host=127.0.0.1 \                                                                                                                                                       
      --port=8088 \                                                                                                                                                            
      --fps=30 \                                                                                                                                                               
      --dataset.repo_id=${HF_USER}/my_dataset \                                                                                                                                
      --dataset.root=/root/gpufree-data/my_dataset3 \                                                                                                                          
      --dataset.episode=0 \                                                                                                                                                    
      --num_repeats=3                                                                                                                                                          
                                                                                                                                                                               
  如果还没有从源码安装，需要先：                                                                                                                                               
                                                                                                                                                                               
  git clone https://github.com/huggingface/lerobot.git                                                                                                                         
  cd lerobot                                                                                                                                                                   
  pip install uv                                                                                                                                                               
  uv pip install -e .                                                                                                                                                          
  uv run python -m lerobot.async_inference.replay_server ...      



## 参考文档
1. https://huggingface.co/docs/lerobot/so101
2. https://huggingface.co/docs/lerobot/il_robots
3. https://huggingface.co/lerobot/smolvla_base

