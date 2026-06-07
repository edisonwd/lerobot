
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
    --dataset.root=/Users/edison/myprojects/lerobot/data/my_dataset2 \
    --dataset.repo_id=${HF_USER}/VisGrasp_OrangeToPlate_Dataset \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=20 \
    --dataset.num_episodes=2 \
    --dataset.single_task="把橘子放到盘子里面" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false
```

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


```

# 回放 3 次后停止
python -m lerobot.async_inference.replay_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --dataset.root=/path/to/dataset \
    --dataset.episode=0 \
    --num_repeats=3

python -m lerobot.async_inference.replay_server \
    --host=127.0.0.1 \
    --port=8080 \
    --fps=30 \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --dataset.root=/Users/edison/myprojects/lerobot/data/my_dataset \
    --dataset.episode=0 \
    --num_repeats=3
```

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




## 参考文档
1. https://huggingface.co/docs/lerobot/so101
2. https://huggingface.co/docs/lerobot/il_robots
3. https://huggingface.co/lerobot/smolvla_base

