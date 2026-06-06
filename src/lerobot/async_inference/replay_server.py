# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replay server: reads pre-recorded actions from a LeRobot dataset and serves them
to a RobotClient over gRPC, enabling motion replay without policy inference.

Example:
```shell
# Infinite repeat (default)
python -m lerobot.async_inference.replay_server \\
     --host=127.0.0.1 \\
     --port=8080 \\
     --fps=30 \\
     --dataset.repo_id=${HF_USER}/my_dataset \\
     --dataset.root=/Users/edison/myprojects/lerobot/data/my_dataset \\
     --dataset.episode=0

# Replay exactly 3 times, then stop
python -m lerobot.async_inference.replay_server \\
     --host=127.0.0.1 \\
     --port=8080 \\
     --fps=30 \\
     --dataset.repo_id=${HF_USER}/my_dataset \\
     --dataset.root=/Users/edison/myprojects/lerobot/data/my_dataset \\
     --dataset.episode=0 \\
     --num_repeats=3
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat

import draccus
import grpc
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks

from .configs import ReplayServerConfig
from .helpers import (
    FPSTracker,
    TimedAction,
    get_logger,
)


class ReplayServer(services_pb2_grpc.AsyncInferenceServicer):
    """A gRPC server that replays recorded actions from a LeRobot dataset.

    Implements the same AsyncInferenceServicer interface as PolicyServer,
    so RobotClient can connect without any modification. Instead of running
    policy inference, this server reads pre-recorded actions from a dataset
    and serves them sequentially to the client.
    """

    prefix = "replay_server"
    logger = get_logger(prefix)

    def __init__(self, config: ReplayServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        # Determine the root directory for the dataset
        dataset_root = config.dataset.root if config.dataset.root else None

        # Validate episode index against total metadata first
        meta_dataset = LeRobotDataset(repo_id=config.dataset.repo_id, root=dataset_root)
        total_episodes = meta_dataset.num_episodes
        if config.dataset.episode >= total_episodes:
            raise ValueError(
                f"Episode index {config.dataset.episode} out of range "
                f"(dataset has {total_episodes} episodes, indexed 0-{total_episodes - 1})"
            )

        # Load only the target episode — indexing becomes 0-based within this episode
        # download_videos=False: we only need actions, not camera images
        self.logger.info(
            f"Loading episode {config.dataset.episode} from "
            f"repo_id={config.dataset.repo_id} root={dataset_root}"
        )
        self.dataset = LeRobotDataset(
            repo_id=config.dataset.repo_id,
            root=dataset_root,
            episodes=[config.dataset.episode],
            download_videos=False,
        )
        self.episode_length = self.dataset.num_frames
        self.dataset_fps = self.dataset.fps

        num_repeats = config.num_repeats
        repeat_desc = "infinite (loop forever)" if num_repeats == 0 else f"{num_repeats} time(s)"
        self.logger.info(
            f"Episode {config.dataset.episode} loaded: {self.episode_length} frames, "
            f"fps={self.dataset_fps}, repeats={repeat_desc}"
        )

        if self.episode_length == 0:
            raise ValueError(
                f"Episode {config.dataset.episode} has 0 frames. "
                f"The dataset may be corrupted or empty."
            )

        # Replay state: current frame index within the episode (0-based)
        self._state_lock = threading.Lock()
        self.current_frame = 0
        self._repeats_done_logged = False
        self._last_logged_repeat = -1

        # Actions per chunk (will be set by SendPolicyInstructions)
        self.actions_per_chunk = 50  # default, overridden by client

        # FPS metrics
        self.last_receive_time = 0.0

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def environment_dt(self) -> float:
        return 1 / self.config.fps

    @property
    def _completed_repeats(self) -> int:
        """Number of complete episode replays so far."""
        return self.current_frame // self.episode_length

    def _reset_server(self) -> None:
        """Flushes server state when a new client connects."""
        self.shutdown_event.set()
        with self._state_lock:
            self.current_frame = 0
            self._repeats_done_logged = False
            self._last_logged_repeat = -1
        self.logger.info("Server state reset")

    def Ready(self, request, context):  # noqa: N802
        """Handshake from RobotClient indicating it is ready."""
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy configuration from the robot client.

        The replay server does not load a policy model, but accepts this call
        to remain compatible with the RobotClient protocol. The only field we
        extract is actions_per_chunk from the RemotePolicyConfig.
        """
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec

        # Extract actions_per_chunk if available
        if hasattr(policy_specs, "actions_per_chunk") and policy_specs.actions_per_chunk:
            self.actions_per_chunk = policy_specs.actions_per_chunk
            self.logger.info(
                f"Received policy instructions from {client_id} | "
                f"Actions per chunk: {self.actions_per_chunk}"
            )
        else:
            self.logger.info(
                f"Received policy instructions from {client_id} | "
                f"Using default actions_per_chunk: {self.actions_per_chunk}"
            )

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client.

        The replay server does not use observations for inference (it replays
        pre-recorded actions), but accepts them to remain compatible with the
        RobotClient protocol.
        """
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()
        try:
            received_bytes = receive_bytes_in_chunks(
                request_iterator, None, self.shutdown_event, self.logger
            )
            timed_observation = pickle.loads(received_bytes)  # nosec

            obs_timestep = timed_observation.get_timestep()

            # Calculate FPS metrics
            fps_metrics = self.fps_tracker.calculate_fps_metrics(
                timed_observation.get_timestamp()
            )

            self.logger.debug(
                f"Received observation #{obs_timestep} | "
                f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                f"Target: {fps_metrics['target_fps']:.2f} | "
                f"One-way latency: {(receive_time - timed_observation.get_timestamp()) * 1000:.2f}ms"
            )

            self.last_receive_time = receive_time

        except Exception as e:
            self.logger.debug(f"Error receiving observation: {e}")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns a chunk of pre-recorded actions to the robot client.

        Reads the next actions_per_chunk actions from the dataset starting at
        current_frame, wraps them as TimedAction objects, and returns the
        serialized list.
        """
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} requesting actions")

        try:
            with self._state_lock:
                actions = self._get_action_chunk()
                self.current_frame += self.actions_per_chunk

                num_repeats = self.config.num_repeats

                if num_repeats == 0:
                    # Infinite repeat: wrap to avoid integer overflow on long sessions
                    if self.current_frame >= self.episode_length * 2:
                        self.current_frame = self.current_frame % self.episode_length
                else:
                    # Finite repeats: clamp once all repeats are exhausted
                    total_limit = num_repeats * self.episode_length
                    if self.current_frame >= total_limit:
                        self.current_frame = total_limit - 1
                        if not self._repeats_done_logged:
                            self.logger.info(
                                f"All {num_repeats} repeat(s) completed "
                                f"({num_repeats * self.episode_length} frames). "
                                f"Clamping to last frame."
                            )
                            self._repeats_done_logged = True

                # Log at episode boundaries
                current_repeat = self._completed_repeats
                if current_repeat > self._last_logged_repeat and current_repeat < num_repeats:
                    self.logger.info(
                        f"Repeat {current_repeat + 1}/{num_repeats} started "
                        f"(frame {self.current_frame}, total_limit={num_repeats * self.episode_length})"
                    )
                    self._last_logged_repeat = current_repeat

            # Serialize and return
            actions_bytes = pickle.dumps(actions)  # nosec
            return services_pb2.Actions(data=actions_bytes)

        except Exception as e:
            self.logger.error(f"Error in GetActions: {e}")
            return services_pb2.Empty()

    def _get_action_chunk(self) -> list[TimedAction]:
        """Read the next actions_per_chunk actions from the dataset.

        Returns a list of TimedAction objects with timestamps derived from
        the dataset's fps. When loop is enabled and the chunk crosses the
        episode boundary, actions seamlessly wrap around to the start.
        """
        actions = []

        for i in range(self.actions_per_chunk):
            absolute_frame = self.current_frame + i

            if self.config.num_repeats == 0:
                # Infinite: always wrap
                frame_idx = absolute_frame % self.episode_length
            elif absolute_frame < self.config.num_repeats * self.episode_length:
                # Within configured repeats: wrap within episode boundaries
                frame_idx = absolute_frame % self.episode_length
            else:
                # Past all configured repeats: clamp to last frame
                frame_idx = self.episode_length - 1

            frame = self.dataset[frame_idx]
            action_tensor = frame["action"]  # shape (N_motors,)

            if isinstance(action_tensor, torch.Tensor):
                action_tensor = action_tensor.detach().cpu()
            else:
                action_tensor = torch.tensor(action_tensor, dtype=torch.float32)

            # Timestamp: current wall-clock time + offset for each action in chunk
            timestamp = time.time() + i / self.dataset_fps

            timed_action = TimedAction(
                timestamp=timestamp,
                timestep=absolute_frame,  # monotonically increasing across repeats
                action=action_tensor,
            )
            actions.append(timed_action)

        return actions

    def stop(self):
        """Stop the server."""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def replay_serve(cfg: ReplayServerConfig):
    """Start the ReplayServer with the given configuration."""
    logging.info(pformat(asdict(cfg)))

    replay_server = ReplayServer(cfg)

    # Setup and start gRPC server with increased message size limits
    # (observations with camera images can exceed the default 4MB limit)
    server_options = [
        ("grpc.max_send_message_length", 100 * 1024 * 1024),    # 100MB
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=server_options,
    )
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(replay_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    replay_server.logger.info(f"ReplayServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()

    replay_server.logger.info("Server terminated")


if __name__ == "__main__":
    replay_serve()
