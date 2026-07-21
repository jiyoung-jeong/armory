from __future__ import annotations

import logging
import multiprocessing
import time
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from evaluation.runtime import subscriber as _subscriber
from evaluation.sims.libero.episodes import Episode

if TYPE_CHECKING:
    from evaluation.sims.libero.env import LiberoSimEnvironment


logger = logging.getLogger(__name__)


class ProgressSubscriber(_subscriber.Subscriber):
    """
    Subscriber that sends progress updates through a multiprocessing queue.

    This subscriber:
    - Tracks step progress
    - Sends updates every N steps (configurable)
    - Reports success/failure from environment

    Designed to be created once per episode
    """

    def __init__(
        self,
        queue: multiprocessing.Queue,
        robot_idx: int,
        episode: Episode,
        environment: LiberoSimEnvironment,
        update_frequency: int = 10,
    ):
        """
        Initialize the progress subscriber.

        Args:
            queue: Multiprocessing queue for sending progress messages
            robot_idx: Worker's assigned robot index
            job_info: Dict with task_suite_name, task_id, num_episodes
            environment: LiberoSimEnvironment for accessing success flag (can be
                updated later via ``self.environment = env`` before each episode)
            update_frequency: Send update every N steps
        """
        self.queue = queue
        self.robot_idx = robot_idx
        self.episode = episode
        self.environment = environment
        self.update_frequency = update_frequency

        # State tracking
        self.current_step_count = 0
        self.total_successes = 0
        self.step_times: list[list[float]] = []

        # Send worker init message
        self._send_message(
            {
                "type": "worker_init",
                "robot_idx": robot_idx,
                "episode": episode,
            }
        )

    def _send_message(self, message: dict):
        """Send a message to the queue (non-blocking)."""
        try:
            self.queue.put_nowait(message)
        except Exception:
            # Don't let queue failures crash the worker
            # In production, might want to log this
            pass

    @override
    def on_episode_start(self) -> None:
        """Called when an episode starts."""
        self.current_step_count = 0
        self.step_times.append([])

        self._send_message(
            {
                "type": "episode_start",
                "robot_idx": self.robot_idx,
                "episode": self.episode,
            }
        )

    def _calculate_steps_per_sec(self) -> float:
        """Calculate steps per second for a robot."""
        if not self.step_times or len(self.step_times[-1]) <= 1:
            return 0.0
        intervals = np.concatenate([np.diff(step_times) for step_times in self.step_times])
        steps_per_sec = 1.0 / float(np.mean(intervals))
        return steps_per_sec

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        """Called on each step. Send update every N steps."""
        self.current_step_count += 1
        self.step_times[-1].append(time.perf_counter())
        # Only send update every N steps to reduce queue traffic
        if self.current_step_count % self.update_frequency == 0:
            self._send_message(
                {
                    "type": "step_batch",
                    "robot_idx": self.robot_idx,
                    "episode": self.episode,
                    "step_count": self.current_step_count,
                    "steps/s": self._calculate_steps_per_sec(),
                }
            )

    @override
    def on_episode_end(self) -> None:
        """Called when an episode ends. Report success/failure."""
        self._send_message(
            {
                "type": "episode_end",
                "robot_idx": self.robot_idx,
                "episode": self.episode,
                "success": self.environment.current_success,
            }
        )
