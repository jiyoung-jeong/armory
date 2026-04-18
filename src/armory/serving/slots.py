from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import pickle
from typing import Any

import numpy as np

MAX_OBS_BYTES = 10 * 1024 * 1024  # 10MB per slot, enough for a few 224x224 images


@dataclass
class SlotData:
    """Observation and request metadata written together atomically into a slot.

    This ensures that when the GPU worker reads a slot, the metadata (timestamps,
    step, etc.) always corresponds to the observation being inferred, even if the
    slot was overwritten by a newer request after the SlotRequest was enqueued.
    """

    obs: dict
    request_id: int
    arrival_timestamp: float
    observation_step: int
    action_start_step: int
    request_timestamp: float
    deadline: float
    execution_horizon: int  # how many steps of the predicted chunk the robot is willing to execute
    infer_type: Any
    params: Any
    noise: Any  # np.ndarray | None


class RobotSlot:
    def __init__(self):
        self._buf = mp.RawArray("B", MAX_OBS_BYTES)  # 'B' = uint8
        self._size = mp.RawValue("i", 0)
        self._lock = mp.Lock()
        self._np_buf = np.frombuffer(self._buf, dtype=np.uint8)  # zero-copy view

    def write(self, data: SlotData) -> None:
        raw = pickle.dumps(data)
        n = len(raw)
        with self._lock:
            self._np_buf[:n] = np.frombuffer(raw, dtype=np.uint8)
            self._size.value = n

    def read(self) -> SlotData:
        with self._lock:
            raw = bytes(self._np_buf[: self._size.value])  # copy under lock
        return pickle.loads(raw)  # unpickle outside lock


class RobotSlots:
    """Pre-allocated slots shared across fork. Created in main process before forking."""

    def __init__(self, max_robots: int):
        self._slots = [RobotSlot() for _ in range(max_robots)]
        self._free: list[int] = list(range(max_robots))
        # robot_id→slot_index mapping lives only in WS main process — scheduler never accesses slot assignments
        self._robot_to_slot: dict[str, int] = {}

    def register(self, robot_id: str) -> int:
        idx = self._free.pop()
        self._robot_to_slot[robot_id] = idx
        return idx

    def slot_for(self, robot_id: str) -> int:
        return self._robot_to_slot[robot_id]

    def has_robot(self, robot_id: str) -> bool:
        return robot_id in self._robot_to_slot

    def write(self, slot_idx: int, data: SlotData) -> None:
        self._slots[slot_idx].write(data)

    def read(self, slot_idx: int) -> SlotData:
        return self._slots[slot_idx].read()

    def free(self, robot_id: str) -> None:
        idx = self._robot_to_slot.pop(robot_id)
        self._free.append(idx)
