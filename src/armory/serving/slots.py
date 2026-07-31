from __future__ import annotations

import multiprocessing as mp
import pickle

import numpy as np

from armory.serving.schemas import RobotID, SlotData

MAX_OBS_BYTES = 10 * 1024 * 1024  # 10MB per slot, enough for a few 224x224 images


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

    def register(self, robot_id: RobotID) -> int:
        idx = self._free.pop()
        self._robot_to_slot[robot_id] = idx
        return idx

    def slot_for(self, robot_id: RobotID) -> int:
        return self._robot_to_slot[robot_id]

    def has_robot(self, robot_id: RobotID) -> bool:
        return robot_id in self._robot_to_slot

    def write(self, slot_idx: int, data: SlotData) -> None:
        self._slots[slot_idx].write(data)

    def read(self, slot_idx: int) -> SlotData:
        return self._slots[slot_idx].read()

    def free(self, robot_id: RobotID, expected_idx: int | None = None) -> None:
        """Release the slot held by ``robot_id``.

        When ``expected_idx`` is provided, only that slot index is recycled,
        and the ``robot_id`` mapping is removed only if it still points to
        ``expected_idx``. This makes the call safe against races where a
        newer connection has re-registered the same ``robot_id`` before this
        (older) connection's disconnect handler runs — common when a single
        server is reused across sequential client runs with the same robot
        ids (e.g. an interactive sweep that keeps serve.py up between cases).
        """
        if expected_idx is None:
            idx = self._robot_to_slot.pop(robot_id, None)
            if idx is not None:
                self._free.append(idx)
            return

        current = self._robot_to_slot.get(robot_id)
        if current == expected_idx:
            del self._robot_to_slot[robot_id]
        # If `current` differs, a newer connection has re-registered with a
        # different slot — leave that newer mapping alone but still recycle
        # the slot this stale handler owned.
        self._free.append(expected_idx)
