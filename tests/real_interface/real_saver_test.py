"""Unit tests for RealSaver — verifies on-disk layout matches sim Saver."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from armory_client.runtime.real_saver import RealSaver, Result, _robot_idx_from_id
from armory_client.schemas import Action, ActionChunk, Observation


class _FakeBroker:
    """Minimal stand-in for ActionChunkBroker."""

    def __init__(self) -> None:
        self.action_chunks: list[ActionChunk] = []
        self.actions_left_history: list[int] = []

    def push_chunk(self, request_timestamp: float, observation_step: int = 0) -> None:
        chunk = ActionChunk(
            chunk_id=len(self.action_chunks),
            observation_step=observation_step,
            action_index_start=0,
            execution_start_step=observation_step,
            actions=np.zeros((1, 7), dtype=np.float32),
            max_execution_horizon=1,
            request_timestamp=request_timestamp,
            response_timestamp=request_timestamp + 0.05,
            noise=None,
        )
        self.action_chunks.append(chunk)
        self.actions_left_history.append(1)


def _obs(step: int) -> Observation:
    return Observation(
        state=np.array([float(step)] * 7, dtype=np.float32),
        step=step,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        wrist_image=np.zeros((4, 4, 3), dtype=np.uint8),
    )


def _action(step: int, chunk_idx: int | None = None) -> Action:
    return Action(
        step=step,
        action=np.zeros(7, dtype=np.float32),
        action_chunk_index=chunk_idx,
        index_in_chunk=0 if chunk_idx is not None else None,
    )


# ── helpers ────────────────────────────────────────────────────


def test_robot_idx_extracts_trailing_number():
    assert _robot_idx_from_id("robot_11") == 11
    assert _robot_idx_from_id("robot_007") == 7
    assert _robot_idx_from_id("ws-3-foo") == 3
    assert _robot_idx_from_id("noname") == 0


# ── lifecycle + on-disk layout ─────────────────────────────────


def _drive_episode(saver: RealSaver, broker: _FakeBroker, n_steps: int = 3) -> None:
    """Push one chunk per step and call on_step. Used by every layout test."""
    saver.on_episode_start()
    for step in range(n_steps):
        chunk_idx = len(broker.action_chunks)
        broker.push_chunk(request_timestamp=time.time() - 0.05, observation_step=step)
        saver.on_step(_obs(step), _action(step, chunk_idx=chunk_idx))
    saver.on_episode_end()
    saver.close()


def test_writes_expected_episode_layout(tmp_path):
    broker = _FakeBroker()
    saver = RealSaver(
        out_dir=tmp_path,
        robot_id="robot_11",
        prompt="pick the cube",
        control_hz=20,
        action_chunk_broker=broker,
    )

    _drive_episode(saver, broker, n_steps=3)

    # Robot dir + episode dir
    robot_dir = tmp_path / "11"
    episodes = list(robot_dir.iterdir())
    assert len(episodes) == 1, f"expected 1 episode dir, got {episodes}"

    ep = episodes[0]
    assert ep.name == "0_real_0_failure", ep.name

    # Required files for calculate_metrics
    expected = {
        "metadata.json",
        "timestamps.csv",
        "action_chunks.parquet",
        "actions_left.npy",
        "cost_history.npy",
        "cost_history.png",
    }
    actual = {p.name for p in ep.iterdir()}
    missing = expected - actual
    assert not missing, f"missing files: {missing}; got {actual}"


def test_metadata_json_matches_result_schema(tmp_path):
    broker = _FakeBroker()
    saver = RealSaver(
        out_dir=tmp_path,
        robot_id="robot_42",
        prompt="sort the legos",
        control_hz=20,
        action_chunk_broker=broker,
        success_default=True,
    )

    _drive_episode(saver, broker, n_steps=5)

    meta_path = tmp_path / "42" / "0_real_0_success" / "metadata.json"
    raw = json.loads(meta_path.read_text())
    assert raw == {
        "robot_idx": 42,
        "success": True,
        "steps_taken": 5,
        "task_suite_name": "real",
        "task_id": 0,
        "task_language": "sort the legos",
        "episode_idx": 0,
    }

    # And it parses back as a Result.
    result = Result.from_json(meta_path)
    assert result.robot_idx == 42
    assert result.steps_taken == 5
    assert result.task_language == "sort the legos"


def test_timestamps_row_count_matches_steps(tmp_path):
    broker = _FakeBroker()
    saver = RealSaver(
        out_dir=tmp_path,
        robot_id="robot_3",
        prompt="x",
        control_hz=20,
        action_chunk_broker=broker,
    )

    _drive_episode(saver, broker, n_steps=4)

    csv = (tmp_path / "3" / "0_real_0_failure" / "timestamps.csv").read_text().splitlines()
    # header + 4 rows
    assert len(csv) == 5


def test_cost_history_shape_matches_steps(tmp_path):
    broker = _FakeBroker()
    saver = RealSaver(
        out_dir=tmp_path,
        robot_id="robot_3",
        prompt="x",
        control_hz=20,
        action_chunk_broker=broker,
    )

    _drive_episode(saver, broker, n_steps=7)

    costs = np.load(tmp_path / "3" / "0_real_0_failure" / "cost_history.npy")
    assert costs.shape == (7,)
    assert np.all(costs >= 0)


def test_actions_left_shape_matches_steps(tmp_path):
    broker = _FakeBroker()
    saver = RealSaver(
        out_dir=tmp_path,
        robot_id="robot_3",
        prompt="x",
        control_hz=20,
        action_chunk_broker=broker,
    )

    _drive_episode(saver, broker, n_steps=6)

    al = np.load(tmp_path / "3" / "0_real_0_failure" / "actions_left.npy")
    assert al.shape == (6,)


def test_close_blocks_until_writes_complete(tmp_path):
    broker = _FakeBroker()
    saver = RealSaver(
        out_dir=tmp_path,
        robot_id="robot_5",
        prompt="x",
        control_hz=20,
        action_chunk_broker=broker,
    )

    saver.on_episode_start()
    for step in range(2):
        broker.push_chunk(request_timestamp=time.time() - 0.01, observation_step=step)
        saver.on_step(_obs(step), _action(step))
    saver.on_episode_end()
    # close() must wait for the background ThreadPoolExecutor to drain — after
    # it returns, the metadata file must already exist on disk.
    saver.close()

    assert (tmp_path / "5" / "0_real_0_failure" / "metadata.json").is_file()


def test_episode_counter_increments_across_calls(tmp_path):
    broker = _FakeBroker()
    saver = RealSaver(
        out_dir=tmp_path,
        robot_id="robot_9",
        prompt="x",
        control_hz=20,
        action_chunk_broker=broker,
    )

    for _ in range(3):
        saver.on_episode_start()
        broker.push_chunk(request_timestamp=time.time() - 0.01, observation_step=0)
        saver.on_step(_obs(0), _action(0))
        saver.on_episode_end()
    saver.close()

    episodes = sorted((tmp_path / "9").iterdir())
    assert [e.name for e in episodes] == [
        "0_real_0_failure",
        "1_real_0_failure",
        "2_real_0_failure",
    ]


def test_no_video_by_default(tmp_path):
    broker = _FakeBroker()
    saver = RealSaver(
        out_dir=tmp_path,
        robot_id="robot_3",
        prompt="x",
        control_hz=20,
        action_chunk_broker=broker,
    )
    _drive_episode(saver, broker, n_steps=2)

    assert not (tmp_path / "3" / "0_real_0_failure" / "out.mp4").exists()


def test_video_written_when_enabled(tmp_path):
    broker = _FakeBroker()
    saver = RealSaver(
        out_dir=tmp_path,
        robot_id="robot_3",
        prompt="x",
        control_hz=20,
        action_chunk_broker=broker,
        save_video=True,
    )
    _drive_episode(saver, broker, n_steps=4)

    out = tmp_path / "3" / "0_real_0_failure" / "out.mp4"
    assert out.is_file()
    assert out.stat().st_size > 0


def test_initial_state_captured_from_first_observation(tmp_path):
    """initial_state should reflect the first observation seen in on_step."""
    broker = _FakeBroker()
    saver = RealSaver(
        out_dir=tmp_path,
        robot_id="robot_3",
        prompt="x",
        control_hz=20,
        action_chunk_broker=broker,
    )

    saver.on_episode_start()
    assert saver._initial_state is None
    saver.on_step(_obs(step=0), _action(0))
    np.testing.assert_array_equal(saver._initial_state, np.zeros(7, dtype=np.float32))
    saver.on_step(_obs(step=1), _action(1))
    # Should not be overwritten by later observations.
    np.testing.assert_array_equal(saver._initial_state, np.zeros(7, dtype=np.float32))
    saver.on_episode_end()
    saver.close()


@pytest.mark.parametrize("idx_in_chunk", [None, 0])
def test_handles_null_action_gracefully(tmp_path, idx_in_chunk):
    """A null action (no chunk) should record NaN cost without crashing."""
    broker = _FakeBroker()
    saver = RealSaver(
        out_dir=tmp_path,
        robot_id="robot_3",
        prompt="x",
        control_hz=20,
        action_chunk_broker=broker,
    )

    saver.on_episode_start()
    if idx_in_chunk is None:
        saver.on_step(_obs(0), _action(0, chunk_idx=None))
    else:
        broker.push_chunk(request_timestamp=time.time() - 0.05, observation_step=0)
        saver.on_step(_obs(0), _action(0, chunk_idx=0))
    saver.on_episode_end()
    saver.close()

    costs = np.load(tmp_path / "3" / "0_real_0_failure" / "cost_history.npy")
    assert costs.shape == (1,)
    if idx_in_chunk is None:
        assert np.isnan(costs[0])
    else:
        assert costs[0] >= 0
