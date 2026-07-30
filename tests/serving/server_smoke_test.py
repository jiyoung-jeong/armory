from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import time
import traceback
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from armory.scheduling.base import RequestScheduler
from armory.serving.config import ServerConfig
from armory.serving.protocol import SchedulerConfig, ServerMetadata
from armory.serving.scheduler import SCHEDULER_REGISTRY
from armory.serving.schemas import SlotRequest
from armory.serving.server import NUM_WARMUP, create_app
from armory_client import msgpack_numpy
from armory_client.messages import (
    ConnectRequest,
    InferRequest,
    InferResponse,
    ResponseAck,
    WarmupAck,
    WarmupPing,
    WarmupPong,
)

ACTION_HORIZON = 4
ACTION_DIM = 2
SCENARIO_TIMEOUT_S = 30.0
TWO_ROBOT_ALGORITHM = "_smoke-two-robot-batch"


class _TwoRobotBatchScheduler(RequestScheduler):
    """Wait for both smoke-test robots so their requests share one real batch."""

    def get_next_batches(
        self, candidates: list[SlotRequest]
    ) -> tuple[list[list[SlotRequest]], dict[str, Any]]:
        if self.mirror.in_flight_batches_count > 0:
            return [], {"reason": "server_busy"}
        if len(candidates) < 2:
            return [], {"reason": "waiting_for_two_robots"}
        return [candidates[:2]], {"rule": "two_robot_smoke_batch"}


class _SmokePolicy:
    """Small CPU policy exercising the real engine process and shared slots."""

    def warmup(self, max_batch_size: int) -> None:
        del max_batch_size

    def make_infer_request(self) -> object:
        return object()

    def infer_batch(self, requests: list[object]) -> list[dict[str, Any]]:
        # The GPU PUB socket is bound immediately before profiling. A nonzero,
        # realistic inference duration gives the scheduler SUB connection time
        # to establish before the one-shot BatchProfile is published.
        time.sleep(0.05)
        actions = np.arange(ACTION_HORIZON * ACTION_DIM, dtype=np.float32).reshape(
            ACTION_HORIZON, ACTION_DIM
        )
        return [
            {
                "actions": actions.copy(),
                "noise": None,
                "rtc_prev_actions": actions.copy(),
            }
            for _ in requests
        ]


class _SmokePolicyFactory:
    def __call__(self) -> _SmokePolicy:
        return _SmokePolicy()


def _send(websocket: Any, message: object) -> None:
    websocket.send_bytes(msgpack_numpy.packb(message))


def _receive(websocket: Any) -> dict[str, Any]:
    return msgpack_numpy.unpackb(websocket.receive_bytes())


def _connect_and_warmup(websocket: Any, robot_id: str) -> None:
    _send(websocket, ConnectRequest(robot_id=robot_id, control_hz=10.0))
    assert _receive(websocket) == {"type": "connect_response"}

    for _ in range(NUM_WARMUP):
        _send(
            websocket,
            WarmupPing(client_timestamp=time.time(), payload=b"observation-payload"),
        )
        pong = WarmupPong(**_receive(websocket))
        assert len(pong.payload) == ACTION_HORIZON * ACTION_DIM * 4
        _send(
            websocket,
            WarmupAck(
                server_send_time=pong.server_send_time,
                client_receive_time=time.time(),
            ),
        )


def _send_infer(websocket: Any, robot_id: str, observation_step: int) -> None:
    requested_at = time.time()
    _send(
        websocket,
        InferRequest(
            robot_id=robot_id,
            observation={
                "state": np.array([observation_step, observation_step + 1], dtype=np.float32)
            },
            observation_step=observation_step,
            action_index_start=observation_step * ACTION_HORIZON,
            request_timestamp=requested_at,
            deadline=requested_at + 10.0,
            min_execution_horizon=1,
            max_execution_horizon=ACTION_HORIZON,
        ),
    )


def _receive_and_ack(websocket: Any, robot_id: str, observation_step: int) -> InferResponse:
    response = InferResponse(**_receive(websocket))
    assert response.robot_id == robot_id
    assert response.observation_step == observation_step
    assert response.action_index_start == observation_step * ACTION_HORIZON
    assert response.actions.shape == (ACTION_HORIZON, ACTION_DIM)
    assert response.actions.dtype == np.float32

    _send(
        websocket,
        ResponseAck(
            request_id=response.request_id,
            chunk_id=response.chunk_id,
            observation_step=response.observation_step,
            receive_time=time.time(),
            action_index_start=response.action_index_start,
            min_execution_horizon=response.min_execution_horizon,
            max_execution_horizon=response.max_execution_horizon,
            execution_start_step=response.action_index_start,
        ),
    )
    return response


def _run_server_scenario(result_queue: mp.Queue) -> None:
    # Put this process and its daemon scheduler/GPU children in a test-owned
    # process group so the pytest parent can clean up the entire topology if any
    # blocking IPC operation times out.
    os.setsid()
    try:
        # The scenario process forks the scheduler process after this local-only
        # registry entry is installed. Production registry contents are untouched.
        SCHEDULER_REGISTRY[TWO_ROBOT_ALGORITHM] = _TwoRobotBatchScheduler
        metadata = ServerMetadata(
            config_name="smoke",
            checkpoint_dir="",
            action_horizon=ACTION_HORIZON,
            action_dim=ACTION_DIM,
            num_steps=1,
            max_batch_size=2,
            env="TEST",
            scheduling_algorithm=TWO_ROBOT_ALGORITHM,
        )

        with TestClient(
            create_app(
                metadata,
                _SmokePolicyFactory(),
                ServerConfig(
                    max_batch_size=2,
                    scheduler=SchedulerConfig(scheduling_algorithm=TWO_ROBOT_ALGORITHM),
                ),
            )
        ) as client:
            initial_metadata = client.get("/metadata")
            assert initial_metadata.status_code == 200
            assert initial_metadata.json()["scheduling_algorithm"] == TWO_ROBOT_ALGORITHM

            # This is the lifecycle used by scripts/run.py: control-plane changes
            # happen before robot connections and their warmup latency seeds.
            reconfigured = client.post(
                "/reconfigure",
                json={"scheduling_algorithm": TWO_ROBOT_ALGORITHM},
            )
            assert reconfigured.status_code == 200
            assert reconfigured.json() == {
                "status": "ok",
                "scheduling_algorithm": TWO_ROBOT_ALGORITHM,
                "scheduler": {
                    "scheduling_algorithm": TWO_ROBOT_ALGORITHM,
                    "alpha": 1.0,
                },
            }
            reset = client.post("/reset")
            assert reset.status_code == 200
            assert reset.json()["status"] == "ok"

            with (
                client.websocket_connect("/ws") as robot_a,
                client.websocket_connect("/ws") as robot_b,
            ):
                _connect_and_warmup(robot_a, "robot-a")
                _connect_and_warmup(robot_b, "robot-b")

                _send_infer(robot_a, "robot-a", observation_step=0)
                _send_infer(robot_b, "robot-b", observation_step=0)
                response_a = _receive_and_ack(robot_a, "robot-a", observation_step=0)
                response_b = _receive_and_ack(robot_b, "robot-b", observation_step=0)

                assert response_a.request_id != response_b.request_id
                assert response_a.chunk_id != response_b.chunk_id
            assert client.get("/metadata").json()["scheduling_algorithm"] == TWO_ROBOT_ALGORITHM

            # Characterize a complete trial boundary: disconnected robots,
            # ResetAll, then a fresh warmup and inference using the same robot id.
            replay_reconfigure = client.post(
                "/reconfigure",
                json={"scheduling_algorithm": "round-robin"},
            )
            assert replay_reconfigure.status_code == 200
            second_reset = client.post("/reset")
            assert second_reset.status_code == 200
            with client.websocket_connect("/ws") as robot_a:
                _connect_and_warmup(robot_a, "robot-a")
                _send_infer(robot_a, "robot-a", observation_step=0)
                replay_response = _receive_and_ack(robot_a, "robot-a", observation_step=0)
                assert replay_response.request_id > response_b.request_id

        result_queue.put(("ok", None))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
        raise


def _kill_test_process_group(process: mp.Process) -> None:
    process_group = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        return True

    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while group_exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    process.join(timeout=0.1)
    if not group_exists():
        return
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.join(timeout=1.0)


def test_mock_policy_three_process_serving_lifecycle() -> None:
    context = mp.get_context("fork")
    result_queue = context.Queue()
    scenario = context.Process(target=_run_server_scenario, args=(result_queue,))
    scenario.start()
    scenario.join(timeout=SCENARIO_TIMEOUT_S)

    if scenario.is_alive():
        _kill_test_process_group(scenario)
        raise AssertionError(
            f"mock serving scenario exceeded {SCENARIO_TIMEOUT_S:.0f}s; "
            "the test-owned process group was terminated"
        )

    if scenario.exitcode != 0:
        # The server watchdog terminates only the scenario process when a worker
        # dies. Clean up any surviving test-owned worker in the now-leaderless
        # process group before reporting the child traceback below.
        _kill_test_process_group(scenario)

    try:
        status, details = result_queue.get(timeout=2.0)
    except queue.Empty:
        status, details = "error", "scenario exited without reporting a result"
    finally:
        result_queue.close()
        result_queue.join_thread()

    assert scenario.exitcode == 0, details
    assert status == "ok", details
