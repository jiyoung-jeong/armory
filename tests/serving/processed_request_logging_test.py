"""A newer slot observation must remain joinable to its real response."""

import io
import json
from types import SimpleNamespace

import pytest

from armory.serving.engine import GpuWorker
from evaluation.metrics.loading import request_timings


@pytest.mark.parametrize("new_format", [True, False])
def test_timing_join_uses_processed_observation(tmp_path, new_format):
    server = tmp_path / "server"
    server.mkdir()
    worker = object.__new__(GpuWorker)
    worker._batches_log = io.StringIO()
    worker._log_batch(
        SimpleNamespace(
            batch_id=3,
            batch_size=1,
            inference_start_time=1.3,
            inference_duration=0.5,
            responses=[
                SimpleNamespace(robot_id="robot_0", request_id=11, chunk_id=7, observation_step=2)
            ],
        ),
        [SimpleNamespace(robot_id="robot_0", request_id=10)],
    )
    batch = json.loads(worker._batches_log.getvalue())
    assert batch["request_ids"] == [10]
    assert batch["processed_request_ids"] == [11]
    assert batch["chunk_ids"] == [7]
    assert batch["processed_observation_steps"] == [2]
    if not new_format:
        batch = {k: v for k, v in batch.items() if not k.startswith("processed_")}
        batch["request_ids"] = [11]  # Legacy logs where the selected request was used.
    (server / "batches.jsonl").write_text(json.dumps(batch) + "\n")
    events = [
        dict(
            kind="request",
            robot_id="robot_0",
            request_id=10,
            request_timestamp=1.0,
            arrival_time=1.01,
        ),
        dict(
            kind="request",
            robot_id="robot_0",
            request_id=11,
            request_timestamp=1.1,
            arrival_time=1.2,
        ),
        dict(
            kind="ack", robot_id="robot_0", request_id=11, server_send_time=1.81, receive_time=1.82
        ),
    ]
    (server / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in events))
    result = request_timings(tmp_path)
    assert len(result) == 1
    assert result.iloc[0]["send_ms"] == pytest.approx(100)
    assert result.iloc[0]["queue_ms"] == pytest.approx(100)
    assert result.iloc[0]["inference_ms"] == pytest.approx(500)
    assert result.iloc[0]["receive_ms"] == pytest.approx(10)


def test_server_log_link_cannot_target_overwritten_client_output(tmp_path):
    from scripts.run import Args

    server = tmp_path / "client" / "server"
    server.mkdir(parents=True)
    with pytest.raises(ValueError, match="outside the client output"):
        Args(output_dir=server.parent, server_log_dir=server, overwrite=True)
