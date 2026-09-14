"""Short deadline-truncated episodes may have no response file."""

import json

import pandas as pd
import pytest
from scripts.analyze_local_batch_sweep import summarize_trial


@pytest.mark.parametrize("remove_served_chunks", [False, True])
def test_missing_chunk_file_only_allowed_without_executed_actions(tmp_path, remove_served_chunks):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                name="fixture",
                status="complete",
                robots=1,
                max_batch_size=1,
                repeat=1,
                seconds=3,
                profiling=False,
            )
        )
    )
    client = tmp_path / "client"
    first = client / "0" / "0_success"
    last = client / "0" / "1_truncated"
    first.mkdir(parents=True)
    last.mkdir()
    pd.DataFrame(dict(timestamp=[1.0, 1.05], action_chunk_index=[None, 0])).to_parquet(
        first / "steps.parquet"
    )
    pd.DataFrame(
        dict(timestamp=[2.0, 2.05, 2.1], action_chunk_index=[None, None, None])
    ).to_parquet(last / "steps.parquet")
    if not remove_served_chunks:
        pd.DataFrame(
            [dict(request_id=2, chunk_id=1, request_timestamp=0.8, response_timestamp=1.02)]
        ).to_parquet(first / "action_chunks.parquet")
    pd.DataFrame(
        [
            dict(
                robot_idx=0,
                task_id=5,
                success=True,
                truncated=False,
                observed_steps=2,
                starvation_steps=1,
                post_first_starvation_steps=0,
                post_first_observed_steps=1,
            ),
            dict(
                robot_idx=0,
                task_id=5,
                success=False,
                truncated=True,
                observed_steps=3,
                starvation_steps=3,
                post_first_starvation_steps=0,
                post_first_observed_steps=0,
            ),
        ]
    ).to_csv(client / "results.csv", index=False)
    server = tmp_path / "policy" / "server"
    server.mkdir(parents=True)
    (server / "batches.jsonl").write_text(
        json.dumps(
            dict(
                batch_id=1,
                batch_size=1,
                robot_ids=["robot_0"],
                request_ids=[2],
                processed_robot_ids=["robot_0"],
                processed_request_ids=[2],
                chunk_ids=[1],
                inference_start_time=0.9,
                inference_duration=0.1,
            )
        )
        + "\n"
    )
    (server / "events.jsonl").write_text(
        "".join(
            json.dumps(x) + "\n"
            for x in [
                dict(kind="request", robot_id="robot_0", request_id=2),
                dict(kind="ack", robot_id="robot_0", request_id=2),
            ]
        )
    )
    (tmp_path / "gpu_samples.jsonl").write_text(json.dumps(dict(value="1000, 5")) + "\n")
    if remove_served_chunks:
        with pytest.raises(ValueError, match="Missing chunks for executed actions"):
            summarize_trial(tmp_path)
    else:
        row, _ = summarize_trial(tmp_path)
        assert row["steps"] == 5
        assert row["starvation_steps"] == 4
        assert row["stored_chunks"] == 1
        assert row["truncated"] == 1
