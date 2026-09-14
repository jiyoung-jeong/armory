"""Accounting examples whose outcomes are independent of the replay implementation."""

import json

import pandas as pd
import pytest
from scripts.analyze_chunk_usage import reconcile_trial, replay_episode


def chunk(index, start, receipt_step, horizon):
    return dict(
        request_id=index + 10,
        chunk_id=index + 100,
        observation_step=0,
        action_index_start=start,
        execution_start_step=receipt_step,
        actions=[[0.0]] * horizon,
        max_execution_horizon=horizon,
        request_timestamp=1.0,
        response_timestamp=1.01 + index * 0.01,
    )


def overlap_example():
    # A supplies six actions. Two execute; B replaces the four remaining ones
    # while skipping its already executed prefix. C is entirely replaced by D
    # between consecutive get_action calls; only one D action executes.
    steps = pd.DataFrame(
        dict(
            env_step=range(6),
            timestamp=[1.0 + i * 0.05 for i in range(6)],
            action_chunk_index=[None, 0, 0, 1, 1, 3],
            action_index=[None, 0, 1, 1, 2, 0],
            actions_left=[0, 6, 5, 4, 3, 3],
        )
    )
    chunks = pd.DataFrame(
        [
            chunk(0, 0, 1, 6),
            chunk(1, 1, 3, 5),
            chunk(2, 4, 5, 3),
            chunk(3, 4, 5, 3),
        ]
    )
    return steps, chunks


def test_replacement_is_not_net_replenishment_or_waste():
    steps, chunks = overlap_example()
    rows, actions, _ = replay_episode(steps, chunks)
    assert [r["executed_actions"] for r in rows] == [2, 2, 0, 1]
    assert [r["net_queue_addition"] for r in rows] == [6, 0, 1, 0]
    assert rows[1]["skipped_past_prefix"] == 1
    assert rows[1]["first_executed_index"] == 1
    assert rows[0]["replaced_before_execution"] == 4
    assert rows[2]["usage_category"] == "fully_replaced_before_execution"
    assert rows[3]["remaining_at_episode_snapshot"] == 2
    assert len(actions) == 5


def test_queue_validation_detects_missing_receipt_information():
    steps, chunks = overlap_example()
    chunks.loc[3, "execution_start_step"] = 6
    with pytest.raises(ValueError, match="Broker replay mismatch"):
        replay_episode(steps, chunks)


def test_unrecorded_tail_step_is_not_invented():
    steps, chunks = overlap_example()
    chunks.loc[3, "execution_start_step"] = 7
    with pytest.raises(ValueError, match="unrecorded control step"):
        replay_episode(steps, chunks)


def test_episode_without_any_response_has_no_post_first_starvation():
    steps = pd.DataFrame(
        dict(
            env_step=[0, 1],
            timestamp=[1.0, 1.05],
            action_chunk_index=[None, None],
            action_index=[None, None],
            actions_left=[0, 0],
        )
    )
    assert replay_episode(steps, pd.DataFrame()) == ([], [], [])


def test_reverse_reconciliation_keeps_ack_missing_distinct_from_unsaved(tmp_path):
    server = tmp_path / "policy/server"
    server.mkdir(parents=True)
    batches = [
        dict(
            processed_robot_ids=["robot_0"],
            processed_request_ids=[10 + i],
            chunk_ids=[100 + i],
            batch_id=i,
            batch_size=1,
            inference_start_time=2.0 + i,
            inference_duration=0.5,
        )
        for i in range(3)
    ]
    events = [
        dict(
            kind="request",
            robot_id="robot_0",
            request_id=10 + i,
            observation_step=i,
            request_timestamp=1.0 + i,
        )
        for i in range(3)
    ] + [
        dict(
            kind="ack",
            robot_id="robot_0",
            request_id=10 + i,
            chunk_id=100 + i,
            receive_time=2.6 + i,
            server_send_time=2.55 + i,
        )
        for i in range(2)
    ]
    for name, rows in [("batches", batches), ("events", events)]:
        (server / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    saved = pd.DataFrame(
        [
            dict(
                robot_id="robot_0",
                request_id=10,
                chunk_id=100,
                request_timestamp=1.0,
                episode=0,
                executed_actions=2,
            )
        ]
    )
    bounds = {("robot_0", 0): dict(first=1.0, last=3.5, truncated=True)}
    rows = reconcile_trial(tmp_path, saved, bounds)
    assert [r["stage"] for r in rows] == ["saved", "acked_not_saved", "no_recorded_ack"]
    assert rows[1]["after_source_last_recorded_step"]
    assert rows[2]["server_send_timestamp"] is None
    assert rows[2]["after_source_last_recorded_step"]
    saved.loc[0, "episode"] = 1
    rows = reconcile_trial(tmp_path, saved, bounds)
    assert not rows[0]["saved_in_source_episode"]
    assert rows[0]["saved_episode"] == 1
    assert rows[0]["recorded_executed_actions"] == 2


def test_replay_distinguishes_no_response_initial_wait_from_later_starvation():
    steps = pd.DataFrame(
        dict(
            env_step=range(6),
            timestamp=[1.0 + i * 0.05 for i in range(6)],
            action_chunk_index=[None, 0, 0, None, None, None],
            action_index=[None, 0, 1, None, None, None],
            actions_left=[0, 2, 1, 0, 0, 0],
        )
    )
    rows, actions, streaks = replay_episode(steps, pd.DataFrame([chunk(0, 0, 1, 2)]))
    assert rows[0]["executed_actions"] == 2
    assert len(actions) == 2
    assert len(streaks) == 1
    assert streaks[0]["length_steps"] == 3
    assert not streaks[0]["next_recorded_action_exists"]
