import json
import logging
import pathlib
from collections.abc import Iterator

import numpy as np
import pandas as pd

from evaluation.save import Result
from evaluation.types import ExperimentConfig

logger = logging.getLogger(__name__)


def load_episodes(output_path: pathlib.Path) -> pd.DataFrame:
    results = [Result.from_json(f) for f in output_path.glob("*/*/metadata.json")]
    return pd.DataFrame([result.model_dump() for result in results])


def completed_episodes(df: pd.DataFrame) -> pd.DataFrame:
    # Only for measuring outcomes. Truncated episodes' steps were taken under
    # real contention, so they still belong in every load and latency metric.
    return df[~df["truncated"].astype(bool)]


def iter_steps(output_path: pathlib.Path) -> Iterator[tuple[pathlib.Path, pd.DataFrame]]:
    for steps_file in sorted(output_path.glob("*/*/steps.parquet")):
        yield steps_file.parent, pd.read_parquet(steps_file, engine="pyarrow")


def steps_by_robot(output_path: pathlib.Path) -> dict[str, list[pd.DataFrame]]:
    by_robot: dict[str, list[tuple[int, pd.DataFrame]]] = {}
    for episode_dir, steps in iter_steps(output_path):
        by_robot.setdefault(episode_dir.parent.name, []).append(
            (int(episode_dir.name.split("_")[0]), steps)
        )
    return {
        robot: [steps for _, steps in sorted(episodes, key=lambda episode: episode[0])]
        for robot, episodes in sorted(by_robot.items(), key=lambda item: int(item[0]))
    }


def load_experiment_config(output_path: pathlib.Path) -> ExperimentConfig | None:
    path = output_path / "experiment_args.json"
    if not path.exists():
        return None
    return ExperimentConfig.model_validate(json.loads(path.read_text())["experiment_config"])


def load_control_hz(output_path: pathlib.Path, fallback: float = 20.0) -> float:
    config = load_experiment_config(output_path)
    if config is None or not config.robots:
        return fallback
    return float(max(robot.control_hz for robot in config.robots))


def load_experiment_duration(output_path: pathlib.Path) -> float | None:
    spans = [
        (float(steps["timestamp"].iloc[0]), float(steps["timestamp"].iloc[-1]))
        for _, steps in iter_steps(output_path)
    ]
    if not spans:
        return None
    return max(end for _, end in spans) - min(start for start, _ in spans)


def load_planner_starvation_metrics(output_path: pathlib.Path) -> pd.DataFrame:
    control_hz = load_control_hz(output_path)
    rows = []
    for episode_dir, steps in iter_steps(output_path):
        result = Result.from_json(episode_dir / "metadata.json")
        starved = steps["action_chunk_index"].isna().to_numpy()
        total = int(starved.shape[0])
        if total == 0:
            logger.warning("No control steps in %s; skipping episode metrics", episode_dir)
            continue
        served = np.flatnonzero(~starved)
        first = int(served[0]) if served.size else total
        rows.append(
            {
                "robot_idx": result.robot_idx,
                "episode_idx": result.episode_idx,
                "task_suite_name": result.task_suite_name,
                "task_id": result.task_id,
                "starvation_steps": int(starved.sum()),
                "observed_steps": total,
                "planner_starvation_seconds": float(starved.sum()) / control_hz,
                "post_first_starvation_steps": int(starved[first:].sum()),
                "post_first_observed_steps": total - first,
            }
        )
    return pd.DataFrame(rows)


def robot_starvation_rates(output_path: pathlib.Path) -> pd.DataFrame:
    df = load_planner_starvation_metrics(output_path)
    if df.empty:
        return df
    agg = (
        df.groupby("robot_idx")[["starvation_steps", "observed_steps"]]
        .sum()
        .reset_index()
        .sort_values("robot_idx")
    )
    agg["starvation_rate"] = agg["starvation_steps"] / agg["observed_steps"]
    return agg


def actions_left_matrix(
    output_path: pathlib.Path, control_hz: float | None = None
) -> tuple[list[str], np.ndarray, list[list[int]], float, float]:
    by_robot: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for robot, episodes in steps_by_robot(output_path).items():
        for steps in episodes:
            traced = steps.dropna(subset=["actions_left"])
            by_robot.setdefault(robot, []).append(
                (
                    traced["timestamp"].to_numpy(dtype=float),
                    traced["actions_left"].to_numpy(dtype=int),
                )
            )

    if not any(len(ts) for episodes in by_robot.values() for ts, _ in episodes):
        return [], np.empty((0, 0)), [], control_hz or load_control_hz(output_path), 0.0

    robots = sorted(by_robot, key=int, reverse=True)

    # Canvas rate: caller-supplied wins; otherwise the max observed rate, so a
    # slow robot's bars render proportionally wider than a fast robot's.
    if control_hz is None:
        gaps = [
            float(np.median(np.diff(ts)))
            for episodes in by_robot.values()
            for ts, _ in episodes
            if len(ts) >= 2
        ]
        rates = [1.0 / gap for gap in gaps if gap > 0]
        control_hz = max(rates) if rates else load_control_hz(output_path)
    control_hz = float(control_hz)

    t0 = min(ts[0] for episodes in by_robot.values() for ts, _ in episodes if len(ts))

    max_col = 0
    placements = []
    for robot in robots:
        per_episode = []
        for ts, values in by_robot[robot]:
            cols = np.clip(np.round((ts - t0) * control_hz).astype(int), 0, None)
            per_episode.append((cols, values))
            if cols.size:
                max_col = max(max_col, int(cols.max()))
        placements.append(per_episode)

    matrix = np.full((len(robots), max_col + 1), np.nan, dtype=float)
    episode_boundaries: list[list[int]] = []
    for row, per_episode in enumerate(placements):
        boundaries = []
        for cols, values in per_episode:
            if not cols.size:
                continue
            # Forward-fill each step's queue depth until the robot's next step,
            # so robots slower than the canvas rate don't leave NaN gaps.
            for j in range(len(cols)):
                start = int(cols[j])
                end = max(int(cols[j + 1]) if j + 1 < len(cols) else start + 1, start + 1)
                matrix[row, start:end] = values[j]
            boundaries.append(int(cols[0]))
        episode_boundaries.append(boundaries)

    return robots, matrix, episode_boundaries, control_hz, t0


def starvation_variance_series(
    output_path: pathlib.Path, control_hz: float | None = None
) -> dict | None:
    per_robot: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for robot, episodes in steps_by_robot(output_path).items():
        steps = pd.concat(episodes).sort_values("timestamp")
        ts = steps["timestamp"].to_numpy(dtype=float)
        if not len(ts):
            continue
        starved = steps["action_chunk_index"].isna().to_numpy()
        per_robot[robot] = (ts, np.cumsum(starved), np.arange(1, len(ts) + 1))
    if not per_robot:
        return None

    control_hz = float(control_hz or load_control_hz(output_path))
    t0 = min(ts[0] for ts, _, _ in per_robot.values())
    t_end = max(ts[-1] for ts, _, _ in per_robot.values())
    n_cols = int(np.ceil((t_end - t0) * control_hz)) + 1
    grid = t0 + np.arange(n_cols) / control_hz

    robots = list(per_robot)
    cumulative_rates = np.full((len(robots), n_cols), np.nan)
    for row, robot in enumerate(robots):
        ts, cumulative_starved, cumulative_observed = per_robot[robot]
        idx = np.searchsorted(ts, grid, side="right") - 1
        valid = idx >= 0
        cumulative_rates[row, valid] = (
            cumulative_starved[idx[valid]] / cumulative_observed[idx[valid]]
        )

    starvation_variance = np.nanvar(cumulative_rates, axis=0)
    return {
        "robots": robots,
        "cumulative_rates": cumulative_rates,
        "starvation_variance": starvation_variance,
        "time_seconds": grid - t0,
        "control_hz": control_hz,
        "final_starvation_variance": float(starvation_variance[-1]),
    }


def _robot_idx(robot_id: str) -> str:
    return str(robot_id).removeprefix("robot_")


def _read_jsonl(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True, convert_dates=False)


def load_server_batches(output_path: pathlib.Path) -> pd.DataFrame:
    df = _read_jsonl(output_path / "server" / "batches.jsonl")
    if not df.empty:
        df["robot_ids"] = df["robot_ids"].map(lambda ids: [_robot_idx(rid) for rid in ids])
    return df


def load_server_events(output_path: pathlib.Path) -> pd.DataFrame:
    df = _read_jsonl(output_path / "server" / "events.jsonl")
    if not df.empty:
        df["robot_id"] = df["robot_id"].map(_robot_idx)
    return df


def request_timings(output_path: pathlib.Path) -> pd.DataFrame:
    events = load_server_events(output_path)
    batches = load_server_batches(output_path)
    if events.empty or batches.empty:
        return pd.DataFrame()

    requests = events[events["kind"] == "request"][
        ["robot_id", "request_id", "request_timestamp", "arrival_time"]
    ]
    acks = events[events["kind"] == "ack"][
        ["robot_id", "request_id", "server_send_time", "receive_time"]
    ]
    # The engine can replace selected observations with newer slot contents.
    # Responses/ACKs refer to processed IDs, not the selection-time IDs.
    if "processed_request_ids" in batches:
        batches = batches.copy()
        batches["request_ids"] = batches["processed_request_ids"]
        batches["robot_ids"] = batches["processed_robot_ids"].map(
            lambda ids: [_robot_idx(rid) for rid in ids]
        )
    served = (
        batches[batches["batch_size"] > 0][
            ["robot_ids", "request_ids", "inference_start_time", "inference_duration"]
        ]
        .explode(["robot_ids", "request_ids"])
        .rename(columns={"robot_ids": "robot_id", "request_ids": "request_id"})
    )

    # Retain robot identity as well as the server-assigned request ID.
    df = requests.merge(served, on=["robot_id", "request_id"]).merge(
        acks, on=["robot_id", "request_id"], how="left"
    )
    return pd.DataFrame(
        {
            "robot_id": df["robot_id"],
            "send_ms": (df["arrival_time"] - df["request_timestamp"]) * 1000.0,
            "queue_ms": (df["inference_start_time"] - df["arrival_time"]) * 1000.0,
            "inference_ms": df["inference_duration"] * 1000.0,
            "receive_ms": (df["receive_time"] - df["server_send_time"]) * 1000.0,
        }
    )


def load_scheduler_decisions(output_path: pathlib.Path) -> list[dict]:
    path = output_path / "server" / "scheduler_decisions.jsonl"
    if not path.exists():
        return []
    decisions = [json.loads(line) for line in path.read_text().splitlines()]
    for decision in decisions:
        decision["scheduled"] = [_robot_idx(rid) for rid in decision["scheduled"]]
        decision["candidates"] = [_robot_idx(rid) for rid in decision["candidates"]]
    return decisions


def load_server_metadata(output_path: pathlib.Path) -> dict:
    path = output_path / "server" / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}
