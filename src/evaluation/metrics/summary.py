import logging
import pathlib

import pandas as pd
from rich.console import Console
from rich.table import Table

from evaluation.metrics.loading import (
    completed_episodes,
    load_episodes,
    load_experiment_duration,
    load_planner_starvation_metrics,
)

logger = logging.getLogger(__name__)

STARVATION_COLUMNS = [
    "starvation_steps",
    "observed_steps",
    "planner_starvation_seconds",
    "post_first_starvation_steps",
    "post_first_observed_steps",
]


def _format_rate(rate: float) -> str:
    return "n/a" if pd.isna(rate) else f"{rate:.2%}"


def calculate_metrics(output_path: pathlib.Path) -> None:
    df = load_episodes(output_path)
    if df.empty:
        logger.warning("No results found")
        return

    starvation = load_planner_starvation_metrics(output_path)
    if starvation.empty:
        df[STARVATION_COLUMNS] = 0
    else:
        df = df.merge(
            starvation,
            on=["robot_idx", "episode_idx", "task_suite_name", "task_id"],
            how="left",
        )
        df[STARVATION_COLUMNS] = df[STARVATION_COLUMNS].fillna(0)

    df.to_csv(output_path / "results.csv", index=False)

    # Starvation and step counts aggregate over every episode; success rates
    # aggregate over completed episodes only.
    completed = completed_episodes(df)
    group_keys = ["task_suite_name", "task_id"]

    summary = df.groupby(group_keys).agg(
        {"truncated": "sum", **{column: "sum" for column in STARVATION_COLUMNS}}
    )
    summary["success"] = completed.groupby(group_keys)["success"].mean()
    summary["planner_starvation_rate"] = summary["starvation_steps"] / summary["observed_steps"]
    summary["post_first_starvation_rate"] = (
        summary["post_first_starvation_steps"] / summary["post_first_observed_steps"]
    )
    summary.reset_index().to_csv(output_path / "summary.csv", index=False)

    console = Console()
    table = Table(title="Task Success Summary")
    table.add_column("Task Suite", style="cyan")
    table.add_column("Task ID", style="magenta")
    table.add_column("Success Rate", style="green")
    table.add_column("Truncated", style="blue")
    table.add_column("Total Starvation Steps", style="yellow")
    table.add_column("Starvation Rate", style="yellow")
    for _, row in summary.reset_index().iterrows():
        table.add_row(
            str(row["task_suite_name"]),
            str(row["task_id"]),
            _format_rate(row["success"]),
            str(int(row["truncated"])),
            str(int(row["starvation_steps"])),
            _format_rate(row["planner_starvation_rate"]),
        )
    console.print(table)

    robot_summary = df.groupby("robot_idx").agg(
        count=("episode_idx", "count"),
        truncated=("truncated", "sum"),
        starvation_steps=("starvation_steps", "sum"),
        observed_steps=("observed_steps", "sum"),
    )
    robot_summary["success"] = completed.groupby("robot_idx")["success"].mean()
    robot_summary["planner_starvation_rate"] = (
        robot_summary["starvation_steps"] / robot_summary["observed_steps"]
    )

    robot_table = Table(title="Per-Robot Success Summary")
    robot_table.add_column("Robot", style="cyan")
    robot_table.add_column("Success Rate", style="green")
    robot_table.add_column("Episodes", style="magenta")
    robot_table.add_column("Truncated", style="blue")
    robot_table.add_column("Total Starvation Steps", style="yellow")
    robot_table.add_column("Starvation Rate", style="yellow")
    for robot_idx, row in robot_summary.sort_index().iterrows():
        robot_table.add_row(
            str(int(robot_idx)),
            _format_rate(row["success"]),
            str(int(row["count"])),
            str(int(row["truncated"])),
            str(int(row["starvation_steps"])),
            _format_rate(row["planner_starvation_rate"]),
        )
    console.print(robot_table)

    total_starvation_steps = int(df["starvation_steps"].sum())
    total_observed_steps = int(df["observed_steps"].sum())
    total_post_first_starvation = int(df["post_first_starvation_steps"].sum())
    total_post_first_observed = int(df["post_first_observed_steps"].sum())
    console.print(
        f"\n[bold green]Total success rate: "
        f"{_format_rate(completed['success'].mean() if len(completed) else float('nan'))}[/bold green]"
    )
    console.print(
        f"[bold yellow]Total starvation steps: {total_starvation_steps} control steps[/bold yellow]"
    )
    console.print(
        f"[bold yellow]Planner starvation rate: "
        f"{_format_rate(total_starvation_steps / total_observed_steps if total_observed_steps else float('nan'))}[/bold yellow]"
    )
    console.print(
        f"[bold yellow]Planner starvation rate (excl. pre-first-action): "
        f"{_format_rate(total_post_first_starvation / total_post_first_observed if total_post_first_observed else float('nan'))}[/bold yellow]"
    )
    console.print(
        f"[bold yellow]Planner starvation time: {df['planner_starvation_seconds'].sum():.2f}s[/bold yellow]"
    )

    experiment_duration = load_experiment_duration(output_path)
    if experiment_duration is not None:
        console.print(
            f"[bold cyan]Total experiment time: {experiment_duration:.1f}s "
            f"({experiment_duration / 60:.1f}min)[/bold cyan]"
        )
        console.print(
            f"[bold cyan]Throughput: {int(df['success'].sum()) / experiment_duration:.3f} "
            f"successes/second[/bold cyan]"
        )
