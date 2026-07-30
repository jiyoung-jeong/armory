import csv
import pathlib
import subprocess
from typing import Any

ARTIFACTS_VOLUME_NAME = "armory-experiment-artifacts"


def download_artifacts(*, stamp: str, out: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    artifacts_dir = out / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading artifacts from volume '{ARTIFACTS_VOLUME_NAME}/{stamp}' -> {artifacts_dir}")
    subprocess.run(
        ["modal", "volume", "get", ARTIFACTS_VOLUME_NAME, stamp, str(artifacts_dir), "--force"],
        check=True,
    )
    for row in rows:
        row["artifact_path"] = str(artifacts_dir / stamp / row["run_id"])


def write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})

    print(f"Wrote {path}")


def summarize(output_dir: pathlib.Path) -> dict[str, Any]:
    """Compute every available metric for a finished run; skip what isn't present."""
    import pandas as pd  # noqa: PLC0415

    from evaluation.metrics import (  # noqa: PLC0415
        compute_server_timing_health,
        load_server_metadata,
        starvation_variance_series,
    )

    out: dict[str, Any] = {}

    results_path = output_dir / "results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        completed = df[~df["truncated"].astype(bool)]
        if len(completed):
            out["success_rate"] = float(completed["success"].mean())
        if "observed_steps" in df.columns and df["observed_steps"].sum() > 0:
            out["starvation_rate"] = float(
                df["starvation_steps"].sum() / df["observed_steps"].sum()
            )
            post_first_observed = df["post_first_observed_steps"].sum()
            out["post_first_starvation_rate"] = (
                float(df["post_first_starvation_steps"].sum() / post_first_observed)
                if post_first_observed
                else 0.0
            )
            by_robot = df.groupby("robot_idx")[["starvation_steps", "observed_steps"]].sum()
            rates = (by_robot["starvation_steps"] / by_robot["observed_steps"]).dropna()
            if len(rates):
                rates = rates.sort_values()
                tail = max(1, int(len(rates) * 0.1))
                out["robot_starvation_rate_max"] = float(rates.max())
                out["robot_starvation_rate_std"] = float(rates.std(ddof=0))
                out["robot_starvation_rate_cvar90"] = float(rates.tail(tail).mean())

    server = load_server_metadata(output_dir)
    if server:
        out["max_batch_size"] = server.get("max_batch_size", "")
        out["action_horizon"] = server.get("action_horizon", "")
        out["alpha_observed"] = (server.get("scheduler") or {}).get("alpha")

    try:
        health = compute_server_timing_health(output_dir)
        if health:
            out.update(health)
        series = starvation_variance_series(output_dir)
        if series is not None:
            out["starvation_variance"] = series["final_starvation_variance"]
    except Exception:  # noqa: BLE001
        pass

    return out
