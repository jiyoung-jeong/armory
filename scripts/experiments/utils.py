import pathlib
import subprocess
from typing import Any

from _setups import ARTIFACTS_VOLUME_NAME


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


# FIXME: too complicated
def summarize(output_dir: pathlib.Path) -> dict[str, Any]:
    """Compute every available metric for a finished run; skip what isn't present.

    Shared by all three sweeps — the starvation and fairness experiments just read
    different columns out of the union.
    """
    import csv  # noqa: PLC0415
    import statistics  # noqa: PLC0415

    def _f(value: Any, default: float = 0.0) -> float:
        try:
            return default if value is None else float(value)
        except (TypeError, ValueError):
            return default

    out: dict[str, Any] = {}

    summary_path = output_dir / "summary.csv"
    if summary_path.exists():
        with summary_path.open() as f:
            rows = list(csv.DictReader(f))
        if rows:
            out["success_rate"] = sum(_f(r.get("success")) for r in rows) / len(rows)
            observed = sum(_f(r.get("observed_steps")) for r in rows)
            starved = sum(_f(r.get("starvation_steps")) for r in rows)
            out["starvation_rate"] = starved / observed if observed else 0.0
            pf_observed = sum(_f(r.get("post_first_observed_steps")) for r in rows)
            pf_starved = sum(_f(r.get("post_first_starvation_steps")) for r in rows)
            out["post_first_starvation_rate"] = pf_starved / pf_observed if pf_observed else 0.0

    results_path = output_dir / "results.csv"
    if results_path.exists():
        by_robot: dict[str, dict[str, float]] = {}
        with results_path.open() as f:
            for row in csv.DictReader(f):
                robot = str(row.get("robot_idx", "unknown"))
                stats = by_robot.setdefault(robot, {"starved": 0.0, "observed": 0.0})
                stats["starved"] += _f(row.get("starvation_steps"))
                stats["observed"] += _f(row.get("observed_steps"))
        rates = sorted(s["starved"] / s["observed"] for s in by_robot.values() if s["observed"] > 0)
        if rates:
            out["robot_starvation_rate_max"] = max(rates)
            out["robot_starvation_rate_std"] = statistics.pstdev(rates) if len(rates) > 1 else 0.0
            tail = max(1, int(len(rates) * 0.1))
            out["robot_starvation_rate_cvar90"] = sum(rates[-tail:]) / tail

    runtime_path = output_dir / "runtime_metadata.json"
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
        out["max_steps"] = runtime.get("max_steps", "")
        out["num_trials_per_task"] = runtime.get("num_trials_per_task", "")
    server_path = output_dir / "server_metadata.json"
    if server_path.exists():
        server = json.loads(server_path.read_text())
        out["max_batch_size"] = server.get("max_batch_size", "")
        out["action_horizon"] = server.get("action_horizon", "")

    try:
        from sims.libero.metrics import compute_server_timing_health  # noqa: PLC0415

        health = compute_server_timing_health(output_dir)
        if health:
            out.update(health)
    except Exception:  # noqa: BLE001
        pass
    try:
        from sims.libero.metrics import compute_fairness_metrics  # noqa: PLC0415

        fairness = compute_fairness_metrics(output_dir)
        if fairness is not None:
            out["alpha_observed"] = fairness.get("alpha")
            out["jain_freshness"] = fairness.get("jain_freshness")
            out["jain_starvation"] = fairness.get("jain_starvation")
            rates = fairness.get("starvation_rate") or []
            if rates:
                out["mean_starvation"] = float(sum(rates) / len(rates))
                out["max_starvation"] = float(max(rates))
                out["min_starvation"] = float(min(rates))
    except Exception:  # noqa: BLE001
        pass
    try:
        from sims.libero.metrics import compute_starvation_variance_series  # noqa: PLC0415

        series = compute_starvation_variance_series(output_dir)
        if series is not None:
            out["starvation_variance"] = series["final_starvation_variance"]
    except Exception:  # noqa: BLE001
        pass

    return out
