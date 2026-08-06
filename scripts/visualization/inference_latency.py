"""Analyze GPU-worker inference latency recorded by Armory servers.

The input may be a single Modal run, a downloaded sweep, or several such
directories. Every ``server/batches.jsonl`` below the supplied roots is treated
as one independent GPU worker. Synthetic/empty batches are excluded.

Example:
    uv run python scripts/visualization/inference_latency.py \
        --run-root experiments/sweeps/inference-latency/20260805_120000
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections.abc import Iterable
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_FILENAMES = {
    "samples": "inference_latency_samples.csv",
    "per_run": "inference_latency_per_run.csv",
    "pooled": "inference_latency_pooled.csv",
    "plot_full_png": "inference_latency_ecdf.png",
    "plot_full_pdf": "inference_latency_ecdf.pdf",
    "plot_png": "inference_latency_tail_ecdf.png",
    "plot_pdf": "inference_latency_tail_ecdf.pdf",
}


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _optional_int(value: Any, *, name: str, path: pathlib.Path) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} in {path} must be an integer, got {value!r}")
    return value


def _server_metadata(case_dir: pathlib.Path) -> tuple[int | None, int | None]:
    path = case_dir / "server_args.json"
    args = _load_json(path)
    if not args:
        return None, None

    server = args.get("server", {})
    if not isinstance(server, dict):
        raise ValueError(f"server in {path} must be a JSON object")
    max_batch_size = server.get("max_batch_size", args.get("max_batch_size"))
    seed = args.get("seed", server.get("seed"))
    max_batch_size = _optional_int(max_batch_size, name="max_batch_size", path=path)
    seed = _optional_int(seed, name="seed", path=path)
    if max_batch_size is not None and max_batch_size <= 0:
        raise ValueError(f"max_batch_size in {path} must be positive")
    return max_batch_size, seed


def _discover_batch_logs(roots: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    found: dict[pathlib.Path, pathlib.Path] = {}
    for root in roots:
        if not root.exists():
            raise ValueError(f"Run root does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Run root is not a directory: {root}")
        for path in root.rglob("server/batches.jsonl"):
            found.setdefault(path.resolve(), path)
    paths = sorted(found.values(), key=lambda path: str(path.resolve()))
    if not paths:
        joined = ", ".join(str(root) for root in roots)
        raise ValueError(f"No server/batches.jsonl files found below: {joined}")
    return paths


def _read_batch_log(path: pathlib.Path, *, worker_id: str) -> list[dict[str, Any]]:
    case_dir = path.parent.parent
    run_id = case_dir.name
    max_batch_size, seed = _server_metadata(case_dir)
    samples: list[dict[str, Any]] = []

    try:
        lines = path.open()
    except OSError as exc:
        raise ValueError(f"Could not open {path}: {exc}") from exc

    with lines:
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")

            batch_size = record.get("batch_size")
            if isinstance(batch_size, bool) or not isinstance(batch_size, int):
                raise ValueError(
                    f"batch_size must be an integer at {path}:{line_number}, got {batch_size!r}"
                )
            if batch_size <= 0:
                continue
            if max_batch_size is not None and batch_size > max_batch_size:
                raise ValueError(
                    f"Observed batch_size={batch_size} above configured max_batch_size="
                    f"{max_batch_size} at {path}:{line_number}"
                )

            duration = record.get("inference_duration")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                raise ValueError(
                    f"inference_duration must be numeric at {path}:{line_number}, got {duration!r}"
                )
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    "inference_duration must be finite and positive at "
                    f"{path}:{line_number}, got {duration!r}"
                )

            samples.append(
                {
                    "worker_id": worker_id,
                    "run_id": run_id,
                    "case_dir": str(case_dir),
                    "source_file": str(path),
                    "line_number": line_number,
                    "batch_id": record.get("batch_id", ""),
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "seed": seed,
                    "inference_start_time": record.get("inference_start_time", ""),
                    "inference_duration_s": duration,
                    "inference_duration_ms": duration * 1000.0,
                }
            )
    return samples


def load_samples(roots: list[pathlib.Path]) -> pd.DataFrame:
    paths = _discover_batch_logs(roots)
    run_id_counts: dict[str, int] = {}
    all_samples: list[dict[str, Any]] = []
    empty_logs: list[pathlib.Path] = []
    for path in paths:
        run_id = path.parent.parent.name
        occurrence = run_id_counts.get(run_id, 0) + 1
        run_id_counts[run_id] = occurrence
        worker_id = run_id if occurrence == 1 else f"{run_id}#{occurrence}"
        samples = _read_batch_log(path, worker_id=worker_id)
        if samples:
            all_samples.extend(samples)
        else:
            empty_logs.append(path)

    if empty_logs:
        print(f"WARNING: ignored {len(empty_logs)} log(s) with no positive-size batches:")
        for path in empty_logs:
            print(f"  {path}")
    if not all_samples:
        raise ValueError("The discovered logs contain no positive-size inference batches")

    samples = pd.DataFrame(all_samples)
    return samples.sort_values(
        ["batch_size", "worker_id", "line_number"], kind="stable", ignore_index=True
    )


def _metrics(values: np.ndarray) -> dict[str, float | int]:
    if not len(values):
        raise ValueError("Cannot summarize an empty sample")
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    return {
        "n": len(values),
        "mean_ms": mean,
        "std_ms": std,
        "cv": std / mean,
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(np.max(values)),
        "p99_over_p50": float(p99 / p50),
    }


def summarize_per_run(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (worker_id, batch_size), group in samples.groupby(["worker_id", "batch_size"], sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "worker_id": worker_id,
                "run_id": first["run_id"],
                "case_dir": first["case_dir"],
                "max_batch_size": first["max_batch_size"],
                "seed": first["seed"],
                "batch_size": batch_size,
                **_metrics(group["inference_duration_ms"].to_numpy(dtype=float)),
            }
        )
    return pd.DataFrame(rows)


def _cluster_bootstrap_p99(
    values_by_worker: list[np.ndarray], *, replicates: int, rng: np.random.Generator
) -> tuple[float, float]:
    estimates = np.empty(replicates, dtype=float)
    num_workers = len(values_by_worker)
    for index in range(replicates):
        selected = rng.integers(0, num_workers, size=num_workers)
        resampled = np.concatenate([values_by_worker[i] for i in selected])
        estimates[index] = np.percentile(resampled, 99)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def _leave_one_worker_out_exceedance(
    values_by_worker: list[np.ndarray],
) -> tuple[float, float, float]:
    if len(values_by_worker) < 2:
        return math.nan, math.nan, math.nan
    rates = []
    for held_out, test_values in enumerate(values_by_worker):
        train_values = np.concatenate(
            [values for index, values in enumerate(values_by_worker) if index != held_out]
        )
        train_p99 = np.percentile(train_values, 99)
        rates.append(float(np.mean(test_values > train_p99)))
    return float(np.mean(rates)), float(np.min(rates)), float(np.max(rates))


def summarize_pooled(
    samples: pd.DataFrame, *, bootstrap_replicates: int, bootstrap_seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(bootstrap_seed)
    for batch_size, group in samples.groupby("batch_size", sort=True):
        worker_groups = list(group.groupby("worker_id", sort=True))
        values_by_worker = [
            worker["inference_duration_ms"].to_numpy(dtype=float) for _, worker in worker_groups
        ]
        worker_p99 = np.asarray(
            [np.percentile(values, 99) for values in values_by_worker], dtype=float
        )
        ci_low, ci_high = _cluster_bootstrap_p99(
            values_by_worker, replicates=bootstrap_replicates, rng=rng
        )
        loo_mean, loo_min, loo_max = _leave_one_worker_out_exceedance(values_by_worker)

        configured_max = sorted({int(value) for value in group["max_batch_size"].dropna().tolist()})
        seeds = sorted({int(value) for value in group["seed"].dropna().tolist()})
        rows.append(
            {
                "batch_size": batch_size,
                "configured_max_batch_sizes": ",".join(map(str, configured_max)),
                "seeds": ",".join(map(str, seeds)),
                "num_runs": len(values_by_worker),
                **_metrics(group["inference_duration_ms"].to_numpy(dtype=float)),
                "worker_p99_min_ms": float(np.min(worker_p99)),
                "worker_p99_max_ms": float(np.max(worker_p99)),
                "worker_p99_range_ms": float(np.ptp(worker_p99)),
                "p99_cluster_bootstrap_ci95_low_ms": ci_low,
                "p99_cluster_bootstrap_ci95_high_ms": ci_high,
                "loo_p99_exceedance_rate": loo_mean,
                "loo_p99_exceedance_rate_min": loo_min,
                "loo_p99_exceedance_rate_max": loo_max,
            }
        )
    return pd.DataFrame(rows)


def _plot_ecdf(
    samples: pd.DataFrame,
    pooled: pd.DataFrame,
    output_dir: pathlib.Path,
    *,
    tail_only: bool,
) -> tuple[pathlib.Path, pathlib.Path]:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    tail_starts: list[float] = []
    tail_ends: list[float] = []

    for batch_size, group in samples.groupby("batch_size", sort=True):
        values = np.sort(group["inference_duration_ms"].to_numpy(dtype=float))
        ecdf = np.arange(1, len(values) + 1, dtype=float) / len(values)
        (line,) = ax.plot(
            values,
            ecdf,
            linewidth=1.8,
            label=f"batch size {batch_size} (n={len(values):,})",
        )
        p99 = float(pooled.loc[pooled["batch_size"] == batch_size, "p99_ms"].iloc[0])
        ax.vlines(
            p99,
            0.9 if tail_only else 0.0,
            0.99,
            color=line.get_color(),
            linestyle="--",
            alpha=0.7,
        )
        ax.plot(p99, 0.99, marker="o", markersize=5, color=line.get_color())
        tail_starts.append(float(np.percentile(values, 90)))
        tail_ends.append(float(np.max(values)))

    if tail_only:
        ax.set_xlim(left=min(tail_starts), right=max(tail_ends) * 1.01)
        ax.set_ylim(0.9, 1.001)
    else:
        ax.set_ylim(0.0, 1.001)
    ax.set_xlabel("GPU-worker inference service time (ms)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(
        "Inference latency tail by batch size"
        if tail_only
        else "Inference latency distribution by batch size"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    suffix = "plot" if tail_only else "plot_full"
    png = output_dir / OUTPUT_FILENAMES[f"{suffix}_png"]
    pdf = output_dir / OUTPUT_FILENAMES[f"{suffix}_pdf"]
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def analyze(
    roots: list[pathlib.Path],
    output_dir: pathlib.Path,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[pathlib.Path]:
    if bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")

    samples = load_samples(roots)
    per_run = summarize_per_run(samples)
    pooled = summarize_pooled(
        samples,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / OUTPUT_FILENAMES["samples"]
    per_run_path = output_dir / OUTPUT_FILENAMES["per_run"]
    pooled_path = output_dir / OUTPUT_FILENAMES["pooled"]
    samples.to_csv(samples_path, index=False)
    per_run.to_csv(per_run_path, index=False)
    pooled.to_csv(pooled_path, index=False)
    full_png, full_pdf = _plot_ecdf(samples, pooled, output_dir, tail_only=False)
    tail_png, tail_pdf = _plot_ecdf(samples, pooled, output_dir, tail_only=True)
    return [samples_path, per_run_path, pooled_path, full_png, full_pdf, tail_png, tail_pdf]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=pathlib.Path,
        nargs="+",
        action="append",
        required=True,
        help="Run root(s) to search recursively; the option may be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory (default: <first-run-root>/_inference_latency).",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    roots = [root for group in args.run_root for root in group]
    output_dir = args.output_dir or roots[0] / "_inference_latency"
    try:
        written = analyze(
            roots,
            output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print("Wrote inference-latency analysis:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
