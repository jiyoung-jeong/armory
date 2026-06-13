"""Plot a network ablation sweep: x = latency median (ms) or jitter sigma,
y = system throughput (one figure) and average starvation (another).

Each case dir is ``scheduler=...__experiment=<VAL>__num_robots=...__seed=...``
where <VAL> is the ablation point: ``<N>ms`` for the median sweep or
``sigma_<S>`` for the variance sweep. The x-axis mode is auto-detected from the
experiment tokens. Per case, system throughput is the cluster total
(successes · control_hz · num_robots / observed_steps, ×60 → per minute) and
starvation is the run's mean per-robot starvation rate. Lines are per scheduler,
points are the mean over seeds with ±1 std error bars.

Run:
    uv run python scripts/interactive/plot_network_ablation.py \\
        experiments/sweeps/interactive/net_ablation_medians/20260528_020615
    uv run python scripts/interactive/plot_network_ablation.py \\
        experiments/sweeps/interactive/net_ablation_var/20260528_020845
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOOKAHEAD = "lookahead-actions"

SCHEDULER_DISPLAY = {
    "max-batch": "EDF",
    "round-robin": "RR",
}
# Match the canonical palette/markers from plot_summary_lines.py so the
# network-ablation figures read consistently with the other sim results.
SCHEDULER_COLORS = {
    "max-batch": "#8E6CA8",  # muted purple (no red/yellow)
    "round-robin": "#5FA86F",  # muted green
    "lookahead-actions@ahm=1": "#6FB0D6",  # blue family, tighter spread
    "lookahead-actions@ahm=3": "#3C86B8",
    "lookahead-actions@ahm=5": "#1E5C84",
}
SCHEDULER_MARKERS = {
    "max-batch": "s",
    "round-robin": "D",
    "lookahead-actions@ahm=1": "o",
    "lookahead-actions@ahm=3": "o",
    "lookahead-actions@ahm=5": "o",
}


def _scheduler_display(label: str) -> str:
    if label in SCHEDULER_DISPLAY:
        return SCHEDULER_DISPLAY[label]
    if label.startswith(f"{LOOKAHEAD}@ahm="):
        w = label.split("=")[-1]
        return "LA" if w == "1" else f"LA@{w}"
    return label


def _scheduler_sort_key(label: str) -> tuple[int, float]:
    if label == "round-robin":
        return (0, 0.0)
    if label == "max-batch":
        return (1, 0.0)
    if label.startswith(f"{LOOKAHEAD}@ahm="):
        return (2, float(label.split("=")[-1]))
    return (3, 0.0)


def _parse_case_dirname(name: str) -> dict[str, str]:
    """``scheduler=...__experiment=...__k=v...`` → {key: value}."""
    out: dict[str, str] = {}
    for tok in name.split("__"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def _scheduler_label(meta: dict[str, str]) -> str:
    sched = meta.get("scheduler", "")
    ahm = meta.get("ahm")
    if sched == LOOKAHEAD and ahm is not None:
        return f"{LOOKAHEAD}@ahm={float(ahm):g}"
    return sched


def _x_value(experiment: str) -> tuple[float, str] | None:
    """Return (numeric_x, mode) where mode is 'median' or 'variance'."""
    if experiment.endswith("ms"):
        try:
            return float(experiment[:-2]), "median"
        except ValueError:
            return None
    if experiment.startswith("sigma_"):
        try:
            return float(experiment[len("sigma_") :]), "variance"
        except ValueError:
            return None
    return None


def _case_metrics(case_dir: pathlib.Path) -> tuple[float, float] | None:
    """Return (throughput_per_min, starvation_fraction) for one ok case."""
    result_json = case_dir / "result.json"
    outputs = case_dir / "outputs"
    runtime_json = outputs / "experiment_args.json"
    results_csv = outputs / "results.csv"
    if not (result_json.is_file() and runtime_json.is_file() and results_csv.is_file()):
        return None
    try:
        result = json.loads(result_json.read_text())
        ec = json.loads(runtime_json.read_text())["experiment_config"]
    except (json.JSONDecodeError, KeyError):
        return None
    if result.get("status") != "ok":
        return None

    control_hz = float(ec.get("control_hz", 20))
    num_robots = int(result.get("num_robots") or ec.get("num_robots") or 0)
    try:
        df = pd.read_csv(results_csv)
    except Exception:
        return None
    if not {"success", "observed_steps"}.issubset(df.columns):
        return None
    df = df[df["robot_idx"].notna()] if "robot_idx" in df.columns else df
    if df["success"].dtype == object:
        succ = df["success"].astype(str).str.lower().eq("true").astype(float).sum()
    else:
        succ = float(df["success"].astype(float).sum())
    steps = float(df["observed_steps"].sum())
    if steps <= 0 or num_robots <= 0:
        return None
    # Cluster throughput in successes/sec → ×60 for successes/min.
    thr_per_min = succ * control_hz * num_robots / steps * 60.0
    starv = float(result.get("mean_starvation"))
    return thr_per_min, starv


def _collect(run_dir: pathlib.Path):
    """{(x, scheduler_label): {'thr': [...], 'starv': [...]}}, mode."""
    data: dict[tuple[float, str], dict[str, list[float]]] = defaultdict(
        lambda: {"thr": [], "starv": []}
    )
    mode = None
    for case_dir in sorted(run_dir.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("scheduler="):
            continue
        meta = _parse_case_dirname(case_dir.name)
        xv = _x_value(meta.get("experiment", ""))
        if xv is None:
            continue
        x, mode = xv
        m = _case_metrics(case_dir)
        if m is None:
            continue
        thr, starv = m
        key = (x, _scheduler_label(meta))
        data[key]["thr"].append(thr)
        data[key]["starv"].append(starv)
    return data, mode


def _plot_metric(
    out_path: pathlib.Path,
    data: dict[tuple[float, str], dict[str, list[float]]],
    metric: str,
    *,
    y_scale: float,
    xlabel: str,
    ylabel: str,
    log_x: bool,
    error_bars: bool = True,
) -> None:
    schedulers = sorted({sch for _, sch in data}, key=_scheduler_sort_key)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.set_facecolor("white")

    for sch in schedulers:
        pts = sorted((x, vals) for (x, s), vals in data.items() if s == sch)
        if not pts:
            continue
        xs = [x for x, _ in pts]
        means = [float(np.mean(v[metric])) * y_scale for _, v in pts]
        if error_bars:
            stds = [
                (float(np.std(v[metric], ddof=1)) if len(v[metric]) > 1 else 0.0) * y_scale
                for _, v in pts
            ]
        else:
            stds = None
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker=SCHEDULER_MARKERS.get(sch, "o"),
            color=SCHEDULER_COLORS.get(sch, "#444444"),
            linewidth=2.0,
            markersize=7,
            capsize=4,
            capthick=1.2,
            label=_scheduler_display(sch),
        )

    if log_x:
        ax.set_xscale("log")
        all_x = sorted({x for x, _ in data})
        ax.set_xticks(all_x)
        ax.set_xticklabels([f"{x:g}" for x in all_x])
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", color="#dddddd", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=11, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "run_dir", type=pathlib.Path, help="Sweep run dir (the timestamp dir with case subdirs)."
    )
    p.add_argument(
        "--out-dir", type=pathlib.Path, default=None, help="Output dir (default: <run_dir>/plots)."
    )
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix override (default: <mode>). Use e.g. "
        "'hom_median' to land several runs in one shared folder "
        "without clobbering.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")
    out_dir = (args.out_dir or (run_dir / "plots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data, mode = _collect(run_dir)
    if not data:
        raise SystemExit("No ok cases parsed.")

    if mode == "median":
        xlabel = "Network latency median (ms)"
        prefix = "median"
        log_x = True
        error_bars = False  # medians: no error bars
    else:
        xlabel = "Network latency jitter ($\\sigma$, log-space)"
        prefix = "variance"
        log_x = False
        error_bars = False
    if args.prefix:
        prefix = args.prefix

    _plot_metric(
        out_dir / f"net_ablation_{prefix}_throughput.png",
        data,
        "thr",
        y_scale=1.0,
        xlabel=xlabel,
        ylabel="System throughput (successes / min)",
        log_x=log_x,
        error_bars=error_bars,
    )
    _plot_metric(
        out_dir / f"net_ablation_{prefix}_starvation.png",
        data,
        "starv",
        y_scale=100.0,
        xlabel=xlabel,
        ylabel="Average starvation rate (%)",
        log_x=log_x,
        error_bars=error_bars,
    )

    n_pts = len({x for x, _ in data})
    n_sched = len({s for _, s in data})
    print(f"Mode: {mode}   x-points: {n_pts}   schedulers: {n_sched}")
    print(
        f"Wrote net_ablation_{prefix}_throughput.{{png,pdf}} and "
        f"net_ablation_{prefix}_starvation.{{png,pdf}} to {out_dir}"
    )


if __name__ == "__main__":
    main()
