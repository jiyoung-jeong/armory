"""Batch-size-choice story for the max_batch_size ablation.

Two panels in one figure:

  (A) Distribution of *chosen* batch sizes (robots dispatched per batch) at a
      fixed cap (default = largest cap present). EDF/RR concentrate on larger
      batches; lookahead picks smaller ones. Mean batch size per scheduler is
      annotated in the legend.
  (B) System throughput (successes / min) vs the max_batch_size cap. Ties the
      batching behavior to performance: lookahead stays high as the cap grows
      while EDF/RR plateau lower.

Reads ``outputs/server_metrics_history.json`` (the ``batches`` list) for the
chosen batch sizes and ``outputs/results.csv`` for throughput.

Run:
    uv run python scripts/interactive/plot_batch_size_choice.py \\
        experiments/sweeps/interactive/batch_size_ablation_10f
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

SCHEDULER_DISPLAY = {"max-batch": "EDF", "round-robin": "RR"}
# Canonical palette/markers from plot_summary_lines.py (LA uses the ahm=1 blue).
SCHEDULER_COLORS = {
    "max-batch": "#8E6CA8",       # muted purple
    "round-robin": "#5FA86F",     # muted green
    "lookahead-actions": "#6FB0D6",  # blue family (ahm=1)
}
SCHEDULER_MARKERS = {
    "max-batch": "s",
    "round-robin": "D",
    "lookahead-actions": "o",
}
SCHEDULER_ORDER = ["round-robin", "max-batch", "lookahead-actions"]


def _display(sched: str) -> str:
    if sched in SCHEDULER_DISPLAY:
        return SCHEDULER_DISPLAY[sched]
    if sched == LOOKAHEAD:
        return "LA"
    return sched


def _parse_meta(name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in name.split("__"):
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def _batch_sizes(case_dir: pathlib.Path) -> list[int]:
    f = case_dir / "outputs" / "server_metrics_history.json"
    if not f.is_file():
        return []
    try:
        d = json.loads(f.read_text())
    except json.JSONDecodeError:
        return []
    # batches: [batch_id, [robot_names], [...], t1, t2, n, ...]; batch size =
    # number of robots dispatched in that batch.
    return [len(b[1]) for b in d.get("batches", []) if isinstance(b, list) and len(b) > 1]


def _throughput(case_dir: pathlib.Path) -> float | None:
    outputs = case_dir / "outputs"
    rc = outputs / "results.csv"
    rt = outputs / "experiment_args.json"
    if not (rc.is_file() and rt.is_file()):
        return None
    try:
        ec = json.loads(rt.read_text())["experiment_config"]
    except (json.JSONDecodeError, KeyError):
        return None
    hz = float(ec.get("control_hz", 20))
    nr = int(ec.get("num_robots", 0))
    try:
        df = pd.read_csv(rc)
    except Exception:
        return None
    if not {"success", "observed_steps"}.issubset(df.columns):
        return None
    if df["success"].dtype == object:
        succ = df["success"].astype(str).str.lower().eq("true").astype(float).sum()
    else:
        succ = float(df["success"].astype(float).sum())
    steps = float(df["observed_steps"].sum())
    if steps <= 0 or nr <= 0:
        return None
    return succ * hz * nr / steps * 60.0


def _starvation(case_dir: pathlib.Path) -> float | None:
    rj = case_dir / "result.json"
    if not rj.is_file():
        return None
    try:
        result = json.loads(rj.read_text())
    except json.JSONDecodeError:
        return None
    if result.get("status") != "ok":
        return None
    v = result.get("mean_starvation")
    return float(v) if v is not None else None


def _collect(run_dir: pathlib.Path):
    """Returns (sizes, thr, starv) keyed by (cap, sched).

    sizes[(cap, sched)] = list of chosen batch sizes (pooled seeds)
    thr[(cap, sched)]   = list of per-seed throughputs
    starv[(cap, sched)] = list of per-seed mean starvation fractions
    """
    sizes: dict[tuple[int, str], list[int]] = defaultdict(list)
    thr: dict[tuple[int, str], list[float]] = defaultdict(list)
    starv: dict[tuple[int, str], list[float]] = defaultdict(list)
    for case_dir in sorted(run_dir.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("scheduler="):
            continue
        meta = _parse_meta(case_dir.name)
        sched = meta.get("scheduler", "")
        try:
            cap = int(meta.get("max_batch_size", ""))
        except ValueError:
            continue
        sizes[(cap, sched)].extend(_batch_sizes(case_dir))
        t = _throughput(case_dir)
        if t is not None:
            thr[(cap, sched)].append(t)
        s = _starvation(case_dir)
        if s is not None:
            starv[(cap, sched)].append(s)
    return sizes, thr, starv


def _ordered_scheds(present: set[str]) -> list[str]:
    head = [s for s in SCHEDULER_ORDER if s in present]
    return head + sorted(s for s in present if s not in SCHEDULER_ORDER)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir", type=pathlib.Path)
    p.add_argument("--cap", type=int, default=None,
                   help="Cap to show the distribution for (default: largest present).")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="Output path (default: <run_dir>/plots/batch_size_choice.png).")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")
    sizes, thr, starv = _collect(run_dir)
    if not sizes:
        raise SystemExit("No batch data parsed.")

    caps = sorted({cap for cap, _ in sizes})
    dist_cap = args.cap if args.cap is not None else caps[-1]
    scheds = _ordered_scheds({s for _, s in sizes})

    out_path = args.out or (run_dir / "plots" / "batch_size_choice.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(19, 4.8))
    fig.set_facecolor("white")

    # --- Panel A: chosen batch-size distribution at dist_cap ---
    for sched in scheds:
        vals = np.asarray(sizes.get((dist_cap, sched), []), dtype=int)
        if vals.size == 0:
            continue
        bins = np.arange(1, dist_cap + 2)
        counts, _ = np.histogram(vals, bins=bins)
        frac = counts / counts.sum()
        xs = bins[:-1]
        mean_bs = float(vals.mean())
        axA.plot(
            xs, frac, marker=SCHEDULER_MARKERS.get(sched, "o"),
            color=SCHEDULER_COLORS.get(sched, "#444"), linewidth=2.0, markersize=7,
            label=f"{_display(sched)} (mean {mean_bs:.1f})",
        )
        axA.axvline(mean_bs, color=SCHEDULER_COLORS.get(sched, "#444"),
                    linestyle="--", linewidth=1.0, alpha=0.6)
    axA.set_xlabel("Chosen batch size (robots per batch)", fontsize=13)
    axA.set_ylabel("Fraction of dispatched batches", fontsize=13)
    axA.set_title(f"Batch-size choice (cap = {dist_cap})", fontsize=14)
    axA.set_xticks(range(1, dist_cap + 1))

    # --- Panel B: throughput vs cap ---
    for sched in scheds:
        pts = []
        for cap in caps:
            vals = thr.get((cap, sched), [])
            if vals:
                pts.append((cap, float(np.mean(vals))))
        if not pts:
            continue
        xs = [c for c, _ in pts]
        ys = [v for _, v in pts]
        axB.plot(
            xs, ys, marker=SCHEDULER_MARKERS.get(sched, "o"),
            color=SCHEDULER_COLORS.get(sched, "#444"), linewidth=2.0, markersize=7,
            label=_display(sched),
        )
    axB.set_xlabel("Max batch size (cap)", fontsize=13)
    axB.set_ylabel("Cluster throughput (successes / min)", fontsize=13)
    axB.set_title("Throughput vs batch-size cap", fontsize=14)
    axB.set_xticks(caps)

    # --- Panel C: avg starvation vs cap ---
    for sched in scheds:
        pts = []
        for cap in caps:
            vals = starv.get((cap, sched), [])
            if vals:
                pts.append((cap, float(np.mean(vals)) * 100.0))
        if not pts:
            continue
        xs = [c for c, _ in pts]
        ys = [v for _, v in pts]
        axC.plot(
            xs, ys, marker=SCHEDULER_MARKERS.get(sched, "o"),
            color=SCHEDULER_COLORS.get(sched, "#444"), linewidth=2.0, markersize=7,
            label=_display(sched),
        )
    axC.set_xlabel("Max batch size (cap)", fontsize=13)
    axC.set_ylabel("Average starvation rate (%)", fontsize=13)
    axC.set_title("Starvation vs batch-size cap", fontsize=14)
    axC.set_xticks(caps)

    for ax in (axA, axB, axC):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="-", color="#dddddd", linewidth=0.8)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(fontsize=11, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"dist cap={dist_cap}  caps={caps}  schedulers={scheds}")
    print(f"Wrote {out_path} (+ .pdf)")


if __name__ == "__main__":
    main()
