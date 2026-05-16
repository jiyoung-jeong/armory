"""Plots for inference latency and throughput profiles from configs/inference_profiles.json."""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import style

PROFILES_PATH = pathlib.Path(__file__).parent.parent / "configs" / "inference_profiles.json"
OUTPUT_DIR = pathlib.Path(__file__).parent


def _load_profiles(path: pathlib.Path = PROFILES_PATH) -> dict:
    return json.loads(path.read_text())


def plot_latency(
    profiles: dict,
    hw: str = "l40s",
    ax: plt.Axes | None = None,
    *,
    max_batch_size: int | None = 10,
    output_path: pathlib.Path | None = None,
) -> plt.Axes:
    """Inference latency (s) vs. batch size for each model."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for idx, (model, hw_data) in enumerate(profiles.items()):
        if hw not in hw_data:
            continue
        data = hw_data[hw]
        batch_sizes = [int(k) for k in data]
        latencies = [data[k] for k in data]
        if max_batch_size is not None:
            batch_sizes, latencies = zip(
                *[(b, l) for b, l in zip(batch_sizes, latencies) if b <= max_batch_size]
            )
        ax.plot(
            batch_sizes,
            latencies,
            marker="o",
            color=style.color(idx),
            label=style.model_label(model),
        )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (s)")
    ax.legend()
    style.yonly_grid(ax)

    if standalone:
        fig.tight_layout()
        out = output_path or (OUTPUT_DIR / "latency_vs_batch.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

    return ax


def plot_throughput(
    profiles: dict,
    hw: str = "l40s",
    ax: plt.Axes | None = None,
    *,
    max_batch_size: int | None = 10,
    output_path: pathlib.Path | None = None,
) -> plt.Axes:
    """Throughput (req/s) vs. batch size for each model."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for idx, (model, hw_data) in enumerate(profiles.items()):
        if hw not in hw_data:
            continue
        data = hw_data[hw]
        batch_sizes = [int(k) for k in data]
        throughputs = [int(k) / data[k] for k in data]
        if max_batch_size is not None:
            batch_sizes, throughputs = zip(
                *[(b, t) for b, t in zip(batch_sizes, throughputs) if b <= max_batch_size]
            )
        ax.plot(
            batch_sizes,
            throughputs,
            marker="o",
            color=style.color(idx),
            label=style.model_label(model),
        )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (req/s)")
    ax.legend()
    style.yonly_grid(ax)

    if standalone:
        fig.tight_layout()
        out = output_path or (OUTPUT_DIR / "throughput_vs_batch.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

    return ax


def plot_robot_time_ratio(
    profiles: dict,
    hw: str = "l40s",
    ax: plt.Axes | None = None,
    *,
    chunk_length: int = 10,
    control_hz: float = 20.0,
    max_batch_size: int | None = 10,
    output_path: pathlib.Path | None = None,
) -> plt.Axes:
    """Robot time / GPU time vs. batch size for each model.

    Two curves per model:
      actual:  batch_size * (chunk_time - inf_latency) / inf_latency
      ideal:   batch_size * chunk_time / inf_latency  (no latency overhead)
    """
    chunk_time = chunk_length / control_hz  # seconds

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for idx, (model, hw_data) in enumerate(profiles.items()):
        if hw not in hw_data:
            continue
        data = hw_data[hw]
        batch_sizes = [int(k) for k in data]
        latencies = [data[k] for k in data]
        if max_batch_size is not None:
            batch_sizes, latencies = zip(
                *[(b, l) for b, l in zip(batch_sizes, latencies) if b <= max_batch_size]
            )

        actual = [b * (chunk_time - L) / L for b, L in zip(batch_sizes, latencies)]
        ideal = [b * chunk_time / L for b, L in zip(batch_sizes, latencies)]

        color = style.color(idx)
        label = style.model_label(model)
        ax.plot(
            batch_sizes,
            actual,
            linestyle="-",
            color=color,
            marker="o",
            label=f"{label} (no action overlap)",
        )
        ax.plot(
            batch_sizes,
            ideal,
            linestyle="--",
            color=color,
            marker="o",
            label=f"{label} (perfect action overlap)",
        )

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Robot time / GPU time")
    ax.legend()
    style.yonly_grid(ax)

    if standalone:
        fig.tight_layout()
        out = output_path or (OUTPUT_DIR / "robot_time_ratio_vs_batch.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

    return ax


def plot_combined(
    profiles: dict,
    hw: str = "l40s",
    *,
    chunk_length: int = 10,
    control_hz: float = 20.0,
    max_batch_size: int | None = 10,
    output_path: pathlib.Path | None = None,
) -> plt.Figure:
    """Three-panel figure: latency | throughput | robot time ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    plot_latency(profiles, hw, axes[0], max_batch_size=max_batch_size)
    plot_throughput(profiles, hw, axes[1], max_batch_size=max_batch_size)
    plot_robot_time_ratio(
        profiles,
        hw,
        axes[2],
        chunk_length=chunk_length,
        control_hz=control_hz,
        max_batch_size=max_batch_size,
    )
    fig.tight_layout()
    out = output_path or (OUTPUT_DIR / "inference_profiles.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")
    return fig


def main() -> None:
    style.apply()
    profiles = _load_profiles()
    plot_latency(profiles)
    plot_throughput(profiles)
    plot_robot_time_ratio(profiles)
    plot_combined(profiles)


if __name__ == "__main__":
    main()
