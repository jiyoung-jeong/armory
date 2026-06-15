"""Thin shim kept for backwards-compatible launch paths.

The eval driver now lives in the ``armory-evaluation`` package and is exposed
as the ``run-libero`` console script (``uv run run-libero ...``). This module
re-exports the public surface so existing callers keep working unchanged:

  - ``uv run scripts/run_libero.py ...``  (README, sbatch, profile, interactive)
  - ``import run_libero`` then ``run_libero.Args`` / ``.ExperimentConfig``  (modal)
"""

from armory_evaluation.sims.libero.run import (
    Args,
    ExecutionHorizon,
    ExperimentConfig,
    NetworkLatency,
    Robot,
    cli,
    main,
)

__all__ = [
    "Args",
    "ExecutionHorizon",
    "ExperimentConfig",
    "NetworkLatency",
    "Robot",
    "cli",
    "main",
]


if __name__ == "__main__":
    cli()
