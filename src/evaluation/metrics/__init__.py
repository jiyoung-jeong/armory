import logging
import pathlib

from evaluation.metrics.loading import (
    load_episodes,
    load_experiment_config,
    load_server_metadata,
    robot_starvation_rates,
    starvation_variance_series,
)
from evaluation.metrics.plots import (
    generate_actions_left_heatmap,
    generate_client_step_intervals_plot,
    generate_per_robot_success_rate_plot,
    generate_staleness_plot,
    generate_starvation_plot,
    generate_starvation_variance_plot,
    generate_steps_plot,
    generate_success_rate_plot,
)
from evaluation.metrics.server import (
    compute_server_timing_health,
    generate_batch_size_plot,
    generate_request_timing_plot,
    generate_server_batch_gantt_plot,
    generate_server_timings_over_time_plot,
    generate_server_timings_plot,
)
from evaluation.metrics.summary import calculate_metrics

logger = logging.getLogger(__name__)


def generate_all_plots(output_path: pathlib.Path) -> None:
    plotters = [
        generate_client_step_intervals_plot,
        generate_success_rate_plot,
        generate_steps_plot,
        generate_per_robot_success_rate_plot,
        generate_actions_left_heatmap,
        generate_starvation_plot,
        generate_starvation_variance_plot,
        generate_staleness_plot,
        generate_batch_size_plot,
        generate_request_timing_plot,
        generate_server_timings_plot,
        generate_server_timings_over_time_plot,
        generate_server_batch_gantt_plot,
    ]
    for plotter in plotters:
        try:
            plotter(output_path)
        except Exception:
            logger.exception("Plot %s failed; continuing", plotter.__name__)


__all__ = [
    "calculate_metrics",
    "compute_server_timing_health",
    "generate_all_plots",
    "load_episodes",
    "load_experiment_config",
    "load_server_metadata",
    "robot_starvation_rates",
    "starvation_variance_series",
]
