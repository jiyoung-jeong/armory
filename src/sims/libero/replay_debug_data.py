"""
Replay debug data from a saved episode.

This script loads saved debug data (observations, noise, actions) and replays them
in the LIBERO environment. There are two modes:

1. --use_saved_actions: Directly use the saved output_actions from debug data
2. Default: Send saved observation and noise to policy to reproduce actions

Usage:
    # Use saved actions directly (fastest, guaranteed deterministic)
    python examples/libero/replay_debug_data.py \
        --debug_data_dir data/libero/multi_robot_videos/0/0_libero_10_8_success \
        --use_saved_actions

    # Re-infer actions from policy with saved noise (verifies reproducibility)
    python examples/libero/replay_debug_data.py \
        --debug_data_dir data/libero/multi_robot_videos/0/0_libero_10_8_success \
        --host localhost --port 8080

The script will:
1. Load metadata to get task info
2. Load debug data chunks
3. For each chunk, either use saved actions or re-infer from policy
4. Execute the actions in the environment
5. Save a replay video and report success/failure
"""

import argparse
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
from libero.libero import benchmark
from openpi_client.schemas import Action, LiberoObservation
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from openpi_client import websocket_client_policy as _websocket_client_policy
from libero import utils
from libero.env import LiberoSimEnvironment

LIBERO_ENV_RESOLUTION = 256


@dataclass
class ReplayConfig:
    """Configuration for replay."""

    debug_data_dir: pathlib.Path
    host: str
    port: int
    seed: int = 7
    resize_size: int = 224
    num_steps_wait: int = 10
    max_steps: int = 500
    control_hz: int = 20
    action_horizon: int = 50  # Number of actions per chunk (model's action horizon)
    action_dim: int = 7  # Actual robot action dimension (6 DoF + gripper for LIBERO)
    output_video: Optional[str] = None
    use_saved_actions: bool = False  # If True, use saved output_actions directly
    debug_report_path: Optional[str] = (
        None  # Where to write per-chunk debug comparison report (jsonl)
    )


def load_metadata(debug_data_dir: pathlib.Path) -> dict:
    """Load episode metadata."""
    metadata_path = debug_data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path) as f:
        return json.load(f)


def load_debug_chunks(
    debug_data_dir: pathlib.Path,
) -> Tuple[List[dict], Optional[np.ndarray]]:
    """Load all debug data chunks from the .npz file.

    Returns:
        Tuple of (chunks, initial_state) where:
        - chunks: List of debug data chunks
        - initial_state: Initial state array if present in debug data, None otherwise
    """
    debug_file = debug_data_dir / "debug_data.npz"
    if not debug_file.exists():
        raise FileNotFoundError(f"Debug data file not found: {debug_file}")

    # Load npz file
    data = np.load(debug_file, allow_pickle=True)

    # Extract initial state if present
    initial_state = data.get("initial_state")

    # Parse chunks by prefix
    chunks = {}
    for key in data.keys():
        # Skip the initial_state key
        if key == "initial_state":
            continue

        parts = key.split("/", 1)
        if len(parts) == 2:
            chunk_prefix, field_path = parts
            if chunk_prefix not in chunks:
                chunks[chunk_prefix] = {}

            # Reconstruct nested structure
            current = chunks[chunk_prefix]
            field_parts = field_path.split("/")
            for part in field_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[field_parts[-1]] = data[key]

    # Convert to list sorted by chunk index
    chunk_list = []
    for chunk_prefix in sorted(chunks.keys()):
        chunk_list.append(chunks[chunk_prefix])

    return chunk_list, initial_state


def create_observation_from_debug(debug_data: dict, prompt: str, step: int) -> dict:
    """Create observation dict from debug data for policy inference."""
    if "observation" not in debug_data:
        raise ValueError("No observation found in debug data")

    obs = debug_data["observation"]

    # Build observation in the format expected by policy
    return LiberoObservation(
        state=obs["state"],
        image=obs["image"],
        wrist_image=obs.get("wrist_image"),
        prompt=str(obs.get("prompt")),
        step=step,
    )


def get_noise_from_debug(debug_data: dict) -> np.ndarray:
    """Extract noise from debug data."""
    noise = debug_data.get("noise")
    if noise is None:
        raise ValueError("No noise found in debug data")
    return noise


def get_saved_actions_from_debug(debug_data: dict, action_dim: int = 7) -> np.ndarray:
    """Extract saved actions from debug data.

    Args:
        debug_data: Debug data dictionary containing 'actions'
        action_dim: The actual robot action dimension (default: 7 for LIBERO)

    Returns:
        Actions with shape (action_horizon, action_dim)
    """
    actions = debug_data.get("actions")
    if actions is None:
        raise ValueError("No actions found in debug data")

    # Actions should already be correct shape (action_horizon, action_dim)
    # Validate dimension matches expected
    if actions.shape[-1] != action_dim:
        # Might be padded, slice to actual dimension
        actions = actions[:, :action_dim]

    return actions


def _compute_array_diff(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "mean_abs": None,
            "max_abs": None,
        }
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "shape": list(a.shape),
        "mean_abs": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
    }


def _safe_get(d: dict, path: List[str]):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _append_jsonl(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def plot_action_comparison(
    replay_actions: np.ndarray,
    saved_actions: np.ndarray,
    output_path: pathlib.Path,
    action_horizon: int = 50,
    action_dim_names: Optional[List[str]] = None,
) -> None:
    """Plot comparison of replay actions vs saved actions for each dimension.

    Args:
        replay_actions: Actions used during replay, shape (num_steps, action_dim)
        saved_actions: Original saved actions, shape (num_steps, action_dim)
        output_path: Path to save the plot image
        action_horizon: Number of actions per chunk (for grid spacing)
        action_dim_names: Optional names for each action dimension
    """
    num_steps, action_dim = replay_actions.shape

    if action_dim_names is None:
        # Default names for LIBERO 7-DoF actions
        action_dim_names = ["X", "Y", "Z", "RX", "RY", "RZ", "Gripper"]
        if action_dim > len(action_dim_names):
            action_dim_names.extend(
                [f"Dim {i}" for i in range(len(action_dim_names), action_dim)]
            )

    # Create figure with subplots for each action dimension
    fig, axes = plt.subplots(action_dim, 1, figsize=(14, 3 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]

    timesteps = np.arange(num_steps)

    # Calculate differences for summary
    differences = np.abs(replay_actions - saved_actions)
    max_diff = np.max(differences, axis=0)
    mean_diff = np.mean(differences, axis=0)

    for dim in range(action_dim):
        ax = axes[dim]

        # Plot saved actions (original)
        ax.plot(
            timesteps,
            saved_actions[:, dim],
            "b-",
            linewidth=2,
            label="Saved (Original)",
            alpha=0.8,
        )

        # Plot replay actions
        ax.plot(
            timesteps,
            replay_actions[:, dim],
            "r--",
            linewidth=2,
            label="Replay",
            alpha=0.8,
        )

        # Shade the difference
        ax.fill_between(
            timesteps,
            saved_actions[:, dim],
            replay_actions[:, dim],
            alpha=0.3,
            color="purple",
            label="Difference",
        )

        # Title with difference stats
        dim_name = (
            action_dim_names[dim] if dim < len(action_dim_names) else f"Dim {dim}"
        )
        ax.set_title(
            f"{dim_name} | Max Diff: {max_diff[dim]:.6f}, Mean Diff: {mean_diff[dim]:.6f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, which="major")

        # Add vertical lines at action horizon boundaries
        for boundary in range(0, num_steps + 1, action_horizon):
            ax.axvline(x=boundary, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Set x-axis ticks at action horizon boundaries
    xticks = np.arange(0, num_steps + 1, action_horizon)
    axes[-1].set_xticks(xticks)
    axes[-1].set_xlabel("Timestep")

    # Overall title with determinism verdict
    total_max_diff = np.max(differences)
    total_mean_diff = np.mean(differences)
    is_deterministic = total_max_diff < 1e-5

    verdict_color = "green" if is_deterministic else "red"

    fig.suptitle(
        f"Action Comparison: Replay vs Saved\n"
        f"Total Max Diff: {total_max_diff:.8f} | Total Mean Diff: {total_mean_diff:.8f}\n",
        fontsize=14,
        fontweight="bold",
        color=verdict_color,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def replay_episode(
    config: ReplayConfig,
    policy: Optional[_websocket_client_policy.WebsocketClientPolicy],
    console: Console,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Replay a single episode from debug data.

    Args:
        config: Replay configuration
        policy: Policy client (only needed if not using saved actions)
        console: Rich console for output

    Returns:
        Tuple of (success, replay_actions, saved_actions) where:
        - success: True if episode was successful, False otherwise
        - replay_actions: Actions used during replay, shape (num_steps, action_dim)
        - saved_actions: Original saved actions, shape (num_steps, action_dim)
    """
    # Load metadata and chunks
    metadata = load_metadata(config.debug_data_dir)
    chunks, saved_initial_state = load_debug_chunks(config.debug_data_dir)

    console.print(f"[bold blue]Loaded {len(chunks)} debug chunks[/bold blue]")
    console.print(f"  Task Suite: {metadata['task_suite_name']}")
    console.print(f"  Task ID: {metadata['task_id']}")
    console.print(f"  Original Success: {metadata['success']}")
    console.print(
        f"  Mode: {'Using saved actions' if config.use_saved_actions else 'Re-inferring from policy'}"
    )
    console.print()

    # Setup environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[metadata["task_suite_name"]]()
    task = task_suite.get_task(metadata["task_id"])

    # Use the directly saved initial state (preferred)
    initial_state = saved_initial_state[np.newaxis, :]  # Add batch dimension
    console.print("[green]Using saved initial state from debug data[/green]")

    env_raw, task_description = utils._get_libero_env(
        task, LIBERO_ENV_RESOLUTION, seed=config.seed
    )

    env = LiberoSimEnvironment(
        env=env_raw,
        task_description=task_description,
        initial_states=initial_state,
        resize_size=config.resize_size,
        num_steps_wait=config.num_steps_wait,
        max_episode_steps=config.max_steps,
        control_hz=config.control_hz,
    )

    console.print("[bold green]Environment initialized[/bold green]")
    console.print(f"  Task: {task_description}")
    console.print()

    # Reset environment
    env.reset()

    # Pre-extract all saved actions from chunks for comparison
    all_saved_actions = []
    for chunk in chunks:
        saved_chunk_actions = get_saved_actions_from_debug(chunk, config.action_dim)
        all_saved_actions.append(saved_chunk_actions)

    # Replay loop
    frames: List[np.ndarray] = []
    replay_actions_list: List[np.ndarray] = []  # Track actions used during replay
    saved_actions_list: List[np.ndarray] = []  # Track corresponding saved actions
    chunk_idx = 0
    action_idx = 0
    current_actions = None
    current_saved_actions = None  # Saved actions for current chunk
    step = 0
    chunk_debug_report_path: pathlib.Path | None = None
    if config.debug_report_path:
        chunk_debug_report_path = pathlib.Path(config.debug_report_path)
        # Reset report file for a clean run.
        if chunk_debug_report_path.exists():
            chunk_debug_report_path.unlink()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task("[cyan]Replaying episode...", total=None)
        ran_out_of_chunks = False
        last_action = None

        while not env.is_episode_complete() and step < config.max_steps:
            # Get observation for frame capture
            obs = env.get_observation()
            frames.append(obs.image)

            # Determine which action to use
            if ran_out_of_chunks:
                # Keep using last action after running out of chunks
                action = last_action
            elif current_actions is None or action_idx >= len(current_actions):
                if chunk_idx >= len(chunks):
                    # Ran out of chunks
                    if not ran_out_of_chunks:
                        console.print(
                            "[yellow]Warning: Ran out of debug chunks, using last action[/yellow]"
                        )
                        ran_out_of_chunks = True
                    if last_action is not None:
                        action = last_action
                    else:
                        console.print("[red]No actions available, stopping[/red]")
                        break
                else:
                    # Load next chunk
                    chunk_data = chunks[chunk_idx]

                    # Always get saved actions for comparison
                    current_saved_actions = all_saved_actions[chunk_idx]

                    if config.use_saved_actions:
                        # Use saved actions directly
                        current_actions = current_saved_actions.copy()
                    else:
                        # Re-infer from policy with saved noise
                        if policy is None:
                            raise ValueError(
                                "Policy is required when not using saved actions"
                            )
                        noise = get_noise_from_debug(chunk_data)

                        # Create observation from debug data
                        debug_obs = create_observation_from_debug(
                            chunk_data, task_description, step
                        )

                        # Call policy with the saved noise
                        response = policy.infer(
                            obs=debug_obs,
                            noise=noise,
                        )
                        # Policy response already has correct action_dim from post-processing
                        current_actions = response["actions"]

                    chunk_idx += 1
                    action_idx = 0

                    progress.update(
                        task_progress,
                        description=f"[cyan]Step {step}, Chunk {chunk_idx}/{len(chunks)}",
                    )

                    # Get action from the new chunk
                    action = current_actions[action_idx]
                    saved_action = current_saved_actions[action_idx]
                    action_idx += 1
            else:
                # Get next action from current chunk
                action = current_actions[action_idx]
                saved_action = current_saved_actions[action_idx]
                action_idx += 1

            # Track actions for comparison (only when we have valid saved actions)
            if not ran_out_of_chunks:
                replay_actions_list.append(
                    action.copy() if hasattr(action, "copy") else np.array(action)
                )
                saved_actions_list.append(
                    saved_action.copy()
                    if hasattr(saved_action, "copy")
                    else np.array(saved_action)
                )

            # Remember last action for when we run out of chunks
            last_action = action

            # Apply action
            env.apply_action(
                Action(
                    step=step,
                    action=action,
                    action_chunk_index=chunk_idx,
                    index_in_chunk=action_idx,
                )
            )
            step += 1

    # Capture final frame
    if not env.is_episode_complete():
        obs = env.get_observation()
        frames.append(obs.image)

    success = env.current_success

    # Save video
    output_path = config.output_video
    if output_path is None:
        output_path = config.debug_data_dir / "replay.mp4"
    else:
        output_path = pathlib.Path(output_path)

    console.print(f"\n[bold]Saving replay video to {output_path}[/bold]")
    imageio.mimwrite(
        str(output_path),
        [np.asarray(f) for f in frames],
        fps=config.control_hz,
    )

    # Cleanup
    env.close()

    # Convert action lists to arrays
    replay_actions = (
        np.array(replay_actions_list) if replay_actions_list else np.array([])
    )
    saved_actions = np.array(saved_actions_list) if saved_actions_list else np.array([])

    return success, replay_actions, saved_actions


def main():
    parser = argparse.ArgumentParser(
        description="Replay debug data from a saved episode"
    )
    parser.add_argument(
        "--debug_data_dir",
        type=str,
        required=True,
        help="Path to the episode directory containing metadata.json and debug_data/",
    )
    parser.add_argument(
        "--use_saved_actions",
        action="store_true",
        help="Use saved output_actions directly instead of re-inferring from policy",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Policy server host (only needed if not using --use_saved_actions)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Policy server port (only needed if not using --use_saved_actions)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for environment",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Output video path (default: <debug_data_dir>/replay.mp4)",
    )
    parser.add_argument(
        "--debug_report_path",
        type=str,
        default=None,
        help="Where to write the per-chunk debug comparison report (jsonl). Default: <debug_data_dir>/triton_debug_compare.jsonl",
    )

    args = parser.parse_args()

    console = Console()

    config = ReplayConfig(
        debug_data_dir=pathlib.Path(args.debug_data_dir),
        host=args.host,
        port=args.port,
        seed=args.seed,
        max_steps=args.max_steps,
        output_video=args.output_video,
        use_saved_actions=args.use_saved_actions,
        debug_report_path=args.debug_report_path,
    )

    console.print(
        "[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]"
    )
    console.print(
        "[bold magenta]                    Debug Data Replay                       [/bold magenta]"
    )
    console.print(
        "[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]"
    )
    console.print()

    policy = None
    if not config.use_saved_actions:
        # Connect to policy server
        console.print(
            f"[bold]Connecting to policy server at {config.host}:{config.port}...[/bold]"
        )
        policy = _websocket_client_policy.WebsocketClientPolicy(
            robot_id="debug_data_replay",
            host=config.host,
            port=config.port,
        )
        console.print("[green]Connected![/green]")
        console.print()
    else:
        console.print("[bold]Using saved actions (no policy server needed)[/bold]")
        console.print()

    # Run replay
    try:
        success, replay_actions, saved_actions = replay_episode(config, policy, console)

        # Generate action comparison plot
        if len(replay_actions) > 0 and len(saved_actions) > 0:
            plot_path = config.debug_data_dir / "action_comparison.png"
            console.print(
                f"\n[bold]Generating action comparison plot: {plot_path}[/bold]"
            )
            plot_action_comparison(
                replay_actions,
                saved_actions,
                plot_path,
                action_horizon=config.action_horizon,
            )

            # Print summary statistics
            differences = np.abs(replay_actions - saved_actions)
            max_diff = np.max(differences)
            mean_diff = np.mean(differences)
            is_deterministic = max_diff < 1e-5

            console.print()
            console.print("[bold cyan]Action Comparison Summary:[/bold cyan]")
            console.print(f"  Total timesteps compared: {len(replay_actions)}")
            console.print(f"  Max absolute difference: {max_diff:.10f}")
            console.print(f"  Mean absolute difference: {mean_diff:.10f}")
            if is_deterministic:
                console.print(
                    "[bold green]  Verdict: DETERMINISTIC (max diff < 1e-5)[/bold green]"
                )
            else:
                console.print(
                    "[bold red]  Verdict: NON-DETERMINISTIC (max diff >= 1e-5)[/bold red]"
                )

        console.print()
        console.print(
            "[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]"
        )
        if success:
            console.print(
                "[bold green]                    REPLAY SUCCESS ✓                        [/bold green]"
            )
        else:
            console.print(
                "[bold red]                    REPLAY FAILURE ✗                        [/bold red]"
            )
        console.print(
            "[bold magenta]═══════════════════════════════════════════════════════════[/bold magenta]"
        )

    except Exception as e:
        console.print(f"[bold red]Error during replay: {e}[/bold red]")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
