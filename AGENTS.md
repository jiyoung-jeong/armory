# AGENTS.md

## Setup

This repo uses **uv** (Python 3.11). Several heavy deps are Linux/CUDA-only (`jax-cuda12`, `triton`, etc.), so a full GPU stack only resolves on Linux; macOS is for client/eval/TUI development.

```bash
git submodule update --init --recursive   # pulls third_party/{libero,openpi}
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

## Common commands

```bash
# Tests (pytest is configured to only collect tests/, armory-client/, and src/; never third_party/)
uv run pytest                                   # full suite
uv run pytest tests/scheduling/mirror_test.py   # single file
uv run pytest tests/scheduling/mirror_test.py::test_name   # single test

# Lint / format (also run via pre-commit on commit)
uv run ruff check --fix
uv run ruff format

# Run a policy server (GPU host)
uv run scripts/serve.py --env LIBERO --max-batch-size 4

# Run the LIBERO eval driver against a server
uv run run-libero <args>
```

## Architecture

This is a research system for **serving robot manipulation policies (π0.5, GR00T) to many robots
at once and evaluating scheduling algorithms** that decide which robots' observations get batched
onto the GPU each step.

The codebase is a **uv workspace**: the root `armory` package (GPU-side serving) plus the standalone
`armory-client` package. The split exists so the client stays lightweight
(no JAX/CUDA) and installable on robots and Python 3.10:

- **`armory-client`** (`armory_client`) — lightweight client library and the **client↔server
  protocol** (`protocol.py`, `messages.py`, `schemas.py`). Owns `SchedulerConfig` and the msgpack/numpy
  wire format. `action_chunkers/` implement how a robot turns returned action chunks into per-step
  actions (sync, temporal ensembling, RTC, naive async). No heavy deps.
- **`src/evaluation`** (`evaluation`) — eval harness. `runtime/` is the env↔agent loop;
  `sims/libero/` is the LIBERO simulation driver (entry point `run-libero`). `toxiproxy.py` injects
  network latency for network-ablation experiments.
- **`src/armory`** — the GPU serving system + scheduling research (depends on JAX/CUDA).
- **`src/armory/backends`** — backend-neutral model/environment types, the serving-policy
  contract, policy resolution, stable picklable factories, and the lightweight mock policy.
- **`src/backends`** — adapters that wrap external policy models behind a common interface
  (`openpi_adapter`, `gr00t_adapter`). `third_party/{openpi,libero,Isaac-GR00T}` are editable submodules.

`scripts/serve_utils.py` is a compatibility shim for older imports and pickles; new backend launch
code belongs in `armory.backends`.

### Serving system (`src/armory/serving/`)

`server.py` runs **3 processes** connected by ZMQ (ipc) + shared-memory RawArrays — read the module
docstring at the top of `server.py` before touching it, the topology is non-obvious:
1. **WS main** — FastAPI ASGI app; robots connect over websockets, large observation arrays cross
   process boundaries via ZMQ (never pickled across `mp`).
2. **Scheduler** — collects per-robot `SlotRequest`s, runs the scheduling algorithm, dispatches batches.
3. **GPU** — loads weights, runs batched inference, sends responses straight back to WS main.

The server accepts `POST /reconfigure` to swap scheduling algorithm / params at runtime without restart.

### Scheduling (the research surface)

Two related but distinct directories:
- `src/armory/serving/scheduler.py` holds `SCHEDULER_REGISTRY` mapping string names
  (`greedy-deadline`, `lookahead-actions`, `action-deficit`, `starvation-fair`, `round-robin`, …) to
  scheduler classes. This name is what `SchedulerConfig.scheduling_algorithm` selects.
- `src/armory/scheduling/` contains the algorithm implementations; all subclass
  `RequestScheduler` (`base.py`). When adding an algorithm, implement the class here and register it in
  the registry.

## Conventions & gotchas

- The LIBERO eval driver lives in `evaluation.sims.libero.run` and is exposed as the `run-libero` console script.
- **Experiment configs** live in `configs/client/**` (per-run robot fleet JSON) and `configs/server/`.
  `experiments/` holds named experiment definitions/results for sweeps. Modal and sbatch launchers
  (`scripts/modal/`, `scripts/sbatch/`) fan these out — see `scripts/sbatch/PHOENIX_NOTES.md`.
- **Don't edit `requirements-modal*.txt` by hand** — they are generated lockfiles for Modal images.
- `jaxtyping` is kept as a dependency for later use but its runtime shape-checking is not currently
  used (ruff ignores `F722` for its string annotations).
- Tests live next to code in packages (`*_test.py`) and under `tests/` for the root package.

## Modal

Use Modal skills for all GPU work:
- modal-basic-skills: foundational Modal platform knowledge
- modal-gpu-dev: launch interactive GPU sandboxes for debugging and prototyping
- modal-gpu-experiment: write and run training apps for experiments
- sub-agents: orchestrate parallel agents across multiple GPUs

If not installed, you can find them at: https://github.com/modal-projects/modal-auto-research-skills
