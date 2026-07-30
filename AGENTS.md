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

# Run the eval driver against a server
uv run scripts/run.py --json-path configs/gen/<sweep>/client/<shape>/<n>_robots.json
```

## Architecture

This is a research system for **serving robot manipulation policies (π0.5, GR00T) to many robots
at once and evaluating scheduling algorithms** that decide which robots' observations get batched
onto the GPU each step.

The codebase is a **uv workspace**: the root `armory` package (GPU-side serving) plus the standalone
`armory-client` package. The split exists so the client stays lightweight
(no JAX/CUDA) and installable on robots and Python 3.10:

- **`armory-client`** (`armory_client`) — lightweight client library and the **client↔server
  protocol** (`messages.py`, `schemas.py`, `msgpack_numpy.py`). Owns the wire format;
  `action_chunk_broker.py` turns returned action chunks into per-step actions. No heavy deps.
- **`src/evaluation`** (`evaluation`) — eval harness. `runtime.py` is the env↔agent loop, `envs/`
  holds the LIBERO and mock environments, `agents/` the policy and mock agents, `types.py` the
  `ExperimentConfig` schema, and `metrics.py` the post-run analysis. `toxiproxy.py` injects
  network latency for network-ablation experiments. The driver that owns the fleet — one process
  per robot — is `scripts/run.py`.
- **`src/armory`** — the GPU serving system + scheduling research (depends on JAX/CUDA).
- **`src/armory/backends`** — backend-neutral model/environment types, the serving-policy
  contract, policy resolution, stable picklable factories, and the lightweight mock policy.
- **`src/backends`** — adapters that wrap external policy models behind a common interface
  (`openpi_adapter`, `gr00t_adapter`). `third_party/{openpi,libero,Isaac-GR00T}` are editable submodules.

Backend launch code belongs in `armory.backends`.

### Serving system (`src/armory/serving/`)

`server.py` runs **3 processes** connected by ZMQ (ipc) + shared-memory RawArrays — read the module
docstring at the top of `server.py` before touching it, the topology is non-obvious:
1. **WS main** — FastAPI ASGI app; robots connect over websockets, large observation arrays cross
   process boundaries via ZMQ (never pickled across `mp`).
2. **Scheduler** — collects per-robot `SlotRequest`s, runs the scheduling algorithm, dispatches batches.
3. **GPU** — loads weights, runs batched inference, sends responses straight back to WS main.

The server accepts `POST /reconfigure` to swap the scheduling algorithm without restart. It rebuilds
the scheduler from scratch, so only issue it while no robots are connected (`scripts/run.py` calls it
at startup, before the fleet dials in).

Serving responsibilities are split without changing that topology:
- `server.py` is the stable composition facade (`create_app`, `PolicyServer`).
- `server_runtime.py` owns subprocess startup/shutdown, IPC endpoints, background tasks, and `ServerState`.
- `session.py` owns one robot's WebSocket handshake, warmup, receive/send loops, and disconnect cleanup.
- `routes.py` owns the HTTP control plane and dashboard mount.

### Scheduling (the research surface)

Two related but distinct directories:
- `src/armory/serving/scheduler.py` holds `SCHEDULER_REGISTRY` mapping string names
  (`greedy-deadline`, `lookahead-actions`, `dynamic-action`, `round-robin`, …) to
  scheduler classes. This name is what `SchedulerConfig.scheduling_algorithm` selects.
- `src/armory/scheduling/` contains the algorithm implementations; all subclass
  `RequestScheduler` (`base.py`). When adding an algorithm, implement the class here and register it in
  the registry.

Per-robot priority is **client-declared, not server-configured**: a robot sends its `weight` in the
`ConnectRequest`, it rides along on every `SlotRequest`, and `mirror.Robot.weight` is what a scheduler
reads (`lookahead-actions` tiers batch candidates by it and scales each robot's score by it). So a
heterogeneous fleet is a client-config question — `Robot.weight` in `ExperimentConfig` — and the
server needs no knowledge of which robots matter more.

## Conventions & gotchas

- **Experiment configs** are generated, not hand-written: `scripts/gen_configs.py` writes a
  `server/` + `client/` tree, and `scripts/modal/sweep.py` sweeps the product of the two
  (`server × client × seeds`). Every scheduler/alpha/batch/fleet axis belongs in the generator,
  so the sweeper never needs to know which scheduler reads which knob. Per-robot knobs (horizon,
  weight, latency) are client-side, so a fleet shape that varies them is a client config, not a
  server one. The hand-written files under `configs/client/**` and `configs/server/` predate the
  current schema and no longer validate. `experiments/` holds named experiment
  definitions/results for sweeps; sbatch launchers
  (`scripts/sbatch/`) still use the older inline-axis style — see `scripts/sbatch/PHOENIX_NOTES.md`.
- **Don't edit `requirements-modal*.txt` by hand** — they are generated lockfiles for Modal images.
- `jaxtyping` is kept as a dependency for later use but its runtime shape-checking is not currently
  used (ruff ignores `F722` for its string annotations).
- Tests live next to code in packages (`*_test.py`) and under `tests/` for the root package.

## Modal

**You can test this repo's serving/eval infra by running it on Modal directly — see
[`scripts/modal/README.md`](scripts/modal/README.md).** Cheap CPU "mock" runs
exercise the full server↔client path in a couple of minutes without a local GPU;
`uv run modal run scripts/modal/run.py --server mock` is the quickest smoke test.

Use Modal skills for all GPU work:
- modal-basic-skills: foundational Modal platform knowledge
- modal-gpu-dev: launch interactive GPU sandboxes for debugging and prototyping
- modal-gpu-experiment: write and run training apps for experiments
- sub-agents: orchestrate parallel agents across multiple GPUs

If not installed, you can find them at: https://github.com/modal-projects/modal-auto-research-skills
