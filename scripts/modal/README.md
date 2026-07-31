# Running & testing on Modal

**You can test this infra by actually running it on Modal.** You don't need a local
GPU — `mock` mode is a couple of cheap CPU containers that finish in minutes and
exercise the whole path (server ↔ client over a tunnel). Use it to smoke-test any
change to the serving/scheduling/eval plumbing before reaching for a real GPU run.

Prereq: `modal profile current` should print a profile (auth is already set up on
dev machines). Every run prints a `modal.com/apps/...` dashboard link.

## The three modes

A run is a **policy server** and a **client**, each on its own container, bridged
by a `modal.forward` TCP tunnel plus a pair of ephemeral `modal.Dict`s. `--mode`
picks both ends at once — it fixes the workers, the agent, and the environment
backend, so the configs can't contradict it:

| `--mode`  | server                           | client                       | use it for                             |
| --------- | -------------------------------- | ---------------------------- | -------------------------------------- |
| `gpu`     | real π0.5 / GR00T on an **L40S** | LIBERO sim on a **T4** (EGL) | the real experiment                    |
| `mock`    | mock policy, **CPU**             | mock envs, **CPU**           | fast/cheap iteration; timing preserved |
| `runtime` | none — the agent returns nulls   | LIBERO sim on a **T4**       | debugging the env/agent loop alone     |

``gpu`` is the real experiment.
``mock`` is for quick experiments.
``runtime`` is for debugging the simulation.
Client CPUs scale with fleet size (1 per robot process plus 2 buffer CPUs, capped
at Modal's 16); LIBERO clients also get 3 GiB per allocated CPU.

All workers are pinned to Modal's `us` region. For a real GPU case, the
server container requests one L40S, 4 CPUs, and 16 GiB of host memory. The single
LIBERO client container for the whole fleet requests one T4, one CPU per robot
plus 2 buffer CPUs (capped at 16), and `max(16 GiB, 3 GiB × CPUs)` of host memory. Single-case
workers have a two-hour Modal timeout, pooled L40S shards have a 12-hour timeout,
and the client subprocess has a separate 90-minute safety timeout.

Two entrypoints, same modes:
- **`run.py`** — one case (one fleet, one scheduler).
- **`sweep.py`** — runs GPU cases through persistent L40S server lanes; mock
  cases still fan out through `CaseRunner.run.map`. `runtime` is rejected here.

## Single cases — `run.py`

```bash
# mock policy + mock envs, all CPU. The cheap smoke test.
uv run modal run scripts/modal/run.py --mode mock

# real policy on an L40S driving LIBERO on a T4 (loads a checkpoint; ~10 min).
uv run modal run scripts/modal/run.py --mode gpu --client-config <experiment.json>

# LIBERO sim, no server, to debug the runtime.
uv run modal run scripts/modal/run.py --mode runtime --client-config <experiment.json>
```

`--client-config` is an `evaluation.types.ExperimentConfig` (a `scripts.run.Args`
payload is also accepted); `--server-config` is a `scripts.serve.Args`, defaulting
to a single-batch π0.5 server. Outputs download to `--output-dir` (default `runs/`).

## Scheduler sweep — `sweep.py`

The sweep entrypoint imports `serve`, so launching it locally needs the eval + web
deps once:

```bash
uv sync --extra evaluation --extra serving-web
```

```bash
uv run modal run scripts/modal/sweep.py \
  --mode gpu \
  --server-config configs/server/libero.json \
  --client-config configs/modal_sweep \
  --schedulers round-robin,max-batch,lookahead-actions \
  --seeds 1,2,3 \
  --max-batch-size 5 \
  --action-horizon-multiplier 1,3,5 \
  --server-pool-size 6 \
  --server-start-timeout-minutes 60 \
  --stamp libero_5min_paper \
  --output-dir experiments/sweeps/libero_5min
```

`--client-config` may be a directory, in which case every `.json`/`.jsonc` under it
becomes a fleet shape. `--server-config` takes a comma-separated list; with several,
config-sensitive schedulers (`lookahead-actions`) run once per config while ordinary
baselines run only against the first. `--action-horizon-multiplier` is a
comma-separated sweep used only by `lookahead-actions`: each value replaces the
shortest key in the server config's `action_horizon_multipliers`, leaving longer
horizons unchanged. The example therefore runs lookahead with `{6: 1, 10: 1}`,
`{6: 3, 10: 1}`, and `{6: 5, 10: 1}`, while each baseline runs once per fleet and
seed. The sweep writes `cases_<stamp>.csv`, `sweep_results_<stamp>.csv`, and plots
under `--output-dir`. Pass `--stamp <name>` to use a stable run root; rerunning
the same grid and stamp resumes successful pooled cases from their checkpoints.

GPU sweeps group cases by settings that require a server restart (including
model/checkpoint, seed, max batch size, sampling steps, and alpha), then split
large compatible groups into balanced persistent-server shards. Each L40S loads
the policy once, runs one T4 client at a time, and hot-swaps the scheduler and
action-horizon multipliers between clients. `--server-pool-size` is an elastic
upper bound, defaults to 6, and has no all-workers-ready barrier: any L40S Modal
can allocate begins its shard immediately while the other shards remain queued.
For the paper grid, three seed groups become six shards of 37 or 38 cases.

The client is launched only after its lane answers `/metadata`, and the next case
does not start until that client exits. Before each client connects, `/prepare`
hot-swaps the scheduler and acknowledges that the scheduler, GPU, and response
router have crossed the previous run boundary. Successful cases are checkpointed
to the artifacts volume as they finish, so a Modal replay after preemption skips
them. A server gets two startup attempts before its remaining shard is failed.

`--server-start-timeout-minutes` is the deadline for model/server startup after
an L40S is allocated; time waiting in Modal's GPU queue does not consume it.
`--max-concurrent-cases` applies only to CPU mock sweeps. Within a lane, policy
RNG continuity matches the persistent main-branch runner. Using six lanes splits
each of the three seed groups across two independently seeded servers; use
`--server-pool-size 3` if exact one-server-per-seed RNG ordering matters more than
the extra parallelism.

`configs/modal_sweep` contains the paper's three fleet shapes (homogeneous,
one-fast, and half-fast/half-slow) at 2, 4, 6, 8, and 10 robots. Each is a
five-minute run. They retain the paper task protocol as well: seeds 1, 2, and 3
select LIBERO-10 task pairs `[0, 1]`, `[2, 3]`, and `[4, 5]`, respectively, and
assign robots to the selected pair round-robin.

### Config schemas

Some older files under `configs/` still use the pre-refactor schema and need
porting to the shapes below.

```jsonc
// server config  ->  scripts/serve.py Args
// `scheduler` is nested; in mock mode `policy` is overwritten with the mock.
{"model": "pi05", "env": "libero", "max_batch_size": 5, "port": 8080,
 "scheduler": {"scheduling_algorithm": "lookahead-actions", "alpha": 1.0,
               "action_horizon_multipliers": {"6": 1.0, "10": 1.0}}}

// client config  ->  evaluation.types.ExperimentConfig
// robots is a list (len = fleet size); task_subset_size pins a fleet to a seeded subset.
{"environment": {"kind": "libero", "task_suite_name": "libero_10",
                 "task_subset_size": 2},
 "seed": 7, "time_limit": 60.0, "robots": [{}, {}]}
```

## LIBERO source modes

The default image build clones the public LIBERO fork at the revision pinned in
`images.py`, so a normal Modal user does **not** need to initialize
`third_party/libero` — a fresh clone plus `uv sync --extra dev` is enough. The
clone and its simulator assets are baked into a cached image layer.

To test edits to the local LIBERO submodule instead:

```bash
git submodule update --init third_party/libero
ARMORY_MODAL_LIBERO_SOURCE=local uv run modal run scripts/modal/run.py --mode runtime
```

When advancing LIBERO, update both the `third_party/libero` gitlink and
`LIBERO_REVISION` in `images.py` to the same commit.

## Gotchas

- **Only JSON / primitives cross the Modal boundary.** Never pass a pydantic
  `serve.Args` (or anything holding one) into a `@app.method` — the bare `serve`
  module isn't importable on worker containers, so Modal raises
  `DeserializationError`. `Case` stays local; `Case.payload()` flattens it.
- **Module invocation, not file paths.** Workers run `-m scripts.serve` and
  `-m scripts.run` — the module form keeps `/app` leading `sys.path` so `src/utils.py`
  wins over the shadowing `scripts/utils.py`.
- **First run builds images** (torch/JAX/EGL) and is slow; later runs reuse the cache.
  Mock CPU runs are ~1–3 min after that; a real-policy L40S server adds checkpoint
  load + JAX compile (~10 min).
- **`serve.py` is separate** — a standalone long-lived GPU server (its own
  `armory-serve` app, GPU memory snapshotting) you attach to by hand. It is not one
  of the three modes and does not share their launch path.
