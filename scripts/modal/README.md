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
Client CPUs scale with fleet size (1 per robot process, capped at Modal's 16); LIBERO clients also get 3 GiB per CPU.

Two entrypoints, same modes:
- **`run.py`** — one case (one fleet, one scheduler).
- **`sweep.py`** — fans a scheduler/seed/alpha grid out over `CaseRunner.run.map`.
  `runtime` is rejected here: a sweep with no server has no scheduler to sweep.

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
  --mode mock \
  --server-config <server.json> \
  --client-config <experiment.json | dir/> \
  --schedulers greedy-deadline,lookahead-actions \
  --seeds 7 \
  --output-dir experiments/sweeps/smoke \
  --stream-logs
```

`--client-config` may be a directory, in which case every `.json`/`.jsonc` under it
becomes a fleet shape. `--server-config` takes a comma-separated list; with several,
config-sensitive schedulers (`lookahead-actions`) run once per config while ordinary
baselines run only against the first. Writes `cases_<stamp>.csv`,
`sweep_results_<stamp>.csv`, and plots under `--output-dir`.

### Config schemas

> **The files under `configs/` are still the pre-refactor schema and will fail
> validation.** They need porting to the shapes below.

```jsonc
// server config  ->  scripts/serve.py Args
// `scheduler` is nested; in mock mode `policy` is overwritten with the mock.
{"model": "pi05", "env": "libero", "max_batch_size": 5, "port": 8080,
 "scheduler": {"scheduling_algorithm": "lookahead-actions", "alpha": 1.0}}

// client config  ->  evaluation.types.ExperimentConfig
// robots is a list (len = fleet size); LIBERO assigns distinct tasks from its suite.
{"environment": {"kind": "libero", "task_suite_name": "libero_10"},
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
