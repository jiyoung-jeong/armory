# Running & testing on Modal

**You can test this infra by actually running it on Modal.** You don't need a local
GPU — mock runs are cheap CPU containers that finish in a couple of minutes and
exercise the whole path (server ↔ client over a tunnel). Use them to smoke-test any
change to the serving/scheduling/eval plumbing before reaching for a real GPU run.

Prereq: `modal profile current` should print a profile (auth is already set up on
dev machines). Every run prints a `modal.com/apps/...` dashboard link.

## What runs where

A run is always a **policy server** talking to a **client**, each on its **own
container**, bridged by a `modal.forward` TCP tunnel + a pair of ephemeral
`modal.Dict`s (`setups.py`). Four image-pinned workers, picked automatically:

| Worker (`setups.py`) | Image / GPU | Used when |
|---|---|---|
| `CpuMockServer` | CPU | policy is `Mock` (no weights) |
| `GpuServer` | L40S | real π0.5 / GR00T policy |
| `CpuMockClient` | CPU | env is `MOCK` |
| `LiberoClient` | **T4** | env is `LIBERO` (needs EGL rendering) |

Client CPUs scale with robot count (1 per robot process, via `.with_options`);
`run.py`'s single robot gets 1.

Two entrypoints:
- **`run.py`** — one robot. Lightweight local entrypoint (no heavy imports); runs
  out of the box with no extra deps.
- **`sweep_experiments.py`** — fleet + scheduler sweep. Builds a `Case` per combo
  locally, then `CaseRunner.run.map(...)` orchestrates each on its own containers.

## Single-robot smoke tests — `run.py`

```bash
# mock agent + mock env, no server (one CPU container). Fastest sanity check.
uv run modal run scripts/modal/run.py

# one robot vs a MOCK policy server (CPU server + CPU client). Tests the
# server-spawn + tunnel + /metadata handshake + policy client path.
uv run modal run scripts/modal/run.py --server mock

# one robot vs the REAL pi05 policy on an L40S (loads a checkpoint; ~10 min).
uv run modal run scripts/modal/run.py --server sim

# LIBERO env on a T4 (mock agent, no server): exercises the T4 LiberoClient + EGL.
#   client.json: {"env": 1, "task_suite_name": "libero_10", "max_steps": 50}
uv run modal run scripts/modal/run.py --json-path client.json
```

Client image is chosen from the run.py `env` field (`1`=LIBERO→T4, `2`=MOCK→CPU).
Outputs download to `--output-dir` (default `modal_run_out/<stamp>/`).

## Scheduler sweep — `sweep_experiments.py`

The sweep entrypoint imports `serve`, so launching it locally needs the eval + web
deps once:

```bash
uv sync --extra evaluation --extra serving-web
```

Cheapest end-to-end test — mock policy server + mock-env fleet, all CPU:

```bash
uv run modal run scripts/modal/sweep_experiments.py \
  --server-config server.json \
  --client-config client.json \
  --server-policy mock \
  --schedulers greedy-deadline \
  --seeds 7 \
  --output-dir experiments/sweeps/smoke \
  --stream-logs
```

Real policy on GPU: drop `--server-policy mock` (defaults to `default` = real
checkpoint) and point `--client-config` at a LIBERO experiment. Writes
`cases_<stamp>.csv`, `sweep_results_<stamp>.csv`, and plots under `--output-dir`.

### Config schemas (post-refactor)

The files under `configs/client/**` are the **old** schema — don't reuse them
verbatim. Current shapes:

```jsonc
// server config  ->  serve.Args  (scripts/serve.py)
{"model": "pi05", "env": "libero", "max_batch_size": 1, "port": 8080}

// sweep client config  ->  evaluation.types.ExperimentConfig (flat)
//   env is an INT enum: 1 = LIBERO, 2 = MOCK.  robots is a list (len = num_robots).
{"env": 2, "task_suite_name": "mock", "max_steps": 50, "seed": 7, "robots": [{}, {}]}
```

`--server-policy mock` rewrites the server config's policy to a weightless `Mock`
(CPU); `default` uses a real checkpoint (GPU); `config` keeps the file as written.

## Gotchas

- **Only JSON strings / primitives cross the Modal boundary.** Never pass a pydantic
  `serve.Args` / `run_all.Args` (or a `Case` holding them) into a `@app.method` —
  the bare `serve` module isn't importable on worker containers, so Modal raises
  `DeserializationError`. `Case` stays local; `Case.to_payload()` flattens it to a
  plain dict, and workers receive `*_args_json` strings.
- **Module invocation, not file paths.** Server runs `-m scripts.serve`, the fleet
  client `-m scripts.run_all`, the single client `-m scripts.run` — the module form
  keeps `/app` leading `sys.path` so `src/utils.py` wins over the shadowing
  `scripts/utils.py`.
- **First run builds images** (torch/JAX/EGL) and is slow; subsequent runs reuse the
  cache. Mock CPU runs are ~1–3 min after that; the real-policy L40S server adds
  checkpoint load + JAX compile (~10 min).
- **Timing:** mock (CPU) → cheap & fast, use for infra changes. T4 LiberoClient →
  a few min. L40S real policy → slow/costly, use once mock passes.
