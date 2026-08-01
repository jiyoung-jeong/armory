# Running & testing on Modal

**You can evaluate this project by actually running it on Modal.** No local GPU required!

Prerequisites: 

- Ensure you have sourced the `uv` environment with `source .venv/bin/activate`.
- `modal profile current` should print a logged-in profile. If not, run `modal setup`.

All modal accounts (at the time of writing) are given $30 per month to test things out. This is more than enough to give the infra in this project a test spin.

## The three modes

A modal run launches a **policy server** and a simulated **client**, each on its own container. Use `--mode` to select the nature of the evaluation run:

| `--mode`  | server                           | client                       | use it for                             |
| --------- | -------------------------------- | ---------------------------- | -------------------------------------- |
| `gpu`     | real π0.5 / GR00T on an **L40S** | LIBERO sim on a **T4** (EGL) | the real experiment                    |
| `mock`    | mock policy, **CPU**             | mock env, **CPU**           | fast/cheap iteration; timing preserved, no real rollout |
| `runtime` | none   | LIBERO sim on a **T4**       | debugging the env/agent loop alone     |

``gpu`` is what was used for experiments in the paper.
``mock`` is for quick experiments (used to verify scheduler performance).
``runtime`` is for debugging the simulation.

Client CPUs scale with fleet size (1 per robot process, capped at Modal's 16); LIBERO clients also get 3 GiB memory per CPU.

Two entrypoints, same modes:
- **`run.py`**: one case (one fleet, one scheduler).
- **`sweep.py`**: takes the product of a server config dir and a client config dir
  over `CaseRunner.run.map`. `runtime` is rejected here: a sweep with no server has
  no scheduler to sweep.

## Single cases — `run.py`

```bash
# mock policy + mock envs, all CPU. The cheap smoke test.
uv run modal run scripts/modal/run.py --mode mock

# real policy on an L40S driving LIBERO on a T4 (loads a checkpoint; ~10 min).
uv run modal run scripts/modal/run.py --mode gpu --client-config <experiment.json>

# LIBERO sim, no server, to debug the runtime.
uv run modal run scripts/modal/run.py --mode runtime --client-config <experiment.json>
```

For example, to run a real policy on an L40s commanding 10 robots, run:
```bash
uv run modal run scripts/modal/run.py --mode gpu --client-config /coc/flash7/rbansal66/vvla/armory/configs/modal_10_robots_libero.json
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

A sweep is a **product of configs**, so it has no scheduler/alpha/batch flags. Those axes are decided when the configs are generated. Generate a tree first:

```bash
uv run python scripts/gen_configs.py \
  --output-dir configs/gen/smoke \
  --env mock --server-env libero \
  --schedulers greedy-deadline lookahead-actions \
  --fleet-sizes 2 4 --shapes one_fast
```

then sweep it:

```bash
uv run modal run scripts/modal/sweep.py \
  --mode mock \
  --server-config configs/gen/smoke/server \
  --client-config configs/gen/smoke/client \
  --seeds 7 \
  --output-dir experiments/sweeps/smoke \
  --stream-logs
```

Both config flags take a single `.json` file or a directory of them (recursively);
cases are `server × client × seeds`. Seed stays a sweep flag because it lands on
both sides of the product. Writes `sweep_results_<stamp>.csv` and downloaded
artifacts under `--output-dir`, then prints the `plot_sweep.py` command.

`gen_configs.py` collapses variants a scheduler would ignore — `--alphas 0 0.5 1`
yields three `dynamic-action` configs but only one `greedy-deadline`, because
`gen_configs.SCHEDULER_AXES` says `alpha` never reaches it. That is why the
sweeper can stay a dumb product.

### Config schemas

> **The hand-written files under `configs/` are still the pre-refactor schema and
> will fail validation.** Use `gen_configs.py` output (`configs/gen/…`) instead;
> the generated shapes are below.

```jsonc
// server config  ->  scripts/serve.py Args
// serving knobs live under `server`; in mock mode `policy` is overwritten with the mock.
{"model": "pi05", "env": "libero", "port": 8080,
 "server": {"max_batch_size": 5,
            "scheduler": {"scheduling_algorithm": "lookahead-actions", "alpha": 1.0},
            "engine": {"num_steps": 10}}}

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

## Things to Know

- **First run builds images** (torch/JAX/EGL) and is slow; later runs reuse the cache.
  Mock CPU runs are ~1–3 min after that; a real-policy L40S server adds checkpoint
  load + JAX compile (~10 min).
- **`serve.py` launches a separate standalone modal GPU server**, with no connected clients. It publishes a tunnel path should clients need to connect in the future.
