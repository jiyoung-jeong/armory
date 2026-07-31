# Running and testing on Modal

Modal can exercise the complete server-to-client path without a local GPU.
`mock` mode is the inexpensive smoke test; `gpu` mode runs the real experiment.
Before launching, `modal profile current` should print an authenticated profile.

## Modes and resources

A run uses separate policy-server and client containers connected through a
`modal.forward` TCP tunnel. `--mode` selects both sides together:

| mode | server | client | purpose |
| --- | --- | --- | --- |
| `gpu` | π0.5/GR00T on one L40S | LIBERO on one T4 | real evaluation |
| `mock` | mock policy on CPU | mock environments on CPU | serving/eval smoke test |
| `runtime` | no server | LIBERO on one T4 | environment/runtime debugging |

Workers use Modal's `us-east` region. A real server requests 4 CPUs and 16 GiB of
host memory. A LIBERO client requests one CPU per robot plus two buffer CPUs
(capped at 16), and `max(16 GiB, 3 GiB x allocated CPUs)` of host memory.
Single-case workers have a two-hour timeout, pooled L40S workers have a 12-hour
timeout, and client subprocesses have a separate 90-minute safety timeout.

## Single cases

`run.py` launches one fleet against one scheduler:

```bash
# Cheap end-to-end CPU smoke test.
uv run modal run scripts/modal/run.py --mode mock

# Real policy on an L40S driving LIBERO on a T4.
uv run modal run scripts/modal/run.py \
  --mode gpu \
  --server-config configs/server/libero.json \
  --client-config configs/client/libero/short.json

# LIBERO without a policy server.
uv run modal run scripts/modal/run.py \
  --mode runtime \
  --client-config configs/client/libero/short.json
```

`--client-config` accepts an `evaluation.types.ExperimentConfig` (an already
wrapped `scripts.run.Args` payload also works). `--server-config` accepts a
`scripts.serve.Args`. Outputs download beneath `--output-dir`, which defaults to
`runs/`.

## Scheduler sweeps

The normal sweep is the product `server configs x client configs x seeds`.
Generate server/fleet axes first; the sweep deliberately does not reinterpret
scheduler, alpha, or batch-size filenames:

```bash
uv run python scripts/gen_configs.py \
  --output-dir configs/gen/smoke \
  --env mock --server-env libero \
  --schedulers greedy-deadline lookahead-actions \
  --fleet-sizes 2 4 --shapes one_fast

uv run modal run scripts/modal/sweep.py \
  --mode mock \
  --server-config configs/gen/smoke/server \
  --client-config configs/gen/smoke/client \
  --seeds 7 \
  --max-concurrent-cases 100 \
  --output-dir experiments/sweeps/smoke
```

Both config flags accept one `.json` file or a directory, searched recursively.
Seed remains a sweep flag because it is copied onto both server and client
configs. A stable `--stamp` makes a rerun use the same run root.

### Five-minute LIBERO grid

The committed `configs/client/sim_sweep` tree contains homogeneous, one-fast,
and half-fast/half-slow fleets at 2, 4, 6, 8, and 10 robots. They run for five
minutes and use `task_subset_size=2`, so seeds 1, 2, and 3 select the same
LIBERO-10 task pairs as the paper evaluation.

Generate the server side, then combine it with those clients:

```bash
uv run python scripts/gen_configs.py \
  --output-dir configs/gen/libero_mbs3 \
  --env libero \
  --schedulers round-robin max-batch lookahead-actions \
  --max-batch-sizes 3

uv run modal run scripts/modal/sweep.py \
  --mode gpu \
  --server-config configs/gen/libero_mbs3/server \
  --client-config configs/client/sim_sweep \
  --seeds 1,2,3 \
  --action-horizon-multiplier 1,3,5 \
  --server-pool-size 5 \
  --server-start-timeout-minutes 60 \
  --stamp libero_5min_paper \
  --output-dir experiments/sweeps/libero_5min
```

The compatibility `--action-horizon-multiplier` axis applies only to
`lookahead-actions`. For each multiplier, the sweep copies the client config and
scales `Robot.weight` for robots on that fleet's shortest execution horizon;
other robots retain their configured weight. Baseline schedulers still run once
per client and seed. Thus the example creates 225 cases: 90 baseline cases plus
135 lookahead cases.

### Persistent GPU server pools

GPU sweeps group cases by settings that require a server restart, including the
model/checkpoint, seed, max batch size, sampling steps, and boot-only parameters.
They then balance each compatible group over persistent-server shards. An L40S
loads its policy once, runs one T4 client at a time, and reconfigures the
scheduler between clients. `--server-pool-size` is an elastic upper bound (at
most 6), not an all-workers-ready barrier: each allocated L40S starts its shard
while other shards remain queued.

A client is launched only after its lane answers `/metadata`. Before it
connects, `/reset` establishes a clean run boundary and swaps the scheduler.
The next case waits for the current client to exit. Completed cases are
fenced again, given their own slice of the pooled server telemetry, and have all
plots regenerated before they are checkpointed to the artifacts volume. Replaying
a preempted shard with the same stamp skips successful cases. A failed client or
artifact finalization forces a clean server restart before that lane continues.

`--server-start-timeout-minutes` measures model/server startup after an L40S is
allocated; time in Modal's GPU queue does not consume it.
`--max-concurrent-cases` applies only to CPU mock sweeps. More pool lanes improve
throughput but split a seed's policy-RNG stream across servers; use one lane per
cold seed group when exact persistent-server RNG ordering is more important.

The sweep writes `cases_<stamp>.csv`, `sweep_results_<stamp>.csv`, and downloaded
artifacts under `--output-dir/<stamp>`. GPU sweeps print the corresponding
`scripts/visualization/plot_libero_sweep.py` command, which validates the cases
and writes only the aligned fast/slow tier breakdown. Mock sweeps print the
generic `scripts/visualization/plot_sweep.py` command.

## Config schemas

Generated files use the current nested server schema:

```jsonc
// scripts.serve.Args
{
  "model": "pi05",
  "env": "libero",
  "port": 8080,
  "server": {
    "max_batch_size": 3,
    "scheduler": {"scheduling_algorithm": "lookahead-actions", "alpha": 1.0},
    "engine": {"num_steps": 10}
  }
}

// evaluation.types.ExperimentConfig
{
  "environment": {"kind": "libero", "task_suite_name": "libero_10"},
  "seed": 7,
  "time_limit": 60.0,
  "robots": [{"weight": 1.0}, {"weight": 1.0}]
}
```

Prefer `scripts/gen_configs.py` over older hand-written experiment trees; many
legacy files under `configs/` still use the pre-refactor schema.

## LIBERO source modes

The default image clones the public LIBERO fork at the revision pinned in
`images.py`, so a normal Modal user does not need the local submodule. To test
edits to the local LIBERO checkout instead:

```bash
git submodule update --init third_party/libero
ARMORY_MODAL_LIBERO_SOURCE=local uv run modal run scripts/modal/run.py --mode runtime
```

When advancing LIBERO, update both the `third_party/libero` gitlink and
`LIBERO_REVISION` in `images.py` to the same commit.

## Gotchas

- Only JSON and primitive values cross the Modal boundary. Do not pass pydantic
  `serve.Args` objects to a Modal method.
- Workers invoke `python -m scripts.serve` and `python -m scripts.run`; module
  invocation keeps package imports stable inside the image.
- The first image build and checkpoint load are slow. Cached mock runs typically
  finish in a few minutes; a cold real-policy server can take about ten minutes.
- `scripts/modal/serve.py` is a separate long-lived, memory-snapshotted service
  for manual attachment. It is not one of the sweep workers.
