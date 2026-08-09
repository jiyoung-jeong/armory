# Modal deployment

Modal runs Armory's policy server and evaluation clients without requiring a local GPU. Run these commands from the repository root.

## Setup

Initialize the pinned source dependencies, install the local tools needed by the Modal entry points, and authenticate:

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
uv sync --extra evaluation --extra serving-web
uv run modal setup
```

## Modes

| Mode | Policy server | Client | Use |
| --- | --- | --- | --- |
| `gpu` | PI05 or GR00T on an L40S | LIBERO on a T4 | Full policy evaluation |
| `mock` | Profiled mock policy on CPU | Mock environment on CPU | Fast scheduler checks |
| `runtime` | None | LIBERO on a T4 with null actions | Simulator and client debugging |

The mode selects the server, agent, and environment backend. `runtime` is available only for single cases because it has no server or scheduler to sweep.

## Run one case

Use `scripts/modal/run.py` to run one fleet as a single case:

```bash
# CPU-only test
uv run modal run scripts/modal/run.py --mode mock

# Real policy with one LIBERO robot
uv run modal run scripts/modal/run.py \
  --mode gpu \
  --client-config configs/modal_single_robot_libero.json

# LIBERO runtime without a policy server
uv run modal run scripts/modal/run.py \
  --mode runtime \
  --client-config configs/modal_single_robot_libero.json
```

`--client-config` accepts a raw `ExperimentConfig` or a complete `scripts.run.Args` payload. `--server-config` accepts a `scripts.serve.Args` payload. The GPU mode defaults to PI05 with batch size 1. Outputs download beneath `runs/` unless `--output-dir` is set.

For scheduler comparisons, use `sweep.py`. It keeps the server scheduler and the client's reconfiguration request in sync. A single run with a raw client config uses the client's default `greedy-deadline` scheduler.

## Generate experiment configs

Sweeps consume separate server and client config trees. Generate both with `scripts/gen_configs.py`:

```bash
uv run python scripts/gen_configs.py \
  --output-dir configs/gen/modal-test \
  --env mock \
  --schedulers round-robin max-batch lookahead-actions \
  --fleet-sizes 2 4 \
  --shapes hom one_fast \
  --time-limit 60
```

This produces:

```text
configs/gen/modal-test/
├── server/
│   ├── lookahead-actions.json
│   ├── max-batch.json
│   └── round-robin.json
└── client/
    ├── hom/
    │   ├── 2_robots.json
    │   └── 4_robots.json
    └── one_fast/
        ├── 2_robots.json
        └── 4_robots.json
```

Server configs hold the model, scheduler, batch size, and inference settings. Client configs hold the environment, fleet, execution horizons, and run length. Seed is not a generation axis. A sweep applies each requested seed to both the server and client configs.

Available fleet shapes are `hom`, `half_fast_half_slow`, and `one_fast`. Use `--short-horizon-weights 1 3 5` to generate weighted variants for heterogeneous fleets. Run `uv run python scripts/gen_configs.py --help` for the full set of axes. Use a fresh output directory for each experiment because the generator does not remove old files.

For sweeps, use generated configs instead of `configs/server/` and the legacy files under `configs/client/`. The top-level `configs/modal_*_libero.json` files and `configs/client/libero/short.json` remain valid single-run examples.

## Run a sweep

Pass one config file or a directory of JSON files for each side:

```bash
uv run modal run scripts/modal/sweep.py \
  --mode mock \
  --server-config configs/gen/modal-smoke/server \
  --client-config configs/gen/modal-smoke/client \
  --seeds 7,42 \
  --output-dir experiments/sweeps/modal-smoke
```

The sweep runs every compatible server config x client config x seed combination. Since `deficit-round-robin` uses action coverage rather than explicit robot weights, it is paired only with unit-weight clients instead of duplicating `w1`, `w3`, and `w5` cases. Use `--mode gpu` to run the same grid with a real policy server and LIBERO clients. Results, logs, and downloaded artifacts are written under `<output-dir>/<timestamp>/`.

## Measure inference latency

The server records the actual batch size and exact `policy.infer_batch()` duration
for every live batch in `server/batches.jsonl`. To measure p99 latency with live
LIBERO observations, generate a saturated 10-robot workload for batch sizes 1-3:

```bash
uv run python scripts/gen_configs.py \
  --output-dir configs/gen/modal-inference-latency \
  --env libero \
  --schedulers max-batch \
  --max-batch-sizes 1 2 3 \
  --fleet-sizes 10 \
  --shapes hom \
  --time-limit 120
```

Run each batch-size wave sequentially. Each wave uses at most three L40S servers
and three T4 clients under a 10-GPU account limit:

```bash
for batch_size in 1 2 3; do
  uv run modal run scripts/modal/sweep.py \
    --mode gpu \
    --server-config "configs/gen/modal-inference-latency/server/max-batch_b${batch_size}.json" \
    --client-config configs/gen/modal-inference-latency/client/hom/10_robots.json \
    --seeds 1,2,3 \
    --output-dir experiments/sweeps/modal-inference-latency
done
```

After all three waves finish, aggregate the downloaded batch records:

```bash
uv run python scripts/visualization/inference_latency.py \
  --run-root experiments/sweeps/modal-inference-latency
```

The analysis groups by the observed batch size, not the configured maximum. It
reports pooled and per-worker p50, p95, and p99 policy service time and generates
full-distribution and tail ECDFs. This timing includes policy preprocessing, GPU
execution, and output transfer, matching the scheduler's latency measurement; it
excludes queueing and network latency.
