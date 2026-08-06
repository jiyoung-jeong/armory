# Slurm sweeps

`scripts/sbatch/` runs Armory sweeps on a Slurm cluster. Each case is one heterogeneous job: a policy server on one GPU component and the evaluation client on another. Run these commands from the repository root.

## Setup

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
uv sync --extra evaluation --extra serving-web
```

`serving-web` is required even to generate configs or launch, because `scripts/gen_configs.py` and `launch_sweep.py` import `scripts.serve`. The case jobs themselves also need the policy and simulator stacks, so add `--extra server --extra libero` (plus `--extra groot` for GR00T) before submitting.

`run_case.sh` sources `~/.bashrc` and sets `MUJOCO_GL=egl`, so module loads or conda activation belong in your shell profile.

## Generate configs

Sweeps consume a server config tree and a client config tree. Generate both with `scripts/gen_configs.py`:

```bash
uv run python scripts/gen_configs.py \
  --output-dir configs/gen/slurm-test \
  --env libero \
  --schedulers round-robin max-batch lookahead-actions \
  --fleet-sizes 2 4 \
  --shapes hom one_fast \
  --time-limit 60
```

This writes `server/<scheduler>.json` and `client/<shape>/<n>_robots.json`. Seed is not a generation axis; the launcher applies each `--seeds` value to both sides. Use a fresh output directory per experiment because the generator does not remove old files.

## Launch a sweep

```bash
uv run python scripts/sbatch/launch_sweep.py \
  --server-config configs/gen/slurm-test/server \
  --client-config configs/gen/slurm-test/client \
  --seeds 7,42 \
  --output-dir experiments/sweeps/slurm \
  --time 2:00:00 \
  --submit-collector
```

Either side accepts one JSON file or a directory of them. The launcher submits every server x client x seed combination and writes everything under `<output-dir>/<stamp>/<run_id>/`:

```text
experiments/sweeps/slurm/20260805_120000/
├── server=lookahead-actions__client=hom_4_robots__seed=7/
│   ├── server_args.json      # scripts.serve payload
│   ├── client_args.json      # scripts.run payload
│   ├── case.json             # sweep axes for this case
│   ├── submit_cmd.json       # exact sbatch argv, used by --requeue
│   ├── logs/                 # server and client stdout/stderr
│   ├── output/               # client artifacts, plus output/server/
│   └── result.json           # status and summary, written by the case job
├── jobs_<stamp>.csv
└── collector.json            # only with --submit-collector
```

Add `--dry-run` to materialize configs and print the sbatch commands without submitting.

## Cluster flags

`--cluster` selects the resource request: `skynet` (default, `overcap` partition, L40S server + A40 client), `ice` (L40S server, V100 or L40S client scaled by fleet size), and `pace` (L40S server + V100 client, `embers` QOS). PACE requires `--account`; see `PHOENIX_NOTES.md`.

Sizing knobs: `--time`, `--server-mem` (default `32G`), `--client-mem` (default `128G`), `--cpus-per-robot` (default 2, client CPUs are `max(8, robots * cpus_per_robot)`), and `--qos`.

## Collect results

With `--submit-collector` a dependent job aggregates results once every case finishes. Run it by hand at any time:

```bash
uv run python scripts/sbatch/collect_results.py --run-dir experiments/sweeps/slurm/<stamp>
```

This writes `sweep_results_<stamp>.csv` and plots under `<run_dir>/plots/`. Pass `--no-plots` to skip plotting.

## Requeue failures

Resubmit every case whose `result.json` is missing or not `status=ok`, reusing its recorded sbatch command:

```bash
uv run python scripts/sbatch/launch_sweep.py --requeue experiments/sweeps/slurm/<stamp> --dry-run
uv run python scripts/sbatch/launch_sweep.py --requeue experiments/sweeps/slurm/<stamp>
```

Old `result.json` and `logs/` are renamed with a `.previous_<stamp>` suffix, and the requeued job ids land in `requeue_<stamp>.csv`.

## Run one case manually

Any prepared case directory can be resubmitted with the command recorded at launch time. `run_case.sh` needs the heterogeneous allocation, so use the stored argv rather than a bare `sbatch`:

```bash
jq -r .shell experiments/sweeps/slurm/<stamp>/<run_id>/submit_cmd.json | bash
```

The script picks a free port, starts `scripts.serve`, waits up to 900s for `/metadata`, runs `scripts.run` under `srun --het-group=1`, then writes `result.json`.
