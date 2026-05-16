# Phoenix Slurm Notes

These notes capture Phoenix-specific Slurm conventions used by the sweep scripts.

## Required Submission Fields

- Phoenix requires a charge account for submitted jobs: `-A <account>` / `--account=<account>`.
- QOS can be selected with `-q <qos>` / `--qos=<qos>`.
- If QOS is omitted, Phoenix defaults to `inferno`.
- `inferno` is the production QOS: paid credits, higher priority, not preempted.
- `embers` is free backfill: shorter walltime, lower priority, eligible for preemption after 1 hour.

## GPU Requests For This Workflow

- The submission form confirmed to work on Phoenix for L40S is:
  - `--partition gpu-l40s`
  - `-G <num_gpus>`
  - `--mem <memory>`
- Heterogeneous jobs are accepted when each component uses that style and components are split with `:`.
- The sweep launcher submits one server component and one client component:
  - server component: `--partition gpu-l40s -G 1 --mem <server_mem>`
  - client component: `--partition gpu-l40s -G 1 --mem <client_mem>`
- The job body uses `srun --het-group=0` for the server and `srun --het-group=1` for the client, so Slurm should isolate `CUDA_VISIBLE_DEVICES` per component.

## Other GPU Request Forms From Docs

- Phoenix docs also mention:
  - `--gres=gpu:<gpu_type>:<gpus_per_node>`
  - `--mem-per-gpu=<memory>`
- In practice, the L40S `--gres=gpu:L40S:<n>` form failed with `Invalid node name specified` in this workspace, while `--partition gpu-l40s -G <n> --mem <memory>` submitted.
- Known GPU type examples from Phoenix docs:
  - `V100`
  - `RTX_6000`
  - `A100`
  - `H100`
  - `H200`
  - `L40S`
  - `rtx_pro_6000_blackwell`
- Some equivalent constraints exist, e.g. `-C L40S`, `-C gpu-l40s`, `-C A100-80GB`, but the sweep launcher uses `--gres`.

## CPU/GPU Ratios

- Phoenix assigns default CPU cores per GPU by GPU type.
- Documented fixed/default ratios:
  - RTX 6000: 6 cores/GPU
  - V100: 12 cores/GPU
  - H100/H200: 8 cores/GPU
  - L40S: 4 cores/GPU
  - A100: default 8 cores/GPU, up to 32 cores/GPU with an explicit CPU request
- Avoid requesting more CPUs than Phoenix permits for the selected GPU.

## Memory

- For this L40S workflow, use plain `--mem` because it matches the confirmed working `salloc`/`sbatch` format.
- Phoenix docs recommend `--mem-per-gpu=<memory>` for GPU jobs generally, but that was not the working form here.
- For CPU-only jobs, use `--mem-per-cpu=<memory>` or `--mem=0` for full node memory.
- The collector job is CPU-only, so it uses `--mem-per-cpu`.

## Dependencies And Collection

- Use `afterany` for result collection so failed case jobs still get summarized:
  - `--dependency=afterany:<job1>:<job2>:...`
- `sbatch --parsable` may return either `jobid` or `jobid;cluster`; dependency strings should use just the numeric job id portion.

## Useful Phoenix Commands

- `pace-quota`: find usable charge accounts and storage quotas.
- `pace-check-queue <qos-or-partition>`: inspect queue/node availability.
- `squeue -u $USER`: see pending/running jobs.
- `sacct -j <jobid> -X`: inspect completed allocations.
- `pace-job-summary <jobid>`: get a Phoenix-formatted job summary.
