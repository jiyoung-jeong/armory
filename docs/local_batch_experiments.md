Local LIBERO batch experiments

Use the project Python 3.11 environment with server, libero, evaluation and
serving-web extras. No administrator privileges or global GPU configuration
changes are required. Stop your own existing server on port 8080 first and
check that the selected GPU is idle.

Run from the repository root:

```bash
.venv/bin/python -u -m scripts.local_batch_sweep \
  --phase profile --output output/local_batch_sweep_20260914
.venv/bin/python -u -m scripts.local_batch_sweep \
  --phase repeat --seconds 180 --repeats 3 \
  --output output/local_batch_sweep_20260914
.venv/bin/python -u -m scripts.local_batch_sweep \
  --phase scale --seconds 180 --repeats 3 \
  --output output/local_batch_sweep_20260914
```

Each trial starts a fresh server, waits for metadata readiness, then runs its
client. The repeat phase uses two robots and maximum batches 1/2; scale uses
four robots and maximum batches 1/2/4. Order is alternated or rotated between
repetitions. Seed 7, 20Hz, execution horizon 1..10 and generation steps 10 are
unchanged. LIBERO rendering and inference share the selected GPU. A competing
compute process aborts the trial. CPU or graphics interference still needs to
be considered when interpreting results. GPU memory/utilization samples are
periodic observations, not exact peaks or a kernel trace.

Outputs are never overwritten. Each trial has a manifest, stdout logs, GPU
samples, client results and a policy/server directory. The client option
--server-log-dir links its server directory to those server logs so automatic
server plots work. Use a dedicated server log directory per trial: logs from a
server reused across trials contain repeated batch IDs and need explicit run
filtering. Server log links must be outside the client output directory.

The engine preserves selection-time request_ids and records processed_robot_ids,
processed_request_ids, chunk_ids and processed_observation_steps. Timing analysis
uses the processed IDs to join request arrivals with responses. Legacy logs are
still readable, but selection/processing mismatches cannot be reconstructed by
that legacy join alone.

The profile phase runs a separate 35-second two-robot trial. It enables optional
ARMORY_NVTX=1 annotations using the existing NVIDIA NVTX runtime, without loading
Torch in the scheduler or initializing CUDA there. Ranges cover scheduler
selection, inference (with batch ID/size), and response send (with robot/chunk/request
IDs). Explicit NVTX range IDs allow asynchronous send tasks to overlap safely.
Normal measurements disable annotations.

Nsight Systems is invoked via /usr/local/cuda-13.2/bin/nsys. Collection is delayed
until five seconds after the first real observation, then stopped after roughly
15 seconds. CPU sampling and CPU context-switch tracing are disabled. The server
uses fork-only workers, so fork-before-exec tracing is enabled for this separate
profiling trial. Verify that the resulting report contains CUDA kernels from the
GPU worker and the NVTX ranges before interpreting it. Capture success and timing
numbers from this trial are not substitutes for unprofiled measurements.

Definitions: post-first starvation starts with the first executed model action,
not the first response receipt. A step with a null/default action still advances
the environment. Success rates exclude rollout-time truncations; preserve both
truncations and step-limit failures in reports. infer_batch wall time is not GPU
utilization. request_timestamp starts just before serialization, after observation
creation; response_timestamp is recorded while converting the response to a chunk
under the broker lock. This is not a pure network RTT. Fresh-observation age before
inference is also not the robot's total service wait.

After completed trials, run:

```bash
.venv/bin/python -m scripts.analyze_local_batch_sweep output/local_batch_sweep_20260914
```

The analyzer verifies every saved chunk against processed request and chunk IDs,
exports per-run and per-robot CSVs, and aggregates repetitions without including
profile trials in the comparison. Standard deviations describe run-to-run
variation; three same-seed repeats do not establish a general performance ranking.
