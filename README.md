# Armory

Tools for serving robot manipulation policies to many robots and evaluating
scheduling algorithms.

## Setup

For local server development, local LIBERO evaluation, or editing a third-party
dependency, initialize the submodules and install the project environment:

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

```bash
uv run scripts/serve.py --env LIBERO --max-batch-size 4
uv run run-libero <normal commands>
```

## Modal quick start

Modal users do not need to initialize the LIBERO submodule. The LIBERO client
image clones the public fork at the exact revision pinned in
`scripts/modal/images.py`, then caches that image layer on Modal.

```bash
uv sync --extra dev
uv run modal run scripts/modal/run.py
```

To run a LIBERO Modal evaluation, follow the [Modal guide](scripts/modal/README.md).
If you are modifying `third_party/libero`, initialize that submodule and use the
documented `ARMORY_MODAL_LIBERO_SOURCE=local` override instead.

# TODOs
acknowledge that much of codebase was evolved from openpi
we will simulate network conditions with toxiproxy on the sender for now but we should think about how to accurately model jitter
prune unused references
