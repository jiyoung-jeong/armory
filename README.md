# integrated repo

`git submodule update --init --recursive`

`GIT_LFS_SKIP_SMUDGE=1 uv sync`

`uv run scripts/serve.py --env LIBERO --max-batch-size 4`

`uv run run-libero <normal commands>`

# TODOs
acknowledge that much of codebase was evolved from openpi
we will simulate network conditions with toxiproxy on the sender for now but we should think about how to accurately model jitter
prune unused references