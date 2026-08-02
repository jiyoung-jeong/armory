<h1 align="center">Armory</h1>

<p align="center"><strong>Action Chunk Scheduling for Batched Robot Policy Serving</strong></p>

<p align="center">
  <a href="https://rbansal.dev">Rohan Bansal</a><sup>*</sup>,
  <a href="https://davidhe137.github.io/">David He</a><sup>*</sup>,
  <a href="https://nadunranawaka1.github.io/">Nadun Ranawaka Arachchige</a>,
  <a href="https://chenzheny.github.io/">Zhenyang Chen</a>,
  <a href="https://skim743.github.io/">Soobum Kim</a>,
  <a href="https://kexinrong.github.io/">Kexin Rong</a><sup>†</sup>,
  <a href="https://faculty.cc.gatech.edu/~danfei/">Danfei Xu</a><sup>†</sup>
</p>

<p align="center">
  Georgia Institute of Technology<br>
  <sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Equal advising
</p>

<p align="center">
  <a href="https://gatech-rl2.github.io/actionchunkscheduling/pdf/paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square" alt="Paper"></a>
  <a href="https://github.com/GaTech-RL2/armory"><img src="https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github" alt="Code"></a>
  <img src="https://img.shields.io/badge/Twitter-coming_soon-1d9bf0?style=flat-square&logo=x" alt="Twitter, coming soon">
  <img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b?style=flat-square&logo=arxiv" alt="arXiv, coming soon">
</p>

Armory serves one robot policy to many robots from a remote GPU. It tracks each robot's action queue and schedules batched inference to reduce starvation when robots consume actions at different rates.

This repository contains the server, robot clients, policy adapters, and evaluation code used in the paper. 

Try commanding 10 robots at once using a single cloud-served Pi-05 model!
```bash
uv run modal run scripts/modal/run.py --mode gpu --client-config ./configs/modal_10_robots_libero.json
```
See the Modal deployment section below for more info.

## Setup

Armory requires Python 3.11 and [uv](https://docs.astral.sh/uv/). Initialize the pinned third-party repositories, then install the base environment:

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

## Repository structure

- `src/armory/` contains the serving engine, schedulers, and policy server.
- `armory-client/` is the lightweight client package used by each robot.
- `src/backends/` contains the OpenPI and NVIDIA Isaac GR00T adapters.
- `src/evaluation/` contains robot runtimes, LIBERO integration, metrics, and result saving.
- `configs/` stores server and fleet configurations.
- `scripts/` contains local, Modal, Slurm, and plotting entry points.
- `third_party/` contains the pinned OpenPI, LIBERO, and Isaac GR00T repositories.

## Deployment

### Modal

For GPU evaluations, Modal runs the policy server and LIBERO client in separate containers. See the [Modal deployment guide](scripts/modal/README.md) for image setup, single runs, and sweeps.

```bash
uv run modal setup
uv run modal run scripts/modal/run.py \
  --mode gpu \
  --client-config configs/modal_single_robot_libero.json
```

### Local or self-hosted

`scripts/serve.py` runs the policy server and `scripts/run.py` launches a client fleet. It is recommended to use the self-host architecture in a cluster environment where GPU and CPU nodes are available for the policy/client servers.

To run with a server GPU:

```bash
uv sync --extra evaluation --extra serving-web

# Terminal 1: policy server
uv run python -m scripts.serve --port 8080 --model PI05 --env LIBERO

# Terminal 2: one robot
uv run python -m scripts.run \
  --host 127.0.0.1 \
  --port 8080 \
  --output-dir output/demo \
  --overwrite
```

When the client runs on another machine, replace `127.0.0.1` with the server's reachable address and make sure port 8080 is open.

If no GPU is present, you may run in mock mode, which has no real robot rollouts but will simulate the GPU and client communication.

```bash
uv run python -m scripts.serve --port 8080 policy:mock
```

## Acknowledgments

Parts of Armory's policy-serving stack were adapted from [OpenPI](https://github.com/Physical-Intelligence/openpi). We thank the Physical Intelligence team for releasing OpenPI and its model checkpoints. Our evaluations also build on [LIBERO](https://libero-project.github.io/) and [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T).
