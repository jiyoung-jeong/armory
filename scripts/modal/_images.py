"""Modal image definitions for the experiment sweep scripts.

Three images, built from two bases:

    _cuda_base ──┬─ gpu_server_image          policy server with real PI05/GR00T
                 │                            weights; no sim code, no EGL.
                 └─ gpu_libero_client_image   LIBERO sim client, hardware EGL
                                              rendering (runs on a small GPU).

    debian-slim ── cpu_mock_image             mock policy and/or mock-env client;
                                              no model/sim stacks, no GPU.

``serve.py`` imports no sim code, so the server image deliberately omits the
EGL apt packages, the MUJOCO_GL env, and the libero data-dir mounts that the
libero client image needs.
"""

from __future__ import annotations

import pathlib
import subprocess

import modal

# Where the repo is mounted inside every container.
REMOTE_ROOT = pathlib.Path("/app")
# openpi reads checkpoints / writes its compilation caches under here; the GPU
# sweep mounts the openpi-checkpoints volume at this path.
CHECKPOINT_VOLUME_PATH = "/checkpoints"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
REQUIREMENTS_FILE = REPO_ROOT / "requirements-modal.txt"
MOCK_REQUIREMENTS_FILE = REPO_ROOT / "requirements-modal-mock.txt"

# Installed separately (CUDA wheels, workspace packages, or not on PyPI) so the
# generated requirements file stays installable on a plain pip.
_MODAL_EXCLUDE = [
    "torch",
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "openpi",
    "openpi-client",
    "gr00t",
    "libero",
    "av",
]

# Local python packages shipped via add_local_python_source. The CUDA images
# get the full set; the mock image omits the third-party model/sim trees
# (openpi/gr00t/libero) because the mock policy never imports them and they'd
# only invite an accidental heavy import.
_FULL_PY_SOURCE = (
    "armory",
    "armory_client",
    "sims",
    "openpi",
    "openpi_client",
    "libero",
    "gr00t",
    "openpi_adapter",
    "gr00t_adapter",
)
_MOCK_PY_SOURCE = (
    "armory",
    "armory_client",
    "sims",
    "openpi_adapter",
    "gr00t_adapter",
)

# bddl/init/assets ship as data, not python source: add_local_python_source
# only copies .py files, and libero auto-discovers these dirs at import time
# via os.path.dirname(__file__) — so they must land next to the package.
_LIBERO_DATA_DIRS = {
    "bddl_files": "/root/libero/libero/bddl_files",
    "init_files": "/root/libero/libero/init_files",
    "assets": "/root/libero/libero/assets",
}


def generate_requirements() -> None:
    """Export a flat requirements.txt for Modal (excludes packages installed separately)."""
    cmd = [
        "uv",
        "export",
        "--no-hashes",
        "--no-dev",
        "--no-emit-workspace",
        *[arg for pkg in _MODAL_EXCLUDE for arg in ("--no-emit-package", pkg)],
        "-o",
        str(REQUIREMENTS_FILE),
        "-q",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    print(f"Wrote {REQUIREMENTS_FILE}")


if modal.is_local():
    generate_requirements()


def _add_repo_sources(image: modal.Image, *, py_source: tuple[str, ...]) -> modal.Image:
    """Mount python packages + the non-package configs/ and scripts/ dirs."""
    return (
        image.add_local_python_source(*py_source)
        .add_local_dir(str(REPO_ROOT / "configs"), remote_path=str(REMOTE_ROOT / "configs"))
        .add_local_dir(str(REPO_ROOT / "scripts"), remote_path=str(REMOTE_ROOT / "scripts"))
    )


def _bake_libero_config(image: modal.Image) -> modal.Image:
    """Pre-write ``~/.libero/config.yaml`` so importing ``libero`` doesn't block
    on its first-run interactive prompt (the container has no stdin).

    Must run before any ``add_local_*`` step — Modal forbids build commands
    after local mounts.
    """
    lines = [
        "assets: /root/libero/libero/assets",
        "bddl_files: /root/libero/libero/bddl_files",
        "benchmark_root: /root/libero/libero",
        "datasets: /root/libero/datasets",
        "init_states: /root/libero/libero/init_files",
    ]
    # printf with %s\n keeps the whole RUN on one Dockerfile line (no embedded
    # newlines, which break Modal's Dockerfile parser).
    args = " ".join(f"'{ln}'" for ln in lines)
    return image.run_commands(
        "mkdir -p /root/.libero",
        f"printf '%s\\n' {args} > /root/.libero/config.yaml",
    )


def _add_libero_data(image: modal.Image) -> modal.Image:
    """Mount the libero bddl/init/assets data dirs next to the libero package."""
    for name, remote_path in _LIBERO_DATA_DIRS.items():
        image = image.add_local_dir(
            str(REPO_ROOT / "third_party/libero/libero/libero" / name),
            remote_path=remote_path,
        )
    return image


# --- CUDA base: shared by the GPU server and GPU libero client ----------------
#
# CUDA 12.2 devel base instead of debian-slim so the nvidia EGL ICD is actually
# present — debian-slim's libegl1 only ships the mesa software path, which is
# why mujoco rendering was so slow even with MUJOCO_GL=egl set.
_cuda_base = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "build-essential",
        # Modal's add_python ships a CPython compiled with clang, so pip uses
        # clang to build C extensions (e.g. evdev). build-essential only
        # provides gcc, so we add clang explicitly.
        "clang",
        "cmake",
    )
    .pip_install("torch==2.7.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "jax[cuda12]==0.5.3",
        find_links="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    )
    .pip_install("av==17.0.0", "pytest==9.0.3")
    .pip_install_from_requirements(str(REQUIREMENTS_FILE))
    .workdir(str(REMOTE_ROOT))
)

# EGL/GL apt packages needed for mujoco hardware rendering on the GPU client.
_EGL_APT = (
    "libgl1",
    "libglib2.0-0",
    "libglfw3",
    "libosmesa6",
    "libegl1",
    "libegl1-mesa-dev",
    "libgles2-mesa-dev",
    "libglvnd-dev",
)

_CUDA_SERVER_ENV = {
    "MPLBACKEND": "Agg",
    "OPENPI_DATA_HOME": CHECKPOINT_VOLUME_PATH,
    "JAX_COMPILATION_CACHE_DIR": f"{CHECKPOINT_VOLUME_PATH}/.cache/jax_compilation",
    "TORCHINDUCTOR_CACHE_DIR": f"{CHECKPOINT_VOLUME_PATH}/.cache/torch_inductor",
    "XLA_FLAGS": "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true",
    "GCLOUD_ANONYMOUS_ACCESS": "True",
    "JAX_PLATFORMS": "cuda",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "ABSL_FLAGS_VERBOSITY": "0",
}

_LIBERO_CLIENT_ENV = {
    "MPLBACKEND": "Agg",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "ABSL_FLAGS_VERBOSITY": "0",
}


# --- GPU server: real policy weights, no sim ----------------------------------
gpu_server_image = _add_repo_sources(
    _cuda_base.env(_CUDA_SERVER_ENV),
    py_source=_FULL_PY_SOURCE,
)

# --- GPU libero client: LIBERO sim with hardware EGL rendering -----------------
gpu_libero_client_image = _add_libero_data(
    _add_repo_sources(
        _bake_libero_config(
            _cuda_base.apt_install(*_EGL_APT).env(
                {
                    **_LIBERO_CLIENT_ENV,
                    # MuJoCo defaults to OSMesa software rendering, far too slow for
                    # the LIBERO sim. Force hardware EGL on the GPU.
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                }
            )
        ),
        py_source=_FULL_PY_SOURCE,
    )
)

# --- CPU mock: mock policy server + mock client in one container --------------
cpu_mock_image = _add_repo_sources(
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements(str(MOCK_REQUIREMENTS_FILE))
    .workdir(str(REMOTE_ROOT))
    .env({"MPLBACKEND": "Agg"}),
    py_source=_MOCK_PY_SOURCE,
)
