"""Modal images for the evaluation, serving, and LIBERO profiles."""

from __future__ import annotations

import pathlib

import modal

REMOTE_ROOT = pathlib.Path("/app")
CHECKPOINT_VOLUME_PATH = "/checkpoints"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

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
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
}


def _sync(image: modal.Image, *extras: str) -> modal.Image:
    """Sync the root lockfile, leaving local Python packages to Modal mounts.

    The client is a local path dependency in the root project. It is excluded
    from the environment install because ``uv_sync`` only copies the root
    metadata and lockfile into its build context; the client source is added
    immediately afterward.
    """
    return image.uv_sync(
        uv_project_dir=str(REPO_ROOT),
        extras=list(extras),
        extra_options=(
            "--no-install-package armory-client "
            "--no-install-package openpi "
            "--no-install-package openpi-client "
            "--no-install-package gr00t "
            "--no-install-package libero"
        ),
    )


def _add_repo_sources(image: modal.Image, *modules: str) -> modal.Image:
    return (
        image.add_local_python_source(*modules)
        .add_local_dir(str(REPO_ROOT / "configs"), remote_path=str(REMOTE_ROOT / "configs"))
        .add_local_dir(str(REPO_ROOT / "scripts"), remote_path=str(REMOTE_ROOT / "scripts"))
    )


def _bake_libero_config(image: modal.Image) -> modal.Image:
    lines = [
        "assets: /root/libero/libero/assets",
        "bddl_files: /root/libero/libero/bddl_files",
        "benchmark_root: /root/libero/libero",
        "datasets: /root/libero/datasets",
        "init_states: /root/libero/libero/init_files",
    ]
    args = " ".join(f"'{line}'" for line in lines)
    return image.run_commands(
        "mkdir -p /root/.libero",
        f"printf '%s\\n' {args} > /root/.libero/config.yaml",
    )


def _add_libero_data(image: modal.Image) -> modal.Image:
    for name in ("bddl_files", "init_files", "assets"):
        image = image.add_local_dir(
            str(REPO_ROOT / "third_party/libero/libero/libero" / name),
            remote_path=f"/root/libero/libero/{name}",
        )
    return image


_cuda_base = modal.Image.from_registry(
    "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
).apt_install("git", "build-essential", "clang", "cmake")

gpu_server_image = _add_repo_sources(
    _sync(
        _cuda_base.pip_install(
            "torch==2.7.1", extra_index_url="https://download.pytorch.org/whl/cu124"
        )
        .pip_install(
            "jax[cuda12]==0.5.3",
            find_links="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        )
        .env(_CUDA_SERVER_ENV)
        .workdir(str(REMOTE_ROOT)),
        "server",
    ),
    "armory",
    "armory_evaluation",
    "armory_client",
    "openpi",
    "openpi_client",
    "openpi_adapter",
    "gr00t",
    "gr00t_adapter",
)

gpu_libero_client_image = _add_libero_data(
    _add_repo_sources(
        _sync(
            _bake_libero_config(_cuda_base.apt_install(*_EGL_APT).env(_LIBERO_CLIENT_ENV)).workdir(
                str(REMOTE_ROOT)
            ),
            "evaluation",
            "libero",
        ),
        "armory",
        "armory_evaluation",
        "armory_client",
        "libero",
    )
)

cpu_mock_image = _add_repo_sources(
    _sync(
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .env({"MPLBACKEND": "Agg"})
        .workdir(str(REMOTE_ROOT)),
        "evaluation",
    ),
    "armory",
    "armory_evaluation",
    "armory_client",
    "openpi_adapter",
    "gr00t_adapter",
)
