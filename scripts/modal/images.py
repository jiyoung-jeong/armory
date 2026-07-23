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
    "libegl1",
)

# openpi/openpi-client/gr00t are mounted by explicit path (see
# _add_server_third_party_sources) instead of add_local_python_source, since these
# submodules typically aren't installed in a macOS dev venv. PYTHONPATH must be set
# before any add_local_* call in the image chain (Modal requires add_local_* to be
# last), hence it lives in this env dict applied early rather than alongside the mounts.
_SERVER_OPENPI_SRC = f"{REMOTE_ROOT}/third_party/openpi/src"
_SERVER_OPENPI_CLIENT_SRC = f"{REMOTE_ROOT}/third_party/openpi/packages/openpi-client/src"
_SERVER_GR00T_ROOT = f"{REMOTE_ROOT}/third_party/Isaac-GR00T"

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
    "PYTHONPATH": ":".join([_SERVER_OPENPI_SRC, _SERVER_OPENPI_CLIENT_SRC, _SERVER_GR00T_ROOT]),
}

_LIBERO_CLIENT_ENV = {
    "MPLBACKEND": "Agg",
    "TF_CPP_MIN_LOG_LEVEL": "2",
    "ABSL_FLAGS_VERBOSITY": "0",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    # nvidia/cuda base images only mount the NVIDIA EGL/GL ICD (needed for
    # hardware-accelerated offscreen rendering) into the container when
    # "graphics" is requested here; the image default (compute,utility) omits
    # it, causing EGL to silently fall back to Mesa's software llvmpipe
    # renderer with no error or warning.
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
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
    # logging_config/utils are standalone modules under src/ that scripts/run.py
    # and scripts/serve.py import by bare name. Mount them by explicit file path
    # (not add_local_python_source): the bare name `utils` is ambiguous — the
    # repo also has scripts/utils.py — and the resolver picks the wrong one, so
    # pin the src/ copies onto /root (already on sys.path via the package mounts).
    return (
        image.add_local_python_source(*modules)
        .add_local_dir(str(REPO_ROOT / "configs"), remote_path=str(REMOTE_ROOT / "configs"))
        .add_local_dir(str(REPO_ROOT / "scripts"), remote_path=str(REMOTE_ROOT / "scripts"))
        .add_local_file(str(REPO_ROOT / "src/utils.py"), "/root/utils.py")
        .add_local_file(str(REPO_ROOT / "src/logging_config.py"), "/root/logging_config.py")
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


def _add_nvidia_egl_icd(image: modal.Image) -> modal.Image:
    """Register the NVIDIA EGL backend with libglvnd.

    The nvidia/cuda base images mount the NVIDIA driver's .so files into the
    container at runtime, but ship no libglvnd ICD JSON for them. Without it,
    libglvnd's EGL dispatcher only ever discovers Mesa's software `llvmpipe`
    renderer (the sole vendor with a JSON in /usr/share/glvnd/egl_vendor.d/)
    and silently renders on CPU instead of the GPU, ~10x slower, with no
    error or warning anywhere.
    """
    icd = '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}'
    return image.run_commands(f"echo '{icd}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json")


def _add_libero_data(image: modal.Image) -> modal.Image:
    for name in ("bddl_files", "init_files", "assets"):
        image = image.add_local_dir(
            str(REPO_ROOT / "third_party/libero/libero/libero" / name),
            remote_path=f"/root/libero/libero/{name}",
            # These simulator assets change rarely. Bake them into the image so
            # Modal can reuse the content-addressed image layer instead of
            # re-uploading a live mount on every app deployment.
            copy=True,
        )
    return image


def _add_libero_source(image: modal.Image) -> modal.Image:
    """Bake LIBERO's Python source without duplicating its static data mounts."""
    return image.add_local_dir(
        str(REPO_ROOT / "third_party/libero/libero"),
        remote_path="/root/libero",
        # The three data trees are added by _add_libero_data above. Excluding
        # them here avoids remounting them as part of the Python package.
        ignore=["libero/assets/**", "libero/bddl_files/**", "libero/init_files/**"],
        copy=True,
    )


def _add_server_third_party_sources(image: modal.Image) -> modal.Image:
    """Add openpi/openpi-client/gr00t source by explicit path instead of
    ``add_local_python_source``.

    ``add_local_python_source`` locates a package via the *local* machine's import
    spec, but these submodules are Linux/CUDA-oriented and typically aren't
    installed in a macOS dev venv (only Linux resolves the full `server` extra;
    see root CLAUDE.md). Mounting them by path sidesteps needing a local install.
    Must be called last in the image chain (nothing but other add_local_* calls
    after it) -- PYTHONPATH for these paths is set via _CUDA_SERVER_ENV instead.
    """
    return (
        image.add_local_dir(
            str(REPO_ROOT / "third_party/openpi/src"), remote_path=_SERVER_OPENPI_SRC
        )
        .add_local_dir(
            str(REPO_ROOT / "third_party/openpi/packages/openpi-client/src"),
            remote_path=_SERVER_OPENPI_CLIENT_SRC,
        )
        .add_local_dir(str(REPO_ROOT / "third_party/Isaac-GR00T"), remote_path=_SERVER_GR00T_ROOT)
    )


_cuda_base = modal.Image.from_registry(
    "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
).apt_install("git", "build-essential", "clang", "cmake")

_cuda_runtime_base = modal.Image.from_registry(
    "nvidia/cuda:12.2.0-runtime-ubuntu22.04", add_python="3.11"
)

gpu_server_image = _add_server_third_party_sources(
    _add_repo_sources(
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
        "evaluation",
        "armory_client",
        "openpi_adapter",
        "gr00t_adapter",
    )
)

gpu_libero_client_image = _add_repo_sources(
    _add_libero_data(
        _add_libero_source(
            _sync(
                _bake_libero_config(
                    _add_nvidia_egl_icd(
                        _cuda_runtime_base.apt_install(
                            *_EGL_APT, "build-essential", "clang", "cmake"
                        ).env(_LIBERO_CLIENT_ENV)
                    )
                ).workdir(str(REMOTE_ROOT)),
                "evaluation",
                "libero",
            )
        )
    ),
    "armory",
    "evaluation",
    "armory_client",
)

cpu_mock_image = _add_repo_sources(
    # `serving-web` gives the mock policy server (armory.serving.server) its
    # web-serving deps (fastapi/uvicorn/dash) without the GPU `server` stack,
    # so a CPU-only mock server can run.
    _sync(
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .env({"MPLBACKEND": "Agg"})
        .workdir(str(REMOTE_ROOT)),
        "evaluation",
        "serving-web",
    ),
    "armory",
    "evaluation",
    "armory_client",
    "openpi_adapter",
    "gr00t_adapter",
)
