import logging
import pathlib
import subprocess
import threading
import time

import modal
import modal.experimental
import requests

log = logging.getLogger(__name__)

app = modal.App("armory-serve")

GPU = "l40s"
REGION = "us-east"
ENV_MODE = "REAL_ACT_100"
MAX_BATCH_SIZE = 5
PORT = 8080
MODEL = "PI05"
SCHEDULING_ALGORITHM = "lookahead-actions"
ALPHA = 2.0
MIN_OBSERVATION_STEP_DIFF = 12

ACTION_HORIZON_MULTIPLIERS = {
    10: 1.0,
    20: 1.0,
}

REPO_ROOT = pathlib.Path(__file__).parent.parent

checkpoint_volume = modal.Volume.from_name("openpi-checkpoints", create_if_missing=True)
CHECKPOINT_VOLUME_PATH = "/checkpoints"

REQUIREMENTS_FILE = REPO_ROOT / "requirements-modal.txt"

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


# NOTE: this is necessary because Modal does not support uv workspaces, which are in the pyproject.toml
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
    print(f"Written {REQUIREMENTS_FILE}")


if modal.is_local():
    generate_requirements()

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
_base = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libglfw3",
        "libosmesa6",
        "libegl1",
        "build-essential",
        "cmake",
    )
    .pip_install("torch==2.7.1", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "jax[cuda12]==0.5.3",
        find_links="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    )
)

image = (
    _base.pip_install("av==17.0.0", "pytest==9.0.3")
    .pip_install_from_requirements(str(REQUIREMENTS_FILE))
    .env(
        {
            "OPENPI_DATA_HOME": CHECKPOINT_VOLUME_PATH,
            "JAX_COMPILATION_CACHE_DIR": f"{CHECKPOINT_VOLUME_PATH}/.cache/jax_compilation",
            "TORCHINDUCTOR_CACHE_DIR": f"{CHECKPOINT_VOLUME_PATH}/.cache/torch_inductor",
            "XLA_FLAGS": "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true",
            "GCLOUD_ANONYMOUS_ACCESS": "True",
            "JAX_PLATFORMS": "cuda",
            "TF_CPP_MIN_LOG_LEVEL": "2",  # to suppress warnings
            "ABSL_FLAGS_VERBOSITY": "0",
        }
    )
    .add_local_python_source(
        "armory",
        "armory_client",
        "openpi",
        "openpi_client",
        "libero",
        "gr00t",
        "openpi_adapter",
        "gr00t_adapter",
        "sims",
    )
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/root/scripts")
    .add_local_dir(str(REPO_ROOT / "configs"), remote_path="/root/configs")
)


@app.cls(
    gpu=GPU,
    image=image,
    volumes={CHECKPOINT_VOLUME_PATH: checkpoint_volume},
    region=[REGION],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=60 * 60,  # seconds, time to wait before scaling down
    timeout=2 * 60 * 60,  # 2 hours
)
class ModalPolicyServer:
    @modal.enter(snap=True)
    def startup(self) -> None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Starting server")

        cmd = [
            "python",
            "/root/scripts/serve.py",
            "--model",
            MODEL,
            "--env",
            ENV_MODE,
            "--max-batch-size",
            str(MAX_BATCH_SIZE),
            "--port",
            str(PORT),
            "--scheduling-algorithm",
            SCHEDULING_ALGORITHM,
            "--alpha",
            str(ALPHA),
            # "--min-observation-step-diff",
            # str(MIN_OBSERVATION_STEP_DIFF),
            "--action-horizon-multipliers",
            *[str(x) for kv in ACTION_HORIZON_MULTIPLIERS.items() for x in kv],
        ]

        def _stream_logs(proc: subprocess.Popen) -> None:
            for line in proc.stdout:
                print(line, end="", flush=True)

        def _start_process() -> subprocess.Popen:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            threading.Thread(target=_stream_logs, args=(proc,), daemon=True).start()
            return proc

        self.process = _start_process()
        while True:
            try:
                requests.get(f"http://localhost:{PORT}/metadata", timeout=5).raise_for_status()
                logger.info("Server ready, snapshot will be taken now.")
                return
            except requests.exceptions.RequestException:
                time.sleep(1)

    @modal.enter(snap=False)
    def open_tunnel(self) -> None:
        self._tunnel_ctx = modal.forward(PORT)
        tunnel = self._tunnel_ctx.__enter__()
        self._url = tunnel.url
        logging.getLogger(__name__).info("Server URL: %s", tunnel.url)

    @modal.asgi_app()
    def stable_endpoint(self):
        from fastapi import FastAPI
        from fastapi.responses import RedirectResponse

        stable = FastAPI()

        @stable.get("/metadata")
        def metadata():
            meta = requests.get(f"http://localhost:{PORT}/metadata").json()
            meta["tunnel_url"] = self._url
            return meta

        @stable.get("/")
        def dashboard():
            return RedirectResponse(f"{self._url}/")

        return stable

    @modal.method()
    def run(self) -> None:
        self.process.wait()

    @modal.exit()
    def teardown(self) -> None:
        """Clean up subprocesses on container exit."""
        self._tunnel_ctx.__exit__(None, None, None)
        self.process.terminate()


@app.local_entrypoint()
def main():
    ModalPolicyServer().run.remote()
