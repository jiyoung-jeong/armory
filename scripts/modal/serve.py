import json
import logging
import pathlib
import subprocess
import threading
import time

import modal
import modal.experimental
import requests
from scripts.modal.images import REMOTE_ROOT, gpu_server_image

log = logging.getLogger(__name__)

app = modal.App("armory-serve")

GPU = "l40s"
REGION = "us"
ENV_MODE = "LIBERO"
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

checkpoint_volume = modal.Volume.from_name("openpi-checkpoints", create_if_missing=True)
CHECKPOINT_VOLUME_PATH = "/checkpoints"

image = gpu_server_image


@app.cls(
    gpu=GPU,
    image=image,
    volumes={CHECKPOINT_VOLUME_PATH: checkpoint_volume},
    region=REGION,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=60 * 60,  # seconds, time to wait before scaling down
    timeout=2 * 60 * 60,  # 2 hours
)
# run() blocks in self.process.wait() for the container's whole lifetime; without
# concurrent inputs, the container's single input slot stays occupied and the
# stable_endpoint ASGI route (same container/class instance) can never be served,
# so external /metadata requests hang forever.
@modal.concurrent(max_inputs=4)
class ModalPolicyServer:
    @modal.enter(snap=True)
    def startup(self) -> None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info("Starting server")

        # serve.py's Args nests scheduler options under a SchedulerConfig submodel and
        # its policy field is a Checkpoint | Default | Mock union, both awkward to hit
        # via plain CLI flags with tyro. --json-path (see scripts.utils.JsonArgs)
        # takes a plain JSON dict validated directly by pydantic, so enum fields need
        # their value (lowercase), not their member name.
        args_path = pathlib.Path("/tmp/serve_args.json")
        args_path.write_text(
            json.dumps(
                {
                    "model": MODEL.lower(),
                    "env": ENV_MODE.lower(),
                    "max_batch_size": MAX_BATCH_SIZE,
                    "port": PORT,
                    "scheduler": {
                        "scheduling_algorithm": SCHEDULING_ALGORITHM,
                        "alpha": ALPHA,
                        "action_horizon_multipliers": {
                            str(k): v for k, v in ACTION_HORIZON_MULTIPLIERS.items()
                        },
                    },
                }
            )
        )
        cmd = [
            "python",
            str(REMOTE_ROOT / "scripts/serve.py"),
            "--json-path",
            str(args_path),
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
            if self.process.poll() is not None:
                raise RuntimeError(f"serve.py exited early with code {self.process.returncode}")
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
            return requests.get(f"http://localhost:{PORT}/metadata").json()

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
