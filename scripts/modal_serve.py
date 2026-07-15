import logging
import subprocess
import threading
import time

import modal
import modal.experimental
import requests
from scripts.modal._images import gpu_server_image

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

checkpoint_volume = modal.Volume.from_name("openpi-checkpoints", create_if_missing=True)
CHECKPOINT_VOLUME_PATH = "/checkpoints"

image = gpu_server_image


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
