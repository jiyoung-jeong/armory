"""Throwaway script to test BidirectionalWebsocket in isolation.

Usage:
    uv run scripts/serve.py policy:mock --policy.action-horizon 50
    uv run scripts/test_ws_client.py
"""

import logging
import time

import numpy as np
import tyro
from dataclasses import dataclass

from armory_client.client import BidirectionalWebsocket
from armory_client.messages import InferType
from armory_client.schemas import Observation

import logging
import websockets

# This will log the raw bytes of every frame
logger = logging.getLogger('websockets')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8080
    robot_id: str = "test_robot_0"
    control_hz: float = 20.0
    num_infer: int = 3


def main(args: Args) -> None:
    logger.info("Connecting to %s:%d as %s", args.host, args.port, args.robot_id)

    ws = BidirectionalWebsocket(
        robot_id=args.robot_id,
        host=args.host,
        port=args.port,
        control_hz=args.control_hz,
    )
    logger.info("Handshake + warmup complete")

    img = np.zeros((224, 224, 3), dtype=np.uint8)
    obs = Observation(
        step=0,
        state=np.zeros(8, dtype=np.float32),
        image=img,
        wrist_image=img,
    )

    for i in range(args.num_infer):
        deadline = time.time() + 1.0
        ws.send(
            obs=obs,
            deadline=deadline,
            action_start_step=i,
            infer_type=InferType.SYNC,
        )
        logger.info("Sent infer request %d", i)

        response = ws.receive()
        logger.info(
            "Got response %d: request_id=%d actions.shape=%s",
            i,
            response.request_id,
            response.actions.shape,
        )

        ws.send_ack(
            request_id=response.request_id,
            receive_time=time.time(),
            execution_start_step=i,
        )

        obs = Observation(
            step=i + 1,
            state=np.zeros(8, dtype=np.float32),
            image=img,
            wrist_image=img,
        )

    logger.info("All %d infer round-trips succeeded", args.num_infer)
    ws.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
