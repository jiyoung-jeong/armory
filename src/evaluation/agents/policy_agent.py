import dataclasses
import json
import logging
import pathlib
import threading
import time
from collections.abc import Callable

from typing_extensions import override

from armory_client.action_chunk_broker import ActionChunkBroker
from armory_client.client import BidirectionalWebsocket
from armory_client.schemas import Action, ActionChunk, Observation
from evaluation.agents.base import Agent

logger = logging.getLogger(__name__)


# TODO manual: create_null_action needs more thought, it's weird to have the function signature like this
class PolicyAgent(Agent):
    def __init__(
        self,
        ws_client: BidirectionalWebsocket,
        broker: ActionChunkBroker,
        create_null_action: Callable[[Observation, ActionChunk | None], Action],
        event_log_path: pathlib.Path | None = None,
    ) -> None:
        self._ws_client = ws_client
        self._broker = broker
        self._create_null_action = create_null_action
        self._lock = threading.Lock()
        self._closed = False
        self._event_log = event_log_path.open("w") if event_log_path is not None else None
        self.reset()
        self._background_thread = threading.Thread(target=self._receive_actions, daemon=True)
        self._background_thread.start()

    @property
    def episode_id(self) -> str:
        return self._ws_client.episode_id

    def _event(self, kind: str, **fields) -> None:
        if self._event_log is not None:
            self._event_log.write(json.dumps(dict(kind=kind, **fields)) + "\n")
            self._event_log.flush()

    @property
    def broker(self) -> ActionChunkBroker:
        return self._broker

    @override
    def get_action(self, observation: Observation) -> Action:
        with self._lock:
            step_start = time.time()
            queue_before = self._broker.num_actions_available
            action = self._broker.get_action(observation.step)
            if action is None:
                action = dataclasses.replace(
                    self._create_null_action(observation, self._broker.current_action_chunk),
                    actions_left=0,
                )
            timings = self._ws_client.send(
                observation,
                self._broker.next_action_step,
                self._broker.num_actions_available,
                min_execution_horizon=self._broker.min_execution_horizon,
                max_execution_horizon=self._broker.max_execution_horizon,
            )
            if self._event_log is not None:
                self._event(
                    "step",
                    time=step_start,
                    episode_id=self.episode_id,
                    observation_step=observation.step,
                    local_chunk_index=action.action_chunk_index,
                    action_index=action.index_in_chunk,
                    queue_before=queue_before,
                    queue_after=self._broker.num_actions_available,
                    **(timings or {}),
                )
            return action

    @override
    def reset(self) -> None:
        with self._lock:
            self._broker.reset()
            self._ws_client.reset()
            if self._event_log is not None:
                self._event("reset", time=time.time(), episode_id=self.episode_id)

    @override
    def close(self) -> None:
        self._closed = True
        self._ws_client.close()
        self._background_thread.join(timeout=5)
        if self._event_log is not None:
            self._event_log.close()

    @property
    @override
    def action_chunks(self) -> tuple[ActionChunk, ...]:
        with self._lock:
            return tuple(self._broker.action_chunks)

    def _receive_actions(self) -> None:
        while not self._closed:
            try:
                infer_response = self._ws_client.receive()
                receive_end = time.time()
            except Exception:
                if self._closed:
                    return
                raise
            if self._closed:
                return
            with self._lock:
                # Check while holding the same lock as reset: recv may finish
                # before reset while this thread is waiting to apply its response.
                if infer_response.episode_id != self._ws_client.episode_id:
                    self._event(
                        "response_discarded",
                        time=time.time(),
                        episode_id=infer_response.episode_id,
                        current_episode_id=self.episode_id,
                        request_id=infer_response.request_id,
                        chunk_id=infer_response.chunk_id,
                        reason="stale_episode",
                    )
                    logger.info(
                        "Discarding stale response request=%s episode=%s current=%s",
                        infer_response.request_id,
                        infer_response.episode_id,
                        self._ws_client.episode_id,
                    )
                    continue
                queue_before = self._broker.num_actions_available
                next_action = self._broker.next_action_step
                action_chunk = self._broker.receive_response(infer_response)
                self._event(
                    "chunk_received",
                    time=action_chunk.response_timestamp,
                    receive_end=receive_end,
                    episode_id=self.episode_id,
                    request_id=action_chunk.request_id,
                    chunk_id=action_chunk.chunk_id,
                    execution_start_step=action_chunk.execution_start_step,
                    queue_before=queue_before,
                    queue_after=self._broker.num_actions_available,
                    next_action_index=next_action,
                    action_index_start=action_chunk.action_index_start,
                    max_execution_horizon=action_chunk.max_execution_horizon,
                )
                first_executed_index = max(
                    0, self._broker.next_action_step - action_chunk.action_index_start
                )
                self._ws_client.send_ack(
                    request_id=action_chunk.request_id,
                    chunk_id=action_chunk.chunk_id,
                    observation_step=action_chunk.observation_step,
                    receive_time=action_chunk.response_timestamp,
                    action_index_start=action_chunk.action_index_start,
                    min_execution_horizon=action_chunk.min_execution_horizon,
                    max_execution_horizon=action_chunk.max_execution_horizon,
                    execution_start_step=action_chunk.execution_start_step,
                    first_executed_index=first_executed_index,
                )
