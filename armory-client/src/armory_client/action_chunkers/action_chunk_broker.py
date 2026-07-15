import threading
import time
from collections import deque

from armory_client.client import BidirectionalWebsocket
from armory_client.messages import InferResponse
from armory_client.schemas import Action, ActionChunk, Observation


# NOTE: use concurrent.futures to infer in background if this takes too long
class ActionChunkBrokerBase:
    def __init__(
        self,
        real: bool = False,
        min_execution_horizon: int = 0,
    ) -> None:
        self._real = real
        self.min_execution_horizon = min_execution_horizon
        self._action_queue: deque[Action] = deque()
        self._action_chunks: list[ActionChunk] = []
        self._next_observation_step: int = 0  # next observation step to see
        self._next_action_step: int = 0  # next action step to execute
        self._prev_action: Action = self._create_null_action(-1, None)
        self._actions_left_history: list[int] = []

    def get_action(self, observation_step: int, obs: Observation | None = None) -> Action:
        self._actions_left_history.append(len(self._action_queue))
        self._next_observation_step = observation_step + 1
        if self._action_queue:
            action = self._action_queue.popleft()
            self._next_action_step += 1
        else:
            action = self._create_null_action(observation_step, obs)

        self._prev_action = action
        return action

    def receive_response(self, infer_response: InferResponse) -> ActionChunk:
        action_chunk = ActionChunk.from_infer_response(
            infer_response=infer_response,
            execution_start_step=self._next_observation_step,
        )

        self._action_chunks.append(action_chunk)
        self._update_action_queue(action_chunk)
        return action_chunk

    def _update_action_queue(self, action_chunk: ActionChunk) -> None:
        while self._action_queue and self._action_queue[-1].step >= action_chunk.action_index_start:
            self._action_queue.pop()

        # assumes that pausing is preferable to executing actions past the execution horizon
        self._action_queue.extend(
            Action(
                step=action_chunk.action_index_start + i,
                action=action_chunk.get_action(i),
                action_chunk_index=len(self._action_chunks) - 1,
                index_in_chunk=i,
            )
            for i in range(action_chunk.max_execution_horizon)
            if action_chunk.action_index_start + i >= self._next_action_step
        )

    def _create_null_action(self, observation_step: int, obs: Observation | None) -> Action:
        # FIXME: hardcoded, should move this outside of this class
        import numpy as np

        if self._real and obs is not None and obs.state is not None:
            action = obs.state.copy()
            step = observation_step
        else:
            action = np.zeros(7)
            action[-1] = (
                self.current_action_chunk.get_action(-1)[-1]
                if self.current_action_chunk is not None
                else 0.0
            )
            step = None

        return Action(
            step=step,
            action=action,
            action_chunk_index=None,
            index_in_chunk=None,
        )

    @property
    def action_chunks(self) -> list[ActionChunk]:
        return self._action_chunks

    @property
    def current_action_chunk(self) -> ActionChunk | None:
        return self._action_chunks[-1] if self._action_chunks else None

    @property
    def actions_left_history(self) -> list[int]:
        """Actions remaining in queue after each step (recorded inside the lock)."""
        return list(self._actions_left_history)


class ActionChunkBroker(ActionChunkBrokerBase):
    def __init__(
        self,
        ws_client: BidirectionalWebsocket,
        control_hz: int,
        realtime: bool = True,
        min_execution_horizon: int = 0,
        max_execution_horizon: int = 0,
        real: bool = False,
    ) -> None:
        super().__init__(real=real, min_execution_horizon=min_execution_horizon)

        self._step_duration = 1 / control_hz
        self._realtime = realtime
        self.max_execution_horizon = max_execution_horizon

        self._ws_client = ws_client
        self._lock = threading.Lock()
        self._background_thread = threading.Thread(target=self._receive_actions, daemon=True)

        self.reset()
        self._background_thread.start()

    def infer(self, obs: Observation) -> Action:
        """Client continuously streams observations to the server."""
        with self._lock:
            # count actions left in queue before we pop the next action
            action = self.get_action(obs.step, obs)

            self._infer(obs)

            return action

    def _infer(self, obs: Observation) -> None:
        self._ws_client.send(
            obs,
            self.deadline,
            self._next_action_step,
            min_execution_horizon=self.min_execution_horizon,
            max_execution_horizon=self.max_execution_horizon,
        )

    def _receive_actions(self) -> None:
        while True:
            infer_response = self._ws_client.receive()
            with self._lock:
                action_chunk = self.receive_response(infer_response)

                # TODO: should be tested too
                first_executed_index = max(
                    0, self._next_action_step - action_chunk.action_index_start
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

    def reset(self) -> None:
        with self._lock:
            self._next_observation_step = 0
            self._next_action_step = 0
            self._action_queue.clear()
            self._action_chunks = []
            self._actions_left_history = []
            self._received_first_chunk = False
            self._ws_client.reset()

    @property
    def deadline(self) -> float:
        return time.time() + len(self._action_queue) * self._step_duration
