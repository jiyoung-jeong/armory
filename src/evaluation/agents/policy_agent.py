import threading
from collections.abc import Callable

from typing_extensions import override

from armory_client.action_chunk_broker import ActionChunkBroker
from armory_client.client import BidirectionalWebsocket
from armory_client.schemas import Action, ActionChunk, Observation
from evaluation.agents.base import Agent, AgentEpisodeData


# TODO: create_null_action needs more thought
class PolicyAgent(Agent):
    def __init__(
        self,
        ws_client: BidirectionalWebsocket,
        broker: ActionChunkBroker,
        create_null_action: Callable[[Observation, ActionChunk | None], Action],
    ) -> None:
        self._ws_client = ws_client
        self._broker = broker
        self._create_null_action = create_null_action
        self._lock = threading.Lock()
        self._closed = False
        self.reset()
        self._background_thread = threading.Thread(target=self._receive_actions, daemon=True)
        self._background_thread.start()

    @property
    def broker(self) -> ActionChunkBroker:
        return self._broker

    @override
    def get_action(self, observation: Observation) -> Action:
        with self._lock:
            action = self._broker.get_action(observation.step)
            if action is None:
                action = self._create_null_action(observation, self._broker.current_action_chunk)
            self._ws_client.send(
                observation,
                self._broker.next_action_step,
                self._broker.num_actions_available,
                min_execution_horizon=self._broker.min_execution_horizon,
                max_execution_horizon=self._broker.max_execution_horizon,
            )
            return action

    @override
    def reset(self) -> None:
        with self._lock:
            self._broker.reset()
            self._ws_client.reset()

    @override
    def close(self) -> None:
        self._closed = True
        self._ws_client.close()
        self._background_thread.join(timeout=5)

    # TODO: shouldn't need a crazy dataclass or special method for this?
    @override
    def snapshot_episode_data(self) -> AgentEpisodeData:
        with self._lock:
            return AgentEpisodeData(
                action_chunks=tuple(self._broker.action_chunks),
                actions_left=tuple(self._broker.actions_left_history),
            )

    def _receive_actions(self) -> None:
        while not self._closed:
            try:
                infer_response = self._ws_client.receive()
            except Exception:
                if self._closed:
                    return
                raise
            if self._closed:
                return
            with self._lock:
                action_chunk = self._broker.receive_response(infer_response)
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
