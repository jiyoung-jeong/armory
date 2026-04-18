import numpy as np
from openpi_client.schemas import Action
from openpi_client.action_chunkers.action_chunk_broker import ActionChunkBroker
from openpi_client.client import BidirectionalWebsocket
from typing_extensions import override
from typing import List


class TemporalEnsemblingBroker(ActionChunkBroker):
    """Temporal ensembling broker using exponential weighting of overlapping predictions.

    Uses formula: w_i = exp(-m * i) where i is chunk age (0=newest, 1=second newest, etc.)
    Smaller m = slower incorporation of new observations (keeps more history)
    Larger m = faster incorporation (weights recent predictions more heavily)
    """

    def __init__(self, ws_client: BidirectionalWebsocket, control_hz: int, realtime: bool = True, m_param: float = 1.0):
        """
        Args:
            ws_client: websocket client for inference
            control_hz: control frequency of the environment
            realtime: whether to run in realtime mode
            m_param: exponential decay rate (smaller = slower incorporation of new observations)
        """
        super().__init__(ws_client=ws_client, control_hz=control_hz, realtime=realtime)
        self._m_param = m_param

    @override
    def _update_action_queue(self, action_chunk):
        """Update action queue with temporal ensembling using exponential weighting."""
        with self._lock:
            existing_actions = list(self._action_queue)

            # Clear queue and rebuild with ensembled actions
            self._action_queue.clear()

            # Determine range of steps to generate
            if existing_actions:
                start_step = existing_actions[0].step
                end_step = max(
                    existing_actions[-1].step, action_chunk.action_start_step + action_chunk.execution_horizon - 1
                )
            else:
                start_step = action_chunk.action_start_step
                end_step = action_chunk.action_start_step + action_chunk.execution_horizon - 1

            # For each step, collect all predictions and ensemble them
            for step in range(start_step, end_step + 1):
                predictions: List[tuple[np.ndarray, int]] = []  # (action, chunk_age)

                # Collect predictions from all chunks that cover this step
                for chunk_idx, chunk in enumerate(self._action_chunks):
                    chunk_start = chunk.action_start_step
                    chunk_end = chunk.action_start_step + len(chunk.actions) - 1

                    if chunk_start <= step <= chunk_end:
                        action_idx = step - chunk_start
                        chunk_age = len(self._action_chunks) - 1 - chunk_idx  # 0=newest
                        predictions.append((chunk.actions[action_idx], chunk_age))

                if predictions:
                    # Apply exponential weighting: w_i = exp(-m * i)
                    weights = np.array([np.exp(-self._m_param * age) for _, age in predictions])
                    weights = weights / weights.sum()  # Normalize

                    # Weighted average
                    ensembled_action = sum(w * pred for (pred, _), w in zip(predictions, weights))

                    self._action_queue.append(
                        Action(
                            step=step,
                            action=ensembled_action,
                            action_chunk_index=len(self._action_chunks) - 1,
                            index_in_chunk=step - action_chunk.action_start_step
                            if step >= action_chunk.action_start_step
                            else None,
                        )
                    )
