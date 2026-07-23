from armory_client.schemas import Action, Observation
from evaluation.agents.base import Agent, AgentEpisodeData
from evaluation.envs.base import Environment
from evaluation.runtime import Runtime


class _Environment(Environment):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reset(self) -> None:
        pass

    def is_episode_complete(self) -> bool:
        return True

    def get_observation(self) -> Observation:
        raise AssertionError("not used")

    def apply_action(self, action: Action) -> None:
        raise AssertionError("not used")

    def close(self) -> None:
        self.events.append("environment")


class _Agent(Agent):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def get_action(self, observation: Observation) -> Action:
        raise AssertionError("not used")

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.events.append("agent")

    def snapshot_episode_data(self) -> AgentEpisodeData:
        return AgentEpisodeData()


def test_close_releases_environment_then_agent() -> None:
    events: list[str] = []
    runtime = Runtime(_Environment(events), _Agent(events))

    runtime.close()

    assert events == ["environment", "agent"]
