import abc

from armory_client.schemas import Action, Observation


class Agent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, observation: Observation) -> Action:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass
