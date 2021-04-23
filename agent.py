from erlyx.agents import BaseAgent, PolicyAgent
from erlyx.types import ActionData
import numpy as np

class RoundRobinReferee(BaseAgent):
    def __init__(self, agent_tuple):
        self._agent_tuple = tuple(agent_tuple)
        self._turn = 0

    def select_action(self, observation):
        action = self._agent_tuple[self._turn].select_action(observation)
        self._turn = (self._turn + 1) % len(self._agent_tuple)
        return action
    
    @property
    def turn(self):
        return self._turn


class SimpleAlphaZeroAgent(PolicyAgent):
    def select_action(self, observation):
        info = self.policy.get_distribution(observation)
        action = np.random.choice(info['legal_moves'], p=info['pi'])
        return ActionData(action=action, info=info)
