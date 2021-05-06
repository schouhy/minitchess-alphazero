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


class MonteCarloTreeSearch:
    def __init__(self, environment, model, cpuct):
        self._environment = environment
        self._model = model
        self._cpuct = cpuct
        self._data = {
            'Q': {},
            'N': {},
            'P': {},
            'terminal': {},
            'visited': [],
            'legal_moves': {},
        }

    def __getitem__(self, item):
        return self._data.get(item, None)

    def simulate(self, num_simulations, observation):
        for _ in range(num_simulations):
            episode, _ = self._environment.new_episode(fen=observation)
            mcts.search(episode)
        return self._data

    def search(self, episode):
        node = episode.get_observation()
        if node not in self['visited']:
            self['visited'].append(node)
            self['terminal'][node] = episode.is_done()
            if self['terminal'][node]:
                return -1
            legal_moves = episode.get_legal_moves()
            self['Q'][node] = np.zeros(len(legal_moves))
            self['N'][node] = np.zeros(len(legal_moves))
            p, v = self._model(episode.get_board_array())
            self['P'][node] = p[0][legal_moves].softmax(0).data.cpu().numpy()
            self['legal_moves'][node] = legal_moves
            return -v[0]

        if self['terminal'][node]:
            return -1

        Q, N, P = self['Q'][node], self['N'][node], self['P'][node]
        legal_moves = self['legal_moves'][node]

        u = Q + self._cpuct * P * np.sqrt(N.sum()) / (1 + N)
        action = u.argmax()
        episode.step(legal_moves[action], return_status=False)
        v = self.search(episode)

        Q[action] = (N[action] * Q[action] + v) / (N[action] + 1)
        N[action] += 1
        return -v


class SimpleAlphaZeroAgent(PolicyAgent):
    def __init__(self, environment, policy, num_simulations, cpuct=1):
        super(SimpleAlphaZeroAgent, self).__init__(policy)
        self._environment = environment
        self._num_simulations = num_simulations
        self._cpuct = cpuct
        self.init_mcts()

    def init_mcts(self):
        self._mcts = MonteCarloTreeSearch(self._environment, self.policy.model,
                                          self._cpuct)

    def select_action(self, observation):
        info = self.policy.get_distribution(observation,
                                            self._mcts,
                                            self._num_simulations)
        action = np.random.choice(info['legal_moves'], p=info['pi'])
        return ActionData(action=action, info=info)
