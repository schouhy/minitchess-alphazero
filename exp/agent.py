from erlyx.agents import BaseAgent, PolicyAgent
from erlyx.types import ActionData
import numpy as np


class RoundRobinReferee(BaseAgent):
    def __init__(self, agent_tuple):
        self._agent_tuple = tuple(agent_tuple)
        self._turn = False

    def select_action(self, observation):
        action = self._agent_tuple[int(self._turn)].select_action(observation)
        self._turn = not self._turn
        return action

    def reset(self):
        self._turn = False

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
            'visited': set(),
            'legal_moves': {},
        }

    def __getitem__(self, item):
        return self._data.get(item, None)

    def simulate(self, num_simulations, observation):
        for _ in range(num_simulations):
            episode, _ = self._environment.new_episode(fen=observation)
            self._search(episode)
        return self._data

    def _backprop(self, value, chain):
        for node, action in chain[::-1]:
            value = -value
            Q, N = self['Q'][node], self['N'][node]
            Q[action] = (N[action] * Q[action] + value) / (N[action] + 1)
            N[action] += 1

    def _search(self, episode, chain=None):
        chain = chain or []
        node = episode.get_observation()
        if node not in self['visited']:
            self['visited'].add(node)
            if episode.is_done():
                value = -episode.get_reward()
                self['terminal'][node] = value
                self._backprop(value, chain)
                return
            legal_moves = episode.get_legal_moves()
            self['Q'][node] = np.zeros(len(legal_moves))
            self['N'][node] = np.zeros(len(legal_moves))
            p, v = self._model(self._model.process_observation(node))
            p = p[0][legal_moves].softmax(0).data.numpy()
            v = v.item()
            self['P'][node] = p
            self['legal_moves'][node] = legal_moves
            self._backprop(v, chain)
            return

        if node in self['terminal'].keys():
            self._backprop(-self['terminal'][node], chain)
            return

        Q, N, P = self['Q'][node], self['N'][node], self['P'][node]
        legal_moves = self['legal_moves'][node]
        if len(chain) == 0:  # is root node
            P = 0.75 * P + 0.25 * np.random.dirichlet([0.6] * len(legal_moves))
        legal_moves = self['legal_moves'][node]
        u = Q + self._cpuct * P * np.sqrt(N.sum()) / (1 + N)
        action = u.argmax()
        episode.step(legal_moves[action], return_status=False)
        chain.append((node, action))
        self._search(episode, chain)


class SimpleAlphaZeroAgent(PolicyAgent):
    def __init__(self,
                 environment,
                 policy,
                 num_simulations,
                 cpuct=1,
                 tau_change=6):
        super(SimpleAlphaZeroAgent, self).__init__(policy)
        self._environment = environment
        self._num_simulations = num_simulations
        self._cpuct = cpuct
        self._tau_change = tau_change
        self.init_mcts()

    def init_mcts(self):
        self._mcts = MonteCarloTreeSearch(self._environment, self.policy.model,
                                          self._cpuct)
        self._count = 0

    def select_action(self, observation):
        info = self.policy.get_distribution(observation, self._mcts,
                                            self._num_simulations)
        num_moves = int(observation.split()[3])
        if num_moves < self._tau_change:
            action = np.random.choice(info['legal_moves'], p=info['pi'])
        else:
            maxima = np.where(info['pi'] == info['pi'].max())[0]
            action = info['legal_moves'][np.random.choice(maxima)]
        return ActionData(action=action, info=info)
