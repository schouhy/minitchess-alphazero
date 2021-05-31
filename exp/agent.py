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
            'visited': [],
            'legal_moves': {},
        }

    def __getitem__(self, item):
        return self._data.get(item, None)

    def simulate(self, num_simulations, observation):
        for _ in range(num_simulations):
            episode, _ = self._environment.new_episode(fen=observation)
            self._search(episode, is_root_node=True)
        return self._data

    def _search(self, episode, is_root_node=False):
        node = episode.get_observation()
        if node not in self['visited']:
            self['visited'].append(node)
            if episode.is_done(): 
                self['terminal'][node]= -episode.get_reward()
                return -self['terminal'][node]
            legal_moves = episode.get_legal_moves()
            self['Q'][node] = np.zeros(len(legal_moves))
            self['N'][node] = np.zeros(len(legal_moves))
            p, v = self._model(self._model.process_observation(node))
            p = p[0][legal_moves].softmax(0).data.cpu().numpy()
            if is_root_node:
                p = p*0.75 + 0.25*np.random.dirichlet([0.6]*len(legal_moves))
            self['P'][node] = p 
            self['legal_moves'][node] = legal_moves
            return -v[0]

        if node in self['terminal'].keys():
            return -self['terminal'][node]

        Q, N, P = self['Q'][node], self['N'][node], self['P'][node]
        legal_moves = self['legal_moves'][node]
        if is_root_node:
            P = P*0.75 + 0.25*np.random.dirichlet([0.6]*len(legal_moves))

        u = Q + self._cpuct * P * np.sqrt(N.sum()) / (1 + N)
        action = u.argmax()
        episode.step(legal_moves[action], return_status=False)
        v = self._search(episode)

        Q[action] = (N[action] * Q[action] + v) / (N[action] + 1)
        N[action] += 1
        return -v


class SimpleAlphaZeroAgent(PolicyAgent):
    def __init__(self, environment, policy, num_simulations, cpuct=1, tau_change=6):
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
        info = self.policy.get_distribution(observation,
                                            self._mcts,
                                            self._num_simulations)
        num_moves = int(observation.split()[3])
        if num_moves < self._tau_change:
            action = np.random.choice(info['legal_moves'], p=info['pi'])
        else:
            maxima = np.where(info['pi'] == info['pi'].max())[0]
            action = info['legal_moves'][np.random.choice(maxima)]
        return ActionData(action=action, info=info)
