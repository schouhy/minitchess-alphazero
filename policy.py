from erlyx.policies import Policy
import chess
import numpy as np
import torch

NUM_ACTIONS = 554


class ConvBlock(torch.nn.Module):
    def __init__(self,
                 nin,
                 nout,
                 kernel_size,
                 stride,
                 padding,
                 batchnorm=True,
                 nonlinearity=True):
        super(ConvBlock, self).__init__()
        layers = []
        layers.append(
            torch.nn.Conv2d(in_channels=nin,
                            out_channels=nout,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding))
        layers.append(torch.nn.BatchNorm2d(nout))
        if nonlinearity:
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(torch.nn.Module):
    def __init__(self, nin, nhid, nout):
        super(ResidualBlock, self).__init__()
        self.convblock1 = ConvBlock(nin, nhid, 3, 1, 1)
        self.convblock2 = ConvBlock(nhid, nout, 3, 1, 1, nonlinearity=False)
        self.nonl = torch.nn.ReLU()

    def forward(self, x):
        x = self.convblock2(self.convblock1(x)) + x
        return self.nonl(x)


class Network(torch.nn.Module):
    def __init__(self, num_actions):
        super(Network, self).__init__()
        self.num_actions = num_actions
        layers = []
        layers.append(ConvBlock(12, 256, 3, 1, 1))
        for _ in range(3):
            layers.append(ResidualBlock(256, 256, 256))
        layers.append(ConvBlock(256, 256, 3, 1, 0))
        self.convblock = torch.nn.Sequential(*layers)
        self.linear = torch.nn.Sequential(torch.nn.Linear(256, 256),
                                          torch.nn.ReLU())
        self.p = torch.nn.Linear(256 * 4 * 3, self.num_actions)
        self.value = torch.nn.Linear(256 * 4 * 3, 1)

    @staticmethod
    def _from_numpy(x):
        return torch.FloatTensor(x)[None]

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = self._from_numpy(x)
        x = self.convblock(x)
        x = x.view(-1, 4, 3, 256)
        x = self.linear(x)
        x = x.view(-1, 256 * 4 * 3)
        dist = self.p(x)
        value = self.value(x)
        return dist, value


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


class SimpleAlphaZeroPolicy(Policy):
    def __init__(self, environment, num_simulations=200, cpuct=1):
        self._num_simulations = num_simulations
        self._environment = environment
        self._network = Network(num_actions=NUM_ACTIONS)
        self._cpuct = cpuct
        self.init_mcts()

    def init_mcts(self):
        self.mcts = MonteCarloTreeSearch(self._environment, self._network,
                                         self._cpuct)

    @property
    def model(self):
        return self._network

    def get_distribution(self, observation):
        with torch.no_grad():
            self._network.eval().cpu()
            for _ in range(self._num_simulations):
                episode, _ = self._environment.new_episode(fen=observation)
                self.mcts.search(episode)
            legal_moves = self.mcts['legal_moves'][observation]
            N = self.mcts['N'][observation]
            dist = N / N.sum()
            return {'legal_moves': legal_moves, 'pi': dist}

    def num_actions(self):
        return NUM_ACTIONS
