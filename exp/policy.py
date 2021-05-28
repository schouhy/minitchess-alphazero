from erlyx.policies import Policy
import chess
import numpy as np
import torch

NUM_ACTIONS = 554
NUM_CHANNELS_BOARD_ARRAY = 14
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
    def __init__(self, num_actions=NUM_ACTIONS):
        super(Network, self).__init__()
        layers = []
        layers.append(ConvBlock(NUM_CHANNELS_BOARD_ARRAY, 256, 3, 1, 1))
        for _ in range(5):
            layers.append(ResidualBlock(256, 256, 256))
        layers.append(ConvBlock(256, 256, 3, 1, 1))
        self.convblock = torch.nn.Sequential(*layers)
        self.linear = torch.nn.Sequential(torch.nn.Linear(256, 256),
                                          torch.nn.ReLU())
        self.p = torch.nn.Sequential(torch.nn.Linear(256 * 6 * 5, 256), torch.nn.ReLU() ,torch.nn.Linear(256, num_actions))
        self.value = torch.nn.Sequential(torch.nn.Linear(256 * 6 * 5, 1), torch.nn.Tanh())

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


class SimpleAlphaZeroPolicy(Policy):
    def __init__(self, network=None):
        self._network = network or Network()

    @property
    def model(self):
        return self._network

    def get_distribution(self, observation, mcts, num_simulations):
        with torch.no_grad():
            self._network.eval().cpu()
            mcts.simulate(num_simulations, observation)
            legal_moves = mcts['legal_moves'][observation]
            N = mcts['N'][observation]
            dist = N / N.sum()
            return {'legal_moves': legal_moves, 'pi': dist}

    def num_actions(self):
        return NUM_ACTIONS
