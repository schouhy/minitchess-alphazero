from erlyx.policies import Policy
import chess
import numpy as np
import torch

NUM_ACTIONS = 554
CODES = {v: k for k, v in enumerate('0prbnqk')}
REP = {'/': '', **{str(i): ''.join(['0'] * i) for i in range(1, 6)}}
EMBEDDING_DIM = 4

# TODO: Traer esto directamente de python-chess
MAX_NUM_MOVES_ALLOWED = 30


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
        self.emb = torch.nn.Embedding(7, EMBEDDING_DIM)
        layers = []
        layers.append(ConvBlock(EMBEDDING_DIM * 2, 256, 3, 1, 1))
        for _ in range(5):
            layers.append(ResidualBlock(256, 256, 256))
        self.resbody = torch.nn.Sequential(*layers)
        self.pconv = ConvBlock(256, 2, 1, 1, 0)
        self.plinear = torch.nn.Linear(2 * 6 * 5 + 1, num_actions)

        self.vconv = ConvBlock(256, 1, 1, 1, 0)
        self.vlinear = torch.nn.Sequential(torch.nn.Linear(6 * 5 + 1, 256),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(256, 1),
                                           torch.nn.Tanh())

    def forward(self, input_data):
        channels, clock = input_data
        channels = self.emb(channels).permute(0, 1, 4, 2, 3)
        channels = channels.contiguous().view(-1, EMBEDDING_DIM * 2, 6, 5)
        channels = self.resbody(channels)
        px = self.pconv(channels).view(-1, 2 * 6 * 5)
        p = self.plinear(torch.cat([px, clock], dim=1))
        vx = self.vconv(channels).view(-1, 6 * 5)
        v = self.vlinear(torch.cat([vx, clock], dim=1))
        return p, v

    @classmethod
    def _player_side_view(cls, bfen, color):
        assert color in ['w', 'b']
        return bfen if color == 'w' else bfen[::-1].swapcase()

    @classmethod
    def tokenize(cls, bfen, color):
        bfen = cls._player_side_view(bfen, color)
        for a, b in REP.items():
            bfen = bfen.replace(a, b)
        white = [CODES.get(o.lower() if o.isupper() else '0') for o in bfen]
        black = [CODES.get(o if o.islower() else '0') for o in bfen]
        return white + black

    @classmethod
    def process_observation(cls, observation):
        bfen, color, halfmove_clock, fullmove_clock = observation.split()
        tokens = cls.tokenize(bfen, color)
        channels = torch.LongTensor(tokens).reshape(1, 2, 6, 5)
        clock = float(fullmove_clock)
        if color == 'b':
            clock += 0.5
        clock = torch.tensor([[clock / MAX_NUM_MOVES_ALLOWED]]).float()
        return channels, clock

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
