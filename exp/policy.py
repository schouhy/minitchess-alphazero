from erlyx.policies import Policy
import chess
import numpy as np
import torch

NUM_ACTIONS = 554

class Network(nn.Module):
    def __init__(self, num_actions=NUM_ACTIONS):
        # game params
        self.board_x, self.board_y = 5, 6
        self.action_size = num_actions

        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 12, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(12, 12, 3, stride=1)
        self.conv4 = nn.Conv2d(12, 12, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(12)
        self.bn3 = nn.BatchNorm2d(12)
        self.bn4 = nn.BatchNorm2d(12)

        self.fc1 = nn.Linear(12*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.12*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return pi, torch.tanh(v)


class Network(torch.nn.Module):
    def __init__(self, num_actions=NUM_ACTIONS):
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
        self.value = torch.nn.Sequential(torch.nn.Linear(256 * 4 * 3, 1), torch.nn.Tanh())

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
