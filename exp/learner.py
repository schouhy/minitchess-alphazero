from erlyx.learners import BaseLearner
from erlyx import run_episodes
from torch.utils.data import DataLoader
from exp.environment import MinitChessEpisode, NUM_ACTIONS
from exp.dataset import SimpleAlphaZeroDataset
from exp.agent import RoundRobinReferee, SimpleAlphaZeroAgent
from exp.callbacks import WinnerRecorder
from exp.policy import Network, SimpleAlphaZeroPolicy
import numpy as np
import torch
import copy


def collate_fn(batch):
    pis = []
    for item in batch:
        pi = np.zeros(NUM_ACTIONS)
        pi[item['legal_moves']] = item['pi']
        pis.append(pi)
    obs = [
        MinitChessEpisode(o['observation']).get_board_array()[None]
        for o in batch
    ]
    rewards = [o['reward'] for o in batch]
    return map(
        torch.FloatTensor,
        [np.vstack(pis), np.vstack(obs),
         np.vstack(rewards)])


class SimpleAlphaZeroLearner(BaseLearner):
    def __init__(self, env, num_simulations, network, batch_size, learning_rate):
        self._env = env
        self._num_simulations = num_simulations
        self._network = network
        self._batch_size = batch_size

    def update(self, dataset: SimpleAlphaZeroDataset):
        optimizer = torch.optim.AdamW(self._network.parameters(),
                                      lr=learning_rate)
        dataloader = DataLoader(dataset,
                                batch_size=self._batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
        model = self._network.train().cuda()
        old_state_dict = copy.deepcopy(model.state_dict())

        # Update params
        for pib, boardb, reward in iter(dataloader):
            pib, boardb, reward = pib.cuda(), boardb.cuda(), reward.cuda()
            pb, vb = model(boardb)
            pb = pb.log_softmax(-1)
            loss = ((vb - reward)**2 - (pib * pb).sum(1)).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Compete against older version
        old_network = Network().load_state_dict(old_state_dict)
        old_policy = SimpleAlphaZeroPolicy(old_network)
        old_agent = SimpleAlphaZeroAgent(self._env, old_policy, self._num_simulations)
        
        new_policy = SimpleAlphaZeroPolicy(network)
        new_agent = SimpleAlphaZeroAgent(self._env, new_policy, self._num_simulations)
        referee = RoundRobinReferee((new_agent, old_agent))
        winner_recorder = WinnerRecorder(referee)
        with torch.no_grad():
            model.eval()
            run_episodes(self._env, referee, n_episodes=10)
        return winner_recorder.get_results()


        

