import logging

logging.basicConfig(filename='log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

from erlyx.learners import BaseLearner
from erlyx import run_episodes
from torch.utils.data import DataLoader
from exp.environment import MinitChessEpisode, NUM_ACTIONS
from exp.dataset import SimpleAlphaZeroDataset
from exp.agent import RoundRobinReferee, SimpleAlphaZeroAgent
from exp.callbacks import WinnerRecorder, MonteCarloInit
from exp.policy import Network, SimpleAlphaZeroPolicy
import numpy as np
import torch
import copy

ARENA_GAME_NUMBER_PER_SIDE = 3

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


class AvgSmoothLoss:
    def __init__(self, beta=0.98): 
        self.beta = beta
        
    def reset(self):   
        self.count = 0
        self.val = 0.
        return self

    def accumulate(self, new_val):
        self.count += 1
        self.val = new_val + self.beta*(self.val - new_val)

    @property
    def value(self): return self.val/(1-self.beta**self.count)


class SimpleAlphaZeroLearner(BaseLearner):
    def __init__(self, env, num_simulations, network, batch_size, learning_rate, epochs):
        self._env = env
        self._num_simulations = num_simulations
        self._network = network
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._epochs = epochs

    def update(self, dataset: SimpleAlphaZeroDataset):
        optimizer = torch.optim.AdamW(self._network.parameters(),
                                      lr=self._learning_rate)
        dataloader = DataLoader(dataset,
                                batch_size=self._batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
        model = self._network.train().cuda()
        old_state_dict = copy.deepcopy(model.state_dict())
        metric = AvgSmoothLoss().reset()

        # Update params
        for epoch in range(self._epochs):
            for pib, boardb, reward in iter(dataloader):
                pib, boardb, reward = pib.cuda(), boardb.cuda(), reward.cuda()
                pb, vb = model(boardb)
                pb = pb.log_softmax(-1)
                loss = ((vb - reward)**2 - (pib * pb).sum(1)).mean()
                optimizer.zero_grad()
                loss.backward()
                metric.accumulate(loss.detach().data.cpu().numpy().item())
                optimizer.step()
            logging.info(f'Epoch {epoch}: {metric.value:.2f}')

        # Compete against older version
        old_network = Network()
        old_network.load_state_dict(old_state_dict)
        old_policy = SimpleAlphaZeroPolicy(old_network)
        old_agent = SimpleAlphaZeroAgent(self._env, old_policy, self._num_simulations)
        
        new_policy = SimpleAlphaZeroPolicy(model)
        new_agent = SimpleAlphaZeroAgent(self._env, new_policy, self._num_simulations)
        new_agent_wins = 0
        old_agent_wins = 0
        with torch.no_grad():
            model.eval()
            # New agent plays white
            referee = RoundRobinReferee((new_agent, old_agent))
            winner_recorder = WinnerRecorder(referee)
            run_episodes(self._env, referee, n_episodes=ARENA_GAME_NUMBER_PER_SIDE, callbacks=[winner_recorder,MonteCarloInit(old_agent), MonteCarloInit(new_agent)])
            new_agent_wins += winner_recorder.results[False]
            old_agent_wins += winner_recorder.results[True]
            logging.info(f'New agent playing with white results: {winner_recorder.results}')
            # New agent plays black
            referee = RoundRobinReferee((old_agent, new_agent))
            winner_recorder = WinnerRecorder(referee)
            run_episodes(self._env, referee, n_episodes=ARENA_GAME_NUMBER_PER_SIDE, callbacks=[winner_recorder,MonteCarloInit(old_agent), MonteCarloInit(new_agent)])
            logging.info(f'New agent playing with black results: {winner_recorder.results}')
            new_agent_wins += winner_recorder.results[True]
            old_agent_wins += winner_recorder.results[False]
        return new_agent_wins/(new_agent_wins + old_agent_wins + 1e-8)


        

