from collections import deque

from erlyx.callbacks import BaseCallback
from dataset import SimpleAlphaZeroDataset


class InfoRecorder(BaseCallback):
    def __init__(self, referee, dataset: SimpleAlphaZeroDataset):
        self._dataset = dataset
        self._referee = referee
    
    def on_episode_begin(self, initial_observation):
        self._episode_record = []
        self._observation = initial_observation
        self._episode_reward = None

    def on_step_end(self, action, observation, reward, done):
        info = {'observation': self._observation}
#         player = (self._referee.turn - 1) % len(self._referee._agent_tuple)
#         info['player'] = player
        info['action'] = action.action
        info.update(action.info)
        self._episode_record.append(info)
        self._episode_reward = reward
        self._observation = observation
            
    def on_episode_end(self):
        reward = self._episode_reward
        for info in self._episode_record[::-1]:
            info['reward'] = reward
            reward = -reward
        self._dataset.push(self._episode_record)


class MonteCarloInit(BaseCallback):
    def __init__(self, agent):
        self._agent = agent
    
    def on_episode_begin(self, initial_observation):
        self._agent.policy.init_mcts()


class WeightUpdater(BaseCallback):
    def __init__(self, learner, dataset, update_interval, init_episodes=0):
        self._learner = learner
        self._dataset = dataset
        self._update_interval = update_interval
        self._episode_counter = -init_episodes

    def on_episode_end(self):
        self._episode_counter += 1
        if (self._episode_counter > 0)  and (self._episode_counter % self._update_interval == 0):
            self._episode_counter = 0
            self._learner.update(self._dataset)

