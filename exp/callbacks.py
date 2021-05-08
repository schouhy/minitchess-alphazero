from collections import deque

from erlyx.callbacks import BaseCallback
from exp.dataset import SimpleAlphaZeroDataset
from exp.agent import RoundRobinReferee

class WinnerRecorder(BaseCallback):
    def __init__(self, referee: RoundRobinReferee):
        self._referee = referee
        self._results = []

    def on_episode_end(self):
        self._results.append(self._referee.get_turn())

    def get_results(self):
        return sum(self._results) / len(self._results) 


class InfoRecorder(BaseCallback):
    def __init__(self, dataset):
        self._dataset = dataset
    
    def on_episode_begin(self, initial_observation):
        self._episode_record = []
        self._observation = initial_observation
        self._episode_reward = None

    def on_step_end(self, action, observation, reward, done):
        info = {'observation': self._observation}
        info.update(action.info)
        info['action'] = int(action.action)
        info['pi'] = info['pi'].tolist()
        self._episode_record.append(info)
        self._episode_reward = reward
        self._observation = observation
            
    def on_episode_end(self):
        reward = self._episode_reward
        for info in self._episode_record[::-1]:
            info['reward'] = reward
            reward = -reward
        return self._dataset.push(self._episode_record)


class MonteCarloInit(BaseCallback):
    def __init__(self, agent):
        self._agent = agent
    
    def on_episode_begin(self, initial_observation):
        self._agent.init_mcts()


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

