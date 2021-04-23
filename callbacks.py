from collections import deque

from erlyx.callbacks import BaseCallback
from dataset import SimpleAlphaZeroDataset


class InfoRecorder(BaseCallback):
    def __init__(self, dataset: SimpleAlphaZeroDataset):
        self._dataset = dataset
    
    def on_episode_begin(self, initial_observation):
        self._episode_record = []
        self._observation = initial_observation
        self._episode_reward = None

    def on_step_end(self, action, observation, reward, done):
        info = {'observation': self._observation}
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
        self._agent.init_mcts()


class WeightUpdater(BaseCallback):
    _instance_counter = 0
    def __init__(self, learner, dataset, update_interval, init_episodes=0):
        self._instance_counter += 1
        self._name = f'{self}_{self._instance_counter}'
        print(self._name)
        self._learner = learner
        self._dataset = dataset
        self._update_interval = update_interval
        self._episode_counter = -init_episodes

    def on_episode_end(self):
        print(f'{self._name}: {self._episode_counter}')
        self._episode_counter += 1
        print(f'{self._name}: {self._episode_counter}')
        if (self._episode_counter > 0)  and (self._episode_counter % self._update_interval == 0):
            print('enter')
            self._episode_counter = 0
            self._learner.update(self._dataset)

