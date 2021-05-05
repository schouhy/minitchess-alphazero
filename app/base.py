import json
import logging
import os
import io
from collections import deque
from datetime import datetime
from enum import IntEnum

import requests
import torch
from erlyx import run_episodes

from exp.agent import SimpleAlphaZeroAgent
from exp.callbacks import InfoRecorder, MonteCarloInit
from exp.dataset import RemoteDataset, SimpleAlphaZeroDataset
from exp.environment import MinitChessEnvironment
from exp.learner import SimpleAlphaZeroLearner
from exp.policy import Network, SimpleAlphaZeroPolicy

MASTER_URL = os.getenv('MASTER_URL', 'localhost')
STATUS_URL = '/'.join([MASTER_URL, 'status'])
PUSH_EPISODE_URL = '/'.join([MASTER_URL, 'push_episode'])
GET_TRAIN_DATA_URL = '/'.join([MASTER_URL, 'get_latest_data'])
GET_WEIGHTS_URL = '/'.join([MASTER_URL, 'get_weights'])
PUSH_WEIGHTS_URL = 'None'
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', 'weights.pt')


class MasterOfPuppetsStatus(IntEnum):
    OFF = 1
    SIMULATE = 2
    TRAIN = 3


def handle_error_code(default_return_value):
    def decorator(func):
        def _inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.ConnectionError:
                logging.info("Master of puppets not responding...")
                return default_return_value
        return _inner
    return decorator

class BasePuppet:
    @handle_error_code(MasterOfPuppetsStatus.OFF)
    def get_master_status(self):
        response = requests.get(STATUS_URL)
        if response.status_code == 200:
            response_json = json.loads(response.content)
            status = MasterOfPuppetsStatus(int(response_json['status']))
            logging.info(f'MasterOfPuppetsStatus: {status}')
            return status
        logging.info(
            f'Master of puppets returned status code: {response.status_code}'
        )
        return MasterOfPuppetsStatus.OFF

    @handle_error_code(None)
    def get_weights(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            return torch.load(io.BytesIO(response.content))


class SimulatePuppet(BasePuppet):
    def __init__(self, userid, key):
        url = '/'.join([PUSH_EPISODE_URL, userid, key])
        self._env = MinitChessEnvironment()
        self._policy = SimpleAlphaZeroPolicy(self._env, num_simulations=25)
        self._agent = SimpleAlphaZeroAgent(environment=self._env,
                                           policy=self._policy)
        self._dataset = RemoteDataset(url=url)

    def run_episode(self):
        logging.debug('call to run_episode')
        self._load_weights()
        callbacks = [InfoRecorder(self._dataset), MonteCarloInit(self._agent)]
        run_episodes(self._env, self._agent, 1, callbacks=callbacks)

    def _load_weights(self):
        try:
            self._policy.model.load_state_dict(torch.load(WEIGHTS_PATH))
            logging.info('Successfully loaded weights')
        except FileNotFoundError:
            logging.info('Weights file not found. Skipping weight loading')


class LearnPuppet(BasePuppet):
    def __init__(self, userid, key, batch_size, learning_rate):
        self._get_data_url = '/'.join([GET_TRAIN_DATA_URL, userid, key])
        self._get_weights_url = '/'.join([GET_WEIGHTS_URL, userid, key])
        self._push_url = '/'.join([PUSH_WEIGHTS_URL, userid, key])
        self._dataset = SimpleAlphaZeroDataset(max_length=1_000_000)
        self._network = Network()
        self._learner = SimpleAlphaZeroLearner(self._network, batch_size, learning_rate)

    def get_train_data(self):
        try:
            response = requests.get(self._get_data_url)
            if response.status_code == 200:
                data = json.loads(response.content)['data']
                self._dataset.push(data)
                logging.info(f'Added {len(data)} new samples')
            else:
                logging.info(
                    f'Master of puppets returned status code {response.status_code}'
                )
        except requests.exceptions.ConnectionError:
            logging.info('Master of puppets not responding..')

    def _push_weights(self):
        pass

    def get_sample_size(self):
        return len(self._dataset)

    def learn(self):
        state_dict = self.get_weights(self._get_weights_url)
        if state_dict is not None:
            self._network.load_state_dict(state_dict)
            logging.info('loaded weights')
        self._learner.update(self._dataset)


class MasterOfPuppets:
    def __init__(self, update_period):
        self._info = []
        self._status = MasterOfPuppetsStatus.SIMULATE
        self._updatePeriod = update_period
        self._data = self._init_dataset()

    def _init_dataset(self):
        return deque(maxlen=1_000_000)

    def flush_data(self):
        data = list(self._data)
        self._data = self._init_dataset()
        return data

    def get_counter(self):
        return len(self._info)

    def get_status(self):
        return self._status

    def get_info(self):
        return self._info

    def simulate(self):
        self._status = MasterOfPuppetsStatus.SIMULATE

    def off(self):
        self._status = MasterOfPuppetsStatus.OFF

    def train(self):
        self._status = MasterOfPuppetsStatus.TRAIN

    def push(self, userid, data):
        if self._status == MasterOfPuppetsStatus.SIMULATE:
            self._info.append((userid, str(datetime.now())))
            self._data.extend(data)
            return 'done'
        return 'not simulating, skipping...'
