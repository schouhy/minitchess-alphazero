import json
import logging
import os
from datetime import datetime
from enum import Enum

import requests
import torch
from erlyx import run_episodes

from exp.agent import SimpleAlphaZeroAgent
from exp.callbacks import InfoRecorder, MonteCarloInit
from exp.dataset import RemoteDataset, SimpleAlphaZeroDataset
from exp.environment import MinitChessEnvironment
from exp.policy import Network, SimpleAlphaZeroPolicy

MASTER_URL = os.getenv('MASTER_URL', 'localhost')
STATUS_URL = '/'.join([MASTER_URL, 'status'])
PUSH_EPISODE_URL = '/'.join([MASTER_URL, 'push_episode'])
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', 'weights.pt')


class MasterOfPuppetsStatus(Enum):
    OFF = 1
    SIMULATE = 2
    TRAIN = 3

class BasePuppet:
    def get_master_status(self):
        response = requests.get(STATUS_URL)
        if response.status_code == 200:
            response_json = json.loads(response.content)
            status = MasterOfPuppetsStatus(int(response_json['status']))
            logging.info(f'MasterOfPuppetsStatus: {status}')
            return status
        logging.info(
            f'Master of puppets returned status code: {response.status_code}')
        return False


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


class TrainPuppet(BasePuppet):
    def __init__(self, userid, key):
        self._get_url = '/'.join([GET_TRAIN_DATA_URL, userid, key])
        self._push_url = '/'.join([PUSH_WEIGHTS_URL, userid, key])

    def _get_train_data(self):
        pass

    def _push_weights(self):
        pass

    def train(self):
        pass
        

class MasterOfPuppets:
    def __init__(self, update_period):
        self._info = []
        self._status = MasterOfPuppetsStatus.SIMULATE
        self._updatePeriod = update_period
        self._dataset = self._init_dataset() 

    def _init_dataset(self):
        return SimpleAlphaZeroDataset(max_length=1_000_000)

    def flush_data(self):
        data = self._dataset._memory.copy()
        self._dataset = self._init_dataset()
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
            self._dataset.push(data)
            if len(self._info) % self._updatePeriod == 0:
                self._status = MasterOfPuppetsStatus.TRAIN
            return 'done'
        return 'not simulating, skipping...'
