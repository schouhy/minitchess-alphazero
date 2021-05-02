import json
import os

import requests
import torch
from erlyx import run_episodes

from datetime import datetime
from exp.agent import SimpleAlphaZeroAgent
from exp.callbacks import InfoRecorder, MonteCarloInit
from exp.dataset import RemoteDataset, SimpleAlphaZeroDataset
from exp.environment import MinitChessEnvironment
from exp.policy import SimpleAlphaZeroPolicy

import logging

MASTER_URL = os.getenv('MASTER_URL', 'localhost')
STATUS_URL = '/'.join([MASTER_URL, 'status'])
PUSH_EPISODE_URL = '/'.join([MASTER_URL, 'push_episode'])
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', 'weights.pt')


class Puppet:
    def __init__(self, userid):
        assert userid is not None
        self._env = MinitChessEnvironment()
        self._policy = SimpleAlphaZeroPolicy(self._env, num_simulations=25)
        self._agent = SimpleAlphaZeroAgent(environment=self._env,
                                           policy=self._policy)
        self._dataset = RemoteDataset(userid=userid, url=PUSH_EPISODE_URL)

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

    def get_master_status(self):
        response = requests.get(STATUS_URL)
        if response.status_code == 200:
            response_json = json.loads(response.content)
            status = bool(response_json['status'])
            logging.info(f'Master of puppets status: {status}')
            return status
        logging.info(
            f'Master of puppets returned status code: {response.status_code}')
        return False


class MasterOfPuppets:
    def __init__(self, update_period):
        self._info = []
        self._status = True
        self._updatePeriod = update_period
        self._dataset = SimpleAlphaZeroDataset(max_length=1_000_000)

    def get_counter(self):
        return len(self._info)

    def get_status(self):
        return self._status

    def get_info(self):
        return self._info

    def turn_on(self):
        self._status = True

    def turn_off(self):
        self._status = False

    def push(self, data):
        self._info.append((data['userid'], str(datetime.now())))
        self._dataset.push(data['episode'])
        if len(self._info) % self._updatePeriod == 0:
            self.updateWeights()
        return 'done'

    def updateWeights(self):
        logging.log('this should trigger weight updates')
