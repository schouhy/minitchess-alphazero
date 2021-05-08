import json
import jsonpickle
import logging
import os
import io
from collections import deque
from datetime import datetime
from enum import IntEnum

import requests
import torch
from pathlib import Path
from erlyx import run_episodes

from exp.agent import SimpleAlphaZeroAgent
from exp.callbacks import InfoRecorder, MonteCarloInit
from exp.dataset import SimpleAlphaZeroDataset
from exp.environment import MinitChessEnvironment
from exp.learner import SimpleAlphaZeroLearner
from exp.policy import Network, SimpleAlphaZeroPolicy

MASTER_URL = os.getenv('MASTER_URL', 'localhost')
STATUS_URL = '/'.join([MASTER_URL, 'status'])
PUSH_EPISODE_URL = '/'.join([MASTER_URL, 'push_episode'])
REPORT_EPISODE_COUNTER_URL = '/'.join([MASTER_URL, 'report_episode_counter'])
PUSH_WEIGHTS_URL = '/'.join([MASTER_URL, 'push_weights'])
GET_TRAIN_DATA_URL = '/'.join([MASTER_URL, 'get_latest_data'])
GET_WEIGHTS_URL = '/'.join([MASTER_URL, 'get_weights'])
WEIGHTS_PATH = Path(os.getenv('WEIGHTS_PATH', '.'))
NUM_SIMULATIONS = 25

class MasterOfPuppetsStatus(IntEnum):
    OFF = 1
    SIMULATE = 2
    TRAIN = 3


class RemoteGetter:
    def __init__(self, url):
        self._url = url

    def get(self):
        try:
            response = requests.get(self._url)
            if response.status_code == 200:
                return response.content
            logging.info(
                f'Master of puppets returned status code: {response.status_code}'
            )
        except requests.exceptions.ConnectionError:
            logging.info("Master of puppets is not responding..")


class RemoteStatus:
    def __init__(self):
        self._remote_getter = RemoteGetter(STATUS_URL)

    def __getitem__(self, attr):
        response = self._remote_getter.get()
        if response is None:
            return None
        data = json.loads(response)[attr]
        if attr == 'system_status':
            if data is None:
                return MasterOfPuppetsStatus.OFF
            return MasterOfPuppetsStatus(int(data))
        if attr == 'num_episodes':
            return int(data)
        return data


class Weights:
    @property
    def state_dict(self):
        raise NotImplementedError

    @property
    def version(self):
        raise NotImplementedError


class LocalWeights(Weights):
    def __init__(self):
        self._version = None
        self._state_dict = None
        self.state_dict = Network().state_dict()

    @property
    def state_dict(self):
        return self._state_dict

    @state_dict.setter
    def state_dict(self, new_state_dict):
        self._version = datetime.now().strftime('%Y%m%d%H%M%S')
        self._state_dict = new_state_dict
        logging.info(f'Updating weights. New weights version: {self.version}')
        torch.save(self.state_dict, WEIGHTS_PATH / self.version)

    @property
    def version(self):
        return self._version

class RemoteWeights(Weights):
    def __init__(self, userid, key):
        get_weights_url = '/'.join([GET_WEIGHTS_URL, userid, key])
        self._weights_getter = RemoteGetter(get_weights_url)
        self._status = RemoteStatus()

    @property
    def state_dict(self):
        response = self._weights_getter.get()
        if response is not None:
            logging.info('Downloaded new weights')
            return torch.load(io.BytesIO(response))

    @property
    def version(self):
        return self._status['weights_version']


class RemoteDataset:
    def __init__(self, url, weights_version):
        self._url = url
        self._weights_version = weights_version
        self._remote_status = RemoteStatus()

    def push(self, data):
        status = self._remote_status['system_status']
        data = {'episode': data, 'weights_version': self._weights_version}
        if status == MasterOfPuppetsStatus.SIMULATE:
            response = requests.post(self._url, json=data)
            return response.status_code != 200
        else:
            logging.info(f'Not pushing episode. Master status is {status}')
            return True


class SimulatePuppet:
    def __init__(self, userid, key):
        self._remote_weights = RemoteWeights(userid, key)
        self._local_weights_version = None

        self._push_url = '/'.join([PUSH_EPISODE_URL, userid, key])

        self._env = MinitChessEnvironment()
        self._network = Network()
        policy = SimpleAlphaZeroPolicy(network=self._network)
        self._agent = SimpleAlphaZeroAgent(environment=self._env,
                                           policy=policy,
                                           num_simulations=NUM_SIMULATIONS)

    def run_episodes(self, num_episodes):
        logging.debug('call to run_episode')
        self._sync_weights()
        self._dataset = RemoteDataset(self._push_url, self._local_weights_version)
        callbacks = [InfoRecorder(self._dataset), MonteCarloInit(self._agent)]
        run_episodes(self._env, self._agent, num_episodes, callbacks=callbacks)

    def _sync_weights(self):
        if self._remote_weights.version == self._local_weights_version:
            return
        new_state_dict = self._remote_weights.state_dict
        if new_state_dict is not None:
            self._local_weights_version = self._remote_weights.version
            self._network.load_state_dict(new_state_dict)


class LearnPuppet:
    class Status(IntEnum):
        ERROR = 0
        SUCCESS = 1

    def __init__(self, userid, key, batch_size, learning_rate):
        self._status = RemoteStatus()
        self._remote_weights = RemoteWeights(userid, key)

        get_data_url = '/'.join([GET_TRAIN_DATA_URL, userid, key])
        self._data_getter = RemoteGetter(get_data_url)
        self._push_url = '/'.join([PUSH_WEIGHTS_URL, userid, key])
        self._report_episode_counter_url = '/'.join([REPORT_EPISODE_COUNTER_URL, userid, key])
        self._episode_counter = 0

        self._env = MinitChessEnvironment()
        self._dataset = SimpleAlphaZeroDataset(max_length=1_000_000)
        self._network = Network()
        self._learner = SimpleAlphaZeroLearner(self._env, NUM_SIMULATIONS, self._network,
                                               batch_size, learning_rate)

    def _report_episode_counter(self):
        response = requests.post(self._report_episode_counter_url, json={'episode_counter': self._episode_counter})
        if response.status_code == 200:
            logging.info(f'Reported episode counter {self._episode_counter}')
        else:
            logging.info(f'Failed to report episode counter. Master of puppets returned status code {response.status_code}')


    def get_train_data(self):
        response = self._data_getter.get()
        if response is None:
            return
        response_json = json.loads(response)
        data = response_json['data']
        relative_episode_counter = int(response_json['relative_episode_counter'])
        self._dataset.push(data)
        self._episode_counter += relative_episode_counter
        logging.info(f'Added {len(data)} new samples, episode counter: {self._episode_counter}')
        self._report_episode_counter()

    def update(self):
        self._network.load_state_dict(self._remote_weights.state_dict)
        results = self._learner.update(self._dataset)
        logging.info(f'New agent won {results*100}% of games')
        data = {'state_dict': None, 'results': results}
        if results > 0.55:
            logginf.info('Pushing new weights!')
            data['state_dict'] = jsonpickle.encode(self._network.state_dict())
        while True:
            response = requests.post(self._push_url, json=data)
            status_code = response.status_code
            if status_code == 200:
                break
            logging.info(f'Cannot report end of training. Master returned status code {status_code}. Retry in 10 seconds...')
            sleep(10)


class MasterOfPuppets:
    def __init__(self, update_frequency):
        self._info = []
        self._system_status = MasterOfPuppetsStatus.SIMULATE
        self._data, self._relative_episode_counter  = self._init_dataset()
        self.update_frequency = update_frequency
        self.next_train_period = 0
        self._weights = LocalWeights()

    def get_system_status(self):
        return self._system_status

    @staticmethod
    def _init_dataset():
        return deque(maxlen=1_000_000), 0 

    def update_weights(self, new_state_dict):
        self._weights.state_dict = new_state_dict

    def flush_data(self):
        data = list(self._data)
        relative_episode_counter = self._relative_episode_counter
        self._data, self._relative_episode_counter = self._init_dataset()
        return {'data': data, 'relative_episode_counter': relative_episode_counter}

    def get_counter(self):
        return len(self._info)

    def get_status(self):
        return {
            'system_status': self.get_system_status(),
            'num_episodes': self.get_counter(),
            'relative_episode_counter': self._relative_episode_counter,
            'weights_version': self._weights.version,
        }

    def get_info(self):
        return self._info

    def simulate(self):
        logging.info('Switching to simulate state')
        self._system_status = MasterOfPuppetsStatus.SIMULATE

    def off(self):
        logging.info('Switching to off state')
        self._system_status = MasterOfPuppetsStatus.OFF

    def train(self):
        logging.info('Switching to train state')
        self._system_status = MasterOfPuppetsStatus.TRAIN

    def push(self, userid, data):
        if self._system_status == MasterOfPuppetsStatus.SIMULATE:
            if data['weights_version'] != self._weights.version:
                logging.info(
                    f'version missmatch on push data (userid: {userid})')
                return 'Version missmatch', 400
            logging.info(
                f'Pushing episode number {self.get_counter()} with {len(data["episode"])} samples from userid={userid}'
            )
            self._info.append((userid, str(datetime.now())))
            self._data.extend(data['episode'])
            self._relative_episode_counter += 1
            return 'Success', 200
        return 'Not simulating', 400

