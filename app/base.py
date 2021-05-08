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
GET_TRAIN_DATA_URL = '/'.join([MASTER_URL, 'get_latest_data'])
GET_WEIGHTS_URL = '/'.join([MASTER_URL, 'get_weights'])
PUSH_WEIGHTS_URL = 'None'
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', 'weights.pt')


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
                'Master of puppets returned status code: {response.status_code}'
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
        self._version = self._version_generator()
        self._state_dict = Network().state_dict()

    @property
    def state_dict(self):
        return self._state_dict

    @property
    def version(self):
        return self._version

    @staticmethod
    def _version_generator():
        return datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    def update(self, state_dict):
        self._version = self._version_generator() 
        self._state_dict = state_dict
        torch.save(self.state_dict, WEIGHTS_PATH / self.version)


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
    def __init__(self, url):
        self._url = url
        self._remote_status = RemoteStatus()

    def push(self, data):
        status = self._remote_status['system_status']
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

        url = '/'.join([PUSH_EPISODE_URL, userid, key])
        self._dataset = RemoteDataset(url=url)

        self._env = MinitChessEnvironment()
        self._network = Network()
        policy = SimpleAlphaZeroPolicy(network=self._network)
        self._agent = SimpleAlphaZeroAgent(environment=self._env,
                                           policy=policy,
                                           num_simulations=25)

    def run_episodes(self, num_episodes):
        logging.debug('call to run_episode')
        self._sync_weights()
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

        self._env = MinitChessEnvironment()
        self._dataset = SimpleAlphaZeroDataset(max_length=1_000_000)
        self._network = Network()
        self._learner = SimpleAlphaZeroLearner(self._env, self._network,
                                               batch_size, learning_rate)

    def get_train_data(self):
        response = self._data_getter.get()
        if response is not None:
            data = json.loads(response)['data']
            self._dataset.push(data)
            logging.info(f'Added {len(data)} new samples')

    def update(self):
        self._network.load_state_dict(self._remote_weights.state_dict)
        results = self._learner.update(self._dataset)
        logging.info('New agent won {results*100}% of games')
        if results > 0.55:
            logginf.info('Pushing new weights!')
            self._push_weights()

    def _push_weights(self):
        state_dict_json = jsonpickle.encode(self._network.state_dict())
        response = requests.post(self._push_url, json=state_dict_json)


class MasterOfPuppets:
    def __init__(self, update_period):
        self._info = []
        self._system_status = MasterOfPuppetsStatus.SIMULATE
        self._updatePeriod = update_period
        self._data = self._init_dataset()
        self.weights = LocalWeights()

    def get_system_status(self):
        return self._system_status

    def _init_dataset(self):
        return deque(maxlen=1_000_000)

    def flush_data(self):
        data = list(self._data)
        self._data = self._init_dataset()
        return data

    def get_counter(self):
        return len(self._info)

    def get_status(self):
        return {
            'system_status': self.get_system_status(),
            'num_episodes': self.get_counter(),
            'weights_version': self.weights.version
        }

    def get_info(self):
        return self._info

    def simulate(self):
        self._system_status = MasterOfPuppetsStatus.SIMULATE

    def off(self):
        self._system_status = MasterOfPuppetsStatus.OFF

    def train(self):
        self._system_status = MasterOfPuppetsStatus.TRAIN

    def push(self, userid, data):
        if self._system_status == MasterOfPuppetsStatus.SIMULATE:
            if data['weights_version'] != self.weights.version:
                logging.info(
                    f'version missmatch on push data (userid: {userid})')
                return 'Version missmatch', 400
            logging.info(
                f'Pushing {len(data["episode"])} episode samples from userid={userid}'
            )
            self._info.append((userid, str(datetime.now())))
            self._data.extend(data['episode'])
            return 'Success', 200
        return 'Not simulating', 400
