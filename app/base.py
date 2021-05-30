import io
import json
import logging
import os
from collections import deque
from datetime import datetime
from enum import IntEnum
from pathlib import Path

import jsonpickle
import requests
import torch
from erlyx import run_episodes

from exp.agent import RoundRobinReferee, SimpleAlphaZeroAgent
from exp.callbacks import InfoRecorder, MonteCarloInit
from exp.dataset import SimpleAlphaZeroDataset
from exp.environment import MinitChessEnvironment
from exp.learner import SimpleAlphaZeroLearner
from exp.policy import Network, SimpleAlphaZeroPolicy

LOGGER_URL = os.getenv('LOGGER_URL', 'localhost')
PUSH_WEIGHTS_URL = '/'.join([LOGGER_URL, 'push_weights'])
NUM_SIMULATIONS = 36
MINITCHESS_ALPHAZERO_VERSION = os.getenv('MINITCHESS_ALPHAZERO_VERSION')
GET_WEIGHTS_URL = '/'.join([LOGGER_URL, 'get_weights']) 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logging.info(f'PyTorch is using device {DEVICE}')

def download_weights():
    response = requests.get(GET_WEIGHTS_URL)
    logging.info('Downloading weights...')
    if response.status_code == 200:
        logging.info('Successfully downloaded weights')
        content = json.loads(response.content)
        content['weights'] = jsonpickle.decode(content['weights'])
        return content
    raise Exception(f"RLWEB returned status code {response.status_code} while getting weights")

class MasterOfPuppetsStatus(IntEnum):
    OFF = 1
    SIMULATE = 2
    TRAIN = 3


class MQTTDataset:
    def __init__(self, mqtt_client, puppet):
        self._puppet = puppet
        self._mqtt_client = mqtt_client

    def push(self, data):
        if self._puppet.remote_status != MasterOfPuppetsStatus.SIMULATE:
            logging.info(
                f'Not pushing episode. Master Status is not SIMULATE'
            )
            return True
        if self._puppet.remote_version != MINITCHESS_ALPHAZERO_VERSION:
            logging.info(
                f'Not pushing episode. Master version differs'
            )
            return True
        data = {
            'episode': data,
            'userid': self._puppet.userid,
            'weights_version': self._puppet.weights_version,
            'minitchess_alphazero_version': MINITCHESS_ALPHAZERO_VERSION
        }
        msginfo = self._mqtt_client.publish(self._puppet.publish_topic, json.dumps(data), qos=2)
        logging.info(f"published episode with mid={msginfo.mid}")


class SimulatePuppet:
    def __init__(self, userid, publish_topic):
        self._env = MinitChessEnvironment()
        self._network = Network().eval()
        self._policy = SimpleAlphaZeroPolicy(network=self._network)
        self._userid = userid
        self._publish_topic = publish_topic
        self._is_simulating = False
        self._weights_version = None
        self._remote_status = None
        self.remote_weights_version = None
        self.remote_version = None

    @property
    def userid(self):
        return self._userid

    @property
    def publish_topic(self):
        return self._publish_topic

    @property
    def weights_version(self):
        return self._weights_version

    @property
    def remote_status(self):
        return self._remote_status

    @remote_status.setter
    def remote_status(self, status):
        assert isinstance(status, MasterOfPuppetsStatus)
        # logging.info(f"Puppet: changing status to {status}")
        self._remote_status = status

    def run_episodes(self, num_episodes, mqtt_client):
        try:
            self._is_simulating = True
            logging.info('Starting simulations')
            dataset = MQTTDataset(mqtt_client, self)
            agents = [SimpleAlphaZeroAgent(environment=self._env,  policy=self._policy, num_simulations=NUM_SIMULATIONS) for _ in range(2)]
            callbacks = [InfoRecorder(dataset), MonteCarloInit(agents[0]), MonteCarloInit(agents[1])]
            with torch.no_grad():
                run_episodes(self._env,
                             RoundRobinReferee(agent_tuple=tuple(agents)),
                             num_episodes,
                             callbacks=callbacks, 
                             use_tqdm=False)
        except Exception as e:
            logging.error(f'Exception occurred: {e}')
        finally:
            self._is_simulating = False

    def load_weights(self, weights, version):
        logging.info(f'Loading weights {version} ...')
        self._network.load_state_dict(weights)
        self._weights_version = version

    def is_simulating(self):
        return self._is_simulating


class LearnPuppet:
    def __init__(self, userid, batch_size, epochs, optim_params):
        self._push_url = '/'.join([PUSH_WEIGHTS_URL, userid])

        self._env = MinitChessEnvironment()
        self._dataset = None 
        self._init_dataset()
        self._network = Network()
        self._learner = SimpleAlphaZeroLearner(self._env, NUM_SIMULATIONS,
                                               self._network, batch_size,
                                               epochs, optim_params)
        self._episode_counter = 0
        self._weights_version = None
        self._weights = None
        self.weights = self._network.state_dict()
        self._status = MasterOfPuppetsStatus.SIMULATE

    def _init_dataset(self):
        self._dataset = SimpleAlphaZeroDataset(max_length=1_000_000)

    @property
    def episode_counter(self):
        return self._episode_counter

    @property
    def push_url(self):
        return self._push_url

    @property
    def weights_version(self):
        return self._weights_version

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = {k: v.cpu() for k, v in value.items()} 
        self._weights_version = datetime.now().strftime('%Y%m%d%H%M%S')

    @property
    def status(self):
        return self._status.name

    def train(self):
        self._status = MasterOfPuppetsStatus.TRAIN

    def simulate(self):
        self._status = MasterOfPuppetsStatus.SIMULATE

    def push_data(self, data):
        if MasterOfPuppetsStatus[self.status] == MasterOfPuppetsStatus.SIMULATE:
            self._episode_counter += 1
            self._dataset.push(data)


    def update(self):
        self._network.load_state_dict(self.weights)
        result = self._learner.update(self._dataset)
        logging.info(f'New agent won {result*100}% of games')
        if result > 0.55:
            self.weights = self._network.state_dict()
            self._init_dataset()
            return self.get_weights_dict()

    def get_weights_dict(self):
        return {'weights': jsonpickle.encode(self.weights),
                'version': self.weights_version}
