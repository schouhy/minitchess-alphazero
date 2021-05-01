import json
import logging
import os

import requests
import torch
from time import sleep
from erlyx import run_episodes

from exp.agent import SimpleAlphaZeroAgent
from exp.callbacks import InfoRecorder, MonteCarloInit
from exp.dataset import RemoteDataset
from exp.environment import MinitChessEnvironment
from exp.policy import SimpleAlphaZeroPolicy

MASTER_URL = os.getenv('MASTER_URL', None)
STATUS_URL = '/'.join([MASTER_URL, 'status'])
PUSH_EPISODE_URL = '/'.join([MASTER_URL, 'push_episode'])
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', None)
USERID = os.getenv('USERID', None)

logging.basicConfig(filename='/app/log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logging.warning('test')


class Slave:
    def __init__(self, userid):
        assert userid is not None
        env = MinitChessEnvironment()
        self._policy = SimpleAlphaZeroPolicy(env, num_simulations=25)
        self._agent = SimpleAlphaZeroAgent(environment=env,
                                           policy=self._policy)
        self._dataset = RemoteDataset(userid=userid, url=PUSH_EPISODE_URL)

    def run_episode():
        logging.debug('call to run_episode')
        self._load_weights()
        callbacks = [InfoRecorder(self._dataset),
                     MonteCarloInit(self._agent)]
        run_episodes(self._env, self._agent, 1, callbacks=callbacks)

    def _load_weights(self):
        try:
            self._policy.load_state_dict(torch.load(WEIGHTS_PATH))
        except FileNotFoundError:
            logging.info('Weights file not found. Skipping weight loading')

    def get_master_status(self):
        response = requests.get(STATUS_URL)
        if response.status_code == '200':
            response_json = json.loads(response.content)
            status = bool(response_json['status']) 
            logging.info(f'Master status: {status}')
            return status 
        logging.info(f'Master returned status code: {response.status_code}')
        return False


if __name__ == '__main__':
    slave = Slave(USERID)
    while True:
        while slave.get_master_status():
            slave.run_episode()
        sleep(60 * 5)

