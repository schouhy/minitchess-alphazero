from exp.environment import MinitChessEnvironment
from exp.policy import SimpleAlphaZeroPolicy
from exp.agent import SimpleAlphaZeroAgent
from exp.callbacks import InfoRecorder, MonteCarloInit
from exp.dataset import RemoteDataset

from erlyx import run_episodes

import os
import logging

import torch

logging.basicConfig(filename='/app/log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logging.warning('test')


MASTER_FLASK_URL = os.getenv('MASTER_FLASK_URL', None)
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', None)


def runEpisode(env, policy, agent, dataset):
    logging.debug('call to run-episode')
    callbacks = [InfoRecorder(dataset), MonteCarloInit(agent)]
    run_episodes(env, agent, 1, callbacks=callbacks)
    return "done"


if __name__ == '__main__':
    env = MinitChessEnvironment()
    policy = SimpleAlphaZeroPolicy(env, num_simulations=25)
    agent = SimpleAlphaZeroAgent(environment=env, policy=policy)
    dataset = RemoteDataset(MASTER_FLASK_URL + '/post-episode')
    while True:
        try:
            policy.model.load_state_dict(torch.load(WEIGHTS_PATH))
        except FileNotFoundError:
            logging.debug(f'No file {WEIGHTS_PATH}, skipping load')
        runEpisode(env, policy, agent, dataset)

