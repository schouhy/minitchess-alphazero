import logging
import os

from time import sleep
from app.base import LearnPuppet, MasterOfPuppetsStatus

USERID = os.getenv('USERID', os.getenv('HOSTNAME', 'Player'))
KEY = os.getenv('KEY', 'None')
logging.basicConfig(filename='log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')

if __name__ == '__main__':
    learner = LearnPuppet(USERID, KEY, 32, 1e-3)
    while True:
        while learner.get_master_status() == MasterOfPuppetsStatus.SIMULATE:
            sleep(15)
            learner.get_train_data()
            logging.info(f'Sample size: {learner.get_sample_size()}')
        sleep(60 * 5)
