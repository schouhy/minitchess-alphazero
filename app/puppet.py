import logging
import os

from time import sleep
from app.base import SimulatePuppet, MasterOfPuppetsStatus

USERID = os.getenv('USERID', os.getenv('HOSTNAME', 'Player'))
KEY = os.getenv('KEY', 'None')
logging.basicConfig(filename='log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')

if __name__ == '__main__':
    puppet = SimulatePuppet(USERID, KEY)
    while True:
        while puppet.get_master_status() == MasterOfPuppetsStatus.SIMULATE:
            puppet.run_episode()
        sleep(60 * 5)
