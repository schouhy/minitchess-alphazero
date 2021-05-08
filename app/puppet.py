import logging
import os

from time import sleep
from app.base import SimulatePuppet, MasterOfPuppetsStatus, RemoteStatus

USERID = os.getenv('USERID', os.getenv('HOSTNAME', 'Player'))
KEY = os.getenv('KEY', 'None')
logging.basicConfig(filename='log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')

if __name__ == '__main__':
    puppet = SimulatePuppet(USERID, KEY)
    remote_status = RemoteStatus()
    while True:
        while remote_status['system_status'] == MasterOfPuppetsStatus.SIMULATE:
            puppet.run_episodes(10)
        sleep(60 * 5)
