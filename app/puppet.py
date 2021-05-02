import logging
import os

from time import sleep
from app.base import Puppet

USERID = os.getenv('USERID', os.getenv('HOSTNAME', None))

logging.basicConfig(filename='log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')

if __name__ == '__main__':
    puppet = Puppet(USERID)
    while True:
        while puppet.get_master_status():
            puppet.run_episode()
        sleep(60 * 5)
