from flask import Flask, request
app = Flask(__name__)

from exp.dataset import SimpleAlphaZeroDataset 

import os
import logging

logging.basicConfig(filename='/app/log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logging.warning('This will get logged to a file')


class Master:
    def __init__(self, updatePeriod):
        self._counter = 0
        self._updatePeriod = updatePeriod
        self._dataset = SimpleAlphaZeroDataset(max_length=1_000_000)

    @property
    def counter(self):
        return self._counter

    def push(self, data):
        self._counter += 1
        self._dataset.push(data)
        if self._counter % self._updatePeriod == 0:
            self.updateWeights()
            self._count_period = 0
        return 'done'

    def updateWeights(self):
        logging.log('this should trigger weight updates')


master = Master(updatePeriod=int(os.getenv('MASTER_UPDATE_PERIOD', 10)))

@app.route('/post-episode', methods=['POST'])
def pushEpisode():
    logging.debug('call to post-episode')
    data = request.get_json()
    master.push(data)
    return 'OK' 

@app.route('/info')
def info():
    logging.debug('call to info')
    response = {'count': master.counter, 'last': master._dataset._memory[-1]}
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


