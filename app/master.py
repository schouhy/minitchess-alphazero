from flask import Flask, request

app = Flask(__name__)

import os
from datetime import datetime

from exp.dataset import SimpleAlphaZeroDataset

import logging

logging.basicConfig(filename='/app/log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')


class Master:
    def __init__(self, update_period):
        self._info = []
        self._status = False
        self._updatePeriod = update_period
        self._dataset = SimpleAlphaZeroDataset(max_length=1_000_000)

    def get_counter(self):
        return len(self._info)

    def get_status(self):
        return self._status

    def get_info(self):
        return self._info

    def turn_on(self):
        self._status = True

    def turn_off(self):
        self._status = False

    def push(self, data):
        self._info.append((data['id'], str(datetime.now())))
        self._dataset.push(data['episode'])
        if self._info % self._updatePeriod == 0:
            self.updateWeights()
            self._count_period = 0
        return 'done'

    def updateWeights(self):
        logging.log('this should trigger weight updates')

master = Master(update_period=10)

@app.route('/turn_on')
def turn_on():
    logging.info('Hit turn_on')
    logging.info(f'remote_addr: {request.remote_addr}')
    master.turn_on()


@app.route('/turn_off')
def turn_off():
    logging.info('Hit turn_off')
    master.turn_off()


@app.route('/status')
def get_status():
    logging.info(f'Hit get_status: {master.get_status()}')
    return {'status': master.get_status()}


@app.route('/info')
def get_info():
    logging.info(f'Hit info: {master.get_info()}')
    return {'info': master.get_info(), 'counter': master.get_counter()}


@app.route('/push_episode')
def push_episode():
    data = request.get_json()
    logging.info('Hit push_episode: {data}')
    master.push(data)
    return 'OK'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
