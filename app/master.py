import os
from datetime import datetime

from flask import Flask, request

app = Flask(__name__)

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

@app.route('/turn_on')
def turn_on():
    master.turn_on()
    return 'OK' 

@app.route('/turn_off')
def turn_off():
    master.turn_off()
    return 'OK'

@app.route('/status')
def get_status():
    return {'status': master.get_status()}

@app.route('/info')
def get_info():
    return master.get_info()

@app.route('/push_episode')
def push_episode():
    data = request.get_json()
    master.push(data)
    return 'OK'

if __name__ == '__main__':
    master = Master(update_period=10)
    app.run(host='0.0.0.0', port='5000')



