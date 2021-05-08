from flask import Flask, request, send_file

app = Flask(__name__)

import os
import datetime
import jsonpickle
from app.base import MasterOfPuppets

master = MasterOfPuppets(update_frequency=3)

import logging

logging.basicConfig(filename='/app/log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')

from pathlib import Path

WEIGHTS_PATH = Path(os.getenv('WEIGHTS_PATH', '.'))


def check_credentials(userid, key, is_admin):
    return (userid is not None) and (key is not None)


def only_authenticated(is_admin=True):
    def decorator(func):
        def _inner(userid, key):
            if check_credentials(userid, key, is_admin):
                return func(userid, key)
            else:
                return "Invalid user or password", 200

        _inner.__name__ = func.__name__
        return _inner

    return decorator


# def block_remote_addr(func):
#     def _inner():
#         if request.remote_addr == '127.0.0.1':
#             func()
#             return 'OK'
#         return 'invalid remote address'
#     return _inner



### Switchs
@app.route('/simulate/<userid>/<key>')
@only_authenticated()
def simulate(userid=None, key=None):
    master.simulate()
    return 'OK'


@app.route('/off/<userid>/<key>')
@only_authenticated()
def turn_off(userid=None, key=None):
    master.off()
    return 'OK'


@app.route('/train/<userid>/<key>')
@only_authenticated()
def train(userid=None, key=None):
    master.train()
    return 'OK'


### Info
@app.route('/status')
def get_status():
    logging.info(f'Hit get_status: {master.get_status()}')
    return master.get_status()


@app.route('/info')
def get_info():
    logging.info(f'Hit info')
    return {'info': master.get_info()}


### Operations
@app.route('/push_episode/<userid>/<key>', methods=['POST'])
@only_authenticated(is_admin=False)
def push_episode(userid=None, key=None):
    data = request.get_json()
    return master.push(userid=userid, data=data)

@app.route('/report_episode_counter/<userid>/<key>', methods=['POST'])
@only_authenticated()
def report_episode_counter(userid=None, key=None):
    data = request.get_json()
    current_period = int(data['episode_counter']) // master.update_frequency
    logging.info(f'Hit to report_episode: current_period={current_period}, next_train_period={master.next_train_period}')
    if current_period > master.next_train_period:
        master.next_train_period = current_period
        master.train()
        master.flush_data()
    return 'OK', 200


@app.route('/get_latest_data/<userid>/<key>')
@only_authenticated()
def get_latest_data(userid=None, key=None):
    logging.info(f'Hit get_latest_data: userid={userid}')
    logging.debug(f'len(data) = {len(master._data)}')
    return master.flush_data()


@app.route('/get_weights/<userid>/<key>')
@only_authenticated(is_admin=False)
def get_weights(userid=None, key=None):
    logging.info(f'userid: {userid}')
    try:
        version = master.get_status()['weights_version']
        path = WEIGHTS_PATH/version
        return send_file(open(path, 'rb'),
                         attachment_filename=version,
                         mimetype='application/octet-stream')
    except FileNotFoundError:
        return "Weights file not found!"

@app.route('/push_weights/<userid>/<key>', methods=['POST'])
@only_authenticated()
def push_weights(userid, key):
    state_dict_json = request.get_json()
    if state_dict_json:
        master.state_dict = jsonpickle.decode(state_dict_json)
    master.simulate()
    return 'OK', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
