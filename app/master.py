from flask import Flask, request

app = Flask(__name__)

from app.base import MasterOfPuppets

import logging

logging.basicConfig(filename='/app/log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')


@app.route('/turn_on')
def turn_on():
    logging.info('Hit turn_on')
    if request.remote_addr == '127.0.0.1':
        master.turn_on()
        return 'OK'
    return 'invalid remote address'


@app.route('/turn_off')
def turn_off():
    logging.info('Hit turn_off')
    if request.remote_addr == '127.0.0.1':
        master.turn_off()
        return 'OK'
    return 'invalid remote address'


@app.route('/status')
def get_status():
    logging.info(f'Hit get_status: {master.get_status()}')
    return {'status': master.get_status()}


@app.route('/info')
def get_info():
    logging.info(f'Hit info: {master.get_info()}')
    return {'info': master.get_info(), 'counter': master.get_counter()}


@app.route('/push_episode', methods=['POST'])
def push_episode():
    data = request.get_json()
    logging.info(f'Hit push_episode: {data}')
    master.push(data)
    return 'OK'


if __name__ == '__main__':
    master = MasterOfPuppets(update_period=10)
    app.run(host='0.0.0.0', port='5000')
