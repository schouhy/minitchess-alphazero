from flask import Flask, request

app = Flask(__name__)

from app.base import MasterOfPuppets

master = MasterOfPuppets(update_period=10)

import logging

logging.basicConfig(filename='/app/log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')

WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', 'weights.pt')


def check_credentials(userid, key, is_admin):
    return True


def only_authenticated(userid, key, is_admin=True):
    def decorator(func):
        def _inner(userid, key):
            if check_credentials(userid, key, is_admin):
                return func()
            else:
                return "Invalid user or password", 200

        return _inner


# def block_remote_addr(func):
#     def _inner():
#         if request.remote_addr == '127.0.0.1':
#             func()
#             return 'OK'
#         return 'invalid remote address'
#     return _inner


@app.route('/simulate/<userid>/<key>')
@only_authenticated(userid, key)
def simulate():
    master.simulate()


@app.route('/off/<userid>/<key>')
@only_authenticated(userid, key)
def turn_off():
    master.off()


@app.route('/train/<userid>/<key>')
@only_authenticated(userid, key)
def train():
    master.train()


@app.route('/status')
def get_status():
    logging.info(f'Hit get_status: {master.get_status()}')
    return {'status': master.get_status()}


@app.route('/info')
def get_info():
    logging.info(f'Hit info: {master.get_info()}')
    return {'info': master.get_info(), 'counter': master.get_counter()}


@app.route('/push_episode/<userid>/<key>', methods=['POST'])
@only_authenticated(userid, key, is_admin=False)
def push_episode():
    data = request.get_json()
    logging.info(f'Hit push_episode: userid={userid}, data={data}')
    master.push(userid=userid, data=data)
    return 'OK'


@app.route('/get_latest_data/<userid>/<key>')
@only_authenticated(userid, key)
def get_latest_data():
    return {'data': master.flush_data()}


#
# @app.route('/push_weights/<uuid>', methods=['POST'])
# def push_weights(uuid):
#     if check_uuid(uuid):
#         with open(WEIGHTS_PATH, 'wb') as outfile:
#             outfile.write(request.files['file'])
#             master.simulate()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
