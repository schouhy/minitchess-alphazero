import logging

logging.basicConfig(filename='/app/log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logging.warning('This will get logged to a file')

from flask import Flask, request
app = Flask(__name__)

from exp.dataset import SimpleAlphaZeroDataset 

import os


weights = {'version': None, 'weights': None}

@app.route('/push_weights/<userid>', methods=['POST'])
def pushEpisode(userid):
    if userid != 'lenovo':
        return 'ERROR', 500
    global weights
    data = request.get_json()
    logging.info(f'Received weights {data["version"]}')
    weights['version'] = data['version']
    weights['weights'] = data['weights']
    return 'OK' 

@app.route('/get_weights')
def get_weights():
    global weights
    logging.info('call to get_weights')
    return weights

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

