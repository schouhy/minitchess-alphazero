import logging

logging.basicConfig(filename='/app/log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logging.warning('This will get logged to a file')

from flask import Flask, request
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

from exp.dataset import SimpleAlphaZeroDataset 

import os
import json


@app.route('/push_weights/<userid>', methods=['POST'])
def push_weights(userid):
    if userid != 'lenovo':
        return 'ERROR', 500
    data = request.get_json()
    logging.info(f'Received weights {data["version"]}')
    with open('weights', 'w') as file:
        json.dump(data, file)
    return 'OK' 

@app.route('/get_weights')
def get_weights():
    logging.info('call to get_weights')
    with open('weights', 'r') as file:
        data = json.load(file)
    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

