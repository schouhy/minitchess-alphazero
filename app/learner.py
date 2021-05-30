import logging
import sys
logging.StreamHandler(sys.stdout)
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')

import os
import json
import requests
from time import sleep

import paho.mqtt.client as mqtt

from app.base import LearnPuppet, MasterOfPuppetsStatus, download_weights

INITIALIZE_WITH_REMOTE_WEIGHTS = False

USERID = os.getenv('USERID', os.getenv('HOSTNAME', 'Player'))

MQTT_BROKER_HOST = os.getenv('MQTT_BROKER_HOST')
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
LEARNER_TOPIC = os.getenv('LEARNER_TOPIC')
PUBLISH_EPISODE_TOPIC = os.getenv('PUBLISH_EPISODE_TOPIC')
MINITCHESS_ALPHAZERO_VERSION = os.getenv('MINITCHESS_ALPHAZERO_VERSION')
logging.info(f'minitchess-alphazero version: {MINITCHESS_ALPHAZERO_VERSION}')

counter_users = {}
counter_versions = {}

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code " + str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    logging.info(f"Subscribed to {PUBLISH_EPISODE_TOPIC}")
    client.subscribe(PUBLISH_EPISODE_TOPIC)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    assert msg.topic == PUBLISH_EPISODE_TOPIC
    learner = userdata['learner']
    msg_payload = json.loads(msg.payload)
    if msg_payload.get('minitchess_alphazero_version', None) != MINITCHESS_ALPHAZERO_VERSION:
        logging.warning(f'Received message from {msg_payload["userid"]} with wrong minitchess-alphazero version')
        return
    if msg_payload.get('weights_version', None) != learner.weights_version:
        logging.warning(f'Received message from {msg_payload["userid"]} with wrong weights version')
        return
    try:
        learner.push_data(msg_payload['episode'])
        counter_users[msg_payload['userid']] = counter_users.get(msg_payload['userid'], 0) + 1
        counter_versions[msg_payload['weights_version']] = counter_versions.get(msg_payload['weights_version'], 0) + 1
        logging.info(f'users counter: {counter_users}')
        logging.info(f'version counter: {counter_versions}')
        logging.info(f'dataset count: {len(learner._dataset)}')
    except Exception as e:
        logging.info(f'Exception on push_data: {e}')


last_episode_period = 0
episode_frequency = 100
batch_size = 64
epochs = 20
optim_params = {'lr': 1e-3}

learner = LearnPuppet(USERID, batch_size, epochs, optim_params)
if INITIALIZE_WITH_REMOTE_WEIGHTS:
    content = download_weights()
    logging.info(f'Initializing weights with remote version: {content["version"]}')
    learner.weights = content['weights']

client = mqtt.Client(userdata={'learner': learner})
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.connect(MQTT_BROKER_HOST, 1883, 60)
client.loop_start()

def push_weights(url, data):
    # Upload new weights
    upload_success = False
    while not upload_success:
        response = requests.post(url, json=data)
        status_code = response.status_code
        if status_code == 200:
            upload_success = True
        else:
            logging.info(
                f'Cannot report end of training. Master returned status code {status_code}. Retry in 10 seconds...'
            )
            sleep(1)


try:
    push_weights(learner.push_url, learner.get_weights_dict())
    while True:
        current_period = learner.episode_counter // episode_frequency
        if last_episode_period < current_period:
            # TRAIN
            logging.info('TRAINING')
            learner.train()
            last_episode_period = current_period
            client.publish(LEARNER_TOPIC, json.dumps({'status': learner.status}), qos=1)
            result = learner.update()
            if result:
                logging.info('Uploading new weights')
                push_weights(learner.push_url, result)
                logging.info('Done uploading weights')
            logging.info('DONE TRAINING')
        # SIMULATE
        learner.simulate()
        data = {
            'status': learner.status,
            'weights_version': learner.weights_version,
            'current_period': current_period,
            'minitchess_alphazero_version': MINITCHESS_ALPHAZERO_VERSION
        }
        client.publish(LEARNER_TOPIC, json.dumps(data), qos=1)
        sleep(3)
finally:
    client.publish(LEARNER_TOPIC, json.dumps({'status': MasterOfPuppetsStatus.OFF.name}), qos=1)
    client.loop_stop()
