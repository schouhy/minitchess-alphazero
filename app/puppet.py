import logging
import sys
logging.StreamHandler(sys.stdout)
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')

import json
import jsonpickle
import os
import requests
from time import sleep

import paho.mqtt.client as mqtt

from app.base import MasterOfPuppetsStatus, SimulatePuppet

USERID = os.getenv('USERID', os.getenv('HOSTNAME', 'Player'))
# MQTT info
MQTT_BROKER_HOST = os.getenv('MQTT_BROKER_HOST')
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
LEARNER_TOPIC = os.getenv('LEARNER_TOPIC')
PUBLISH_EPISODE_TOPIC = os.getenv('PUBLISH_EPISODE_TOPIC')
LOGGER_URL = os.getenv('LOGGER_URL', 'localhost')
GET_WEIGHTS_URL = '/'.join([LOGGER_URL, 'get_weights']) 
MINITCHESS_ALPHAZERO_VERSION = os.getenv('MINITCHESS_ALPHAZERO_VERSION')



# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code " + str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(LEARNER_TOPIC)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    assert msg.topic == LEARNER_TOPIC
    puppet = userdata['puppet']
    msg_payload = json.loads(msg.payload)
    logging.info(msg_payload)
    puppet.remote_status = MasterOfPuppetsStatus[msg_payload['status']]
    if 'minitchess_alphazero_version' in msg_payload:
        puppet.remote_version = msg_payload['minitchess_alphazero_version']
    if 'weights_version' in msg_payload:
        puppet.remote_weights_version = msg_payload['weights_version']


puppet = SimulatePuppet(USERID, PUBLISH_EPISODE_TOPIC)
client = mqtt.Client(userdata={'puppet': puppet})
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.connect(MQTT_BROKER_HOST, 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
client.loop_start()
try:
    while (puppet.remote_version is None) or (puppet.remote_version == MINITCHESS_ALPHAZERO_VERSION):
        if (not puppet.is_simulating()) and (puppet.remote_status is not None) and (puppet.remote_status != MasterOfPuppetsStatus.OFF):
            if puppet.weights_version != puppet.remote_weights_version:
                try:
                    response = requests.get(GET_WEIGHTS_URL)
                    if response.status_code == 200:
                        content = json.loads(response.content)
                        assert content['version'] == puppet.remote_weights_version
                        content['weights'] = jsonpickle.decode(content['weights'])
                        puppet.load_weights(**content)
                except Exception as e:
                    logging.error(f'Exception raised while updating weights: {e}')
                    continue
            data = puppet.run_episodes(10, client)
        sleep(5)
finally:
    client.loop_stop()
