import logging

logging.basicConfig(filename='log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')

import os
import json
import requests
from time import sleep

import paho.mqtt.client as mqtt

from app.base import LearnPuppet, MasterOfPuppetsStatus

USERID = os.getenv('USERID', os.getenv('HOSTNAME', 'Player'))

MQTT_BROKER_HOST = os.getenv('MQTT_BROKER_HOST')
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
LEARNER_TOPIC = os.getenv('LEARNER_TOPIC')
PUBLISH_EPISODE_TOPIC = os.getenv('PUBLISH_EPISODE_TOPIC')

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
    logging.info(f'received a message {msg.topic} from {msg_payload["userid"]}')
    try:
        learner.push_data(msg_payload['episode'])
        counter_users[msg_payload['userid']] = counter_users.get(msg_payload['userid'], 0) + 1
        counter_versions[msg_payload['weights_version']] = counter_versions.get(msg_payload['weights_version'], 0) + 1
        logging.info(f'users counter: {counter_users}')
        logging.info(f'version counter: {counter_versions}')
    except Exception as e:
        logging.info(f'Exception on push_data: {e}')
    logging.info(f'Episode count: {learner.episode_counter}')


learner = LearnPuppet(USERID, 32, 1e-4)
last_episode_period = 0
episode_frequency = 100

client = mqtt.Client(userdata={'learner': learner})
client.on_connect = on_connect
client.on_message = on_message

client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.connect(MQTT_BROKER_HOST, 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_start()

def push_data(url, data):
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
    push_data(learner.push_url, learner.get_weights_dict())
    while True:
        current_period = learner.episode_counter // episode_frequency
        logging.info(f'Current period: {current_period}')
        if last_episode_period < current_period:
            # TRAIN
            logging.info('TRAINING')
            learner.train()
            last_episode_period = current_period
            client.publish(LEARNER_TOPIC, json.dumps({'status': learner.status}), qos=1)
            result = learner.update()
            if result:
                logging.info('Uploading new weights')
                push_data(learner.push_url, result)
                logging.info('Done uploading weights')
            logging.info('DONE TRAINING')
        # SIMULATE
        learner.simulate()
        data = {
            'status': learner.status,
            'weights_version': learner.weights_version,
            'current_period': current_period
        }
        client.publish(LEARNER_TOPIC, json.dumps(data), qos=1)
        sleep(3)
finally:
    client.loop_stop()
