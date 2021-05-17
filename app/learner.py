import logging

logging.basicConfig(filename='log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')

import os
from time import sleep

import paho.mqtt.client as mqtt

from app.base import LearnPuppet, MasterOfPuppetsStatus, RemoteStatus

USERID = os.getenv('USERID', os.getenv('HOSTNAME', 'Player'))

MQTT_BROKER_HOST = os.getenv('MQTT_BROKER_HOST')
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
LEARNER_TOPIC = os.getenv('LEARNER_TOPIC')
PUBLISH_EPISODE_TOPIC = os.getenv('PUBLISH_EPISODE_TOPIC')


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(PUBLISH_EPISODE_TOPIC)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    assert msg.topic == PUBLISH_EPISODE_TOPIC
    learner = userdata['learner']
    learner.push_data(json.loads(msg.payload)['episode'])


learner = LearnPuppet(USERID, 32, 1e-4)
last_episode_period = 0
episode_frequency = 3

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

try:
    while True:
        current_period = learner.episode_counter // episode_frequency
        if last_episode_period < current_period:
            # TRAIN
            learner.train()
            last_episode_period = current_period
            client.publish(LEARNER_TOPIC, {'status': learner.status})
            result = learner.update()
            if result:
                # Upload new weights
                logging.info('Pushing new weights!')
                upload_success = False
                while not upload_success:
                    response = requests.post(self._push_url, json=result)
                    status_code = response.status_code
                    if status_code != 200:
                        upload_success = True
                    else:
                        logging.info(
                            f'Cannot report end of training. Master returned status code {status_code}. Retry in 10 seconds...'
                        )
                        sleep(1)

        # SIMULATE
        learner.simulate()
        data = {
            'status': learner.status,
            'weights_version': learner.weights_version,
            'current_period': current_period
        }
        client.publish(LEARNER_TOPIC, data)
        sleep(3)
finally:
    client.loop_stop()
