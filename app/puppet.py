import logging
import os
from time import sleep

import paho.mqtt.client as mqtt

from app.base import MasterOfPuppetsStatus, RemoteStatus, SimulatePuppet

USERID = os.getenv('USERID', os.getenv('HOSTNAME', 'Player'))
# MQTT info
MQTT_BROKER_HOST = os.getenv('MQTT_BROKER_HOST')
MQTT_USERNAME = os.getenv('MQTT_USERNAME')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
LEARNER_TOPIC = os.getenv('LEARNER_TOPIC')
PUBLISH_EPISODE_TOPIC = os.getenv('PUBLISH_EPISODE_TOPIC')

logging.basicConfig(filename='log',
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)
logging.warning('test')


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(LEARNER_TOPIC)


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    assert msg.topic == LEARNER_TOPIC
    puppet = userdata['puppet']
    msg_payload = json.loads(msg.payload)
    puppet.remote_status = MasterOfPuppetsStatus[msg_payload['status']]
    if puppet.is_simulating() or (puppet.remote_status != MasterOfPuppetsStatus.SIMULATE):
        return
    if puppet.weights_version != msg_payload['weights_version']:
        response = requests.get(GET_WEIGHTS_URL)
        if response.status_code == 200:
            content = json.loads(response.content)
            assert content['version'] == msg_payload['weights_version']
            content['weights'] = jsonpickle.decode(content['weights'])
            puppet.load_remote_weights(**content)
    puppet.run_episodes(NUM_SIMULATIONS, client)

puppet = SimulatePuppet(USERID, PUBLISH_EPISODE_TOPIC)
client = mqtt.Client(userdata={'puppet': puppet})
client.on_connect = on_connect
client.on_message = on_message

client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.connect(MQTT_BROKER_HOST, 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
client.loop_forever()
