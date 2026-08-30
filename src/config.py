"""EdgeEye configuration."""

import os

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC_DETECTIONS = "edge/{device_id}/detections"
MQTT_TOPIC_HEARTBEAT = "edge/{device_id}/heartbeat"
MQTT_TOPIC_ALL_DETECTIONS = "edge/+/detections"
MQTT_TOPIC_ALL_HEARTBEATS = "edge/+/heartbeats"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://edgeeye:edgeeye@localhost:5432/edgeeye",
)

# Edge device settings
DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL", "5"))  # seconds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
