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

def _load_confidence_threshold() -> float:
    """Load and validate CONFIDENCE_THRESHOLD from environment.

    Returns:
        Confidence threshold as a float.

    Raises:
        ValueError: If the value is not a valid float or is outside [0.0, 1.0].
    """
    raw = os.getenv("CONFIDENCE_THRESHOLD", "0.5")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"CONFIDENCE_THRESHOLD must be a number, got: {raw!r}"
        )
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got: {value}"
        )
    return value


CONFIDENCE_THRESHOLD: float = _load_confidence_threshold()
