"""
Alert rule engine — checks detections against configurable rules.

In production edge AI systems, you don't just detect objects — you
trigger ACTIONS based on what's detected:
- "Person detected in restricted area" → security alert
- "Fire/smoke detected" → emergency notification
- "Package on conveyor belt" → production count +1

This is a simple rule engine that checks each detection against
a list of rules and returns triggered alerts.
"""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class Alert:
    """A triggered alert."""

    rule_name: str
    device_id: str
    message: str
    severity: str  # "info", "warning", "critical"
    triggered_at: str


# Configurable alert rules.
# In production these would come from a database or config service.
ALERT_RULES = [
    {
        "name": "person_detected",
        "description": "Alert when a person is detected",
        "class": "person",
        "min_confidence": 0.7,
        "severity": "info",
        "message": "Person detected with high confidence",
    },
    {
        "name": "vehicle_in_zone",
        "description": "Alert when a vehicle (car/truck/bus) is detected",
        "class_list": ["car", "truck", "bus"],
        "min_confidence": 0.6,
        "severity": "warning",
        "message": "Vehicle detected in monitored zone",
    },
    {
        "name": "animal_detected",
        "description": "Alert when an animal is detected",
        "class_list": ["dog", "cat", "bird", "horse", "sheep", "cow", "bear"],
        "min_confidence": 0.5,
        "severity": "info",
        "message": "Animal detected",
    },
]


def check_alerts(device_id: str, detections: list[dict]) -> list[Alert]:
    """
    Check detections against all alert rules.

    Args:
        device_id: Which device generated the detections
        detections: List of detection dicts with 'class' and 'confidence'

    Returns:
        List of triggered Alert objects
    """
    triggered = []
    now = datetime.now(timezone.utc).isoformat()

    for detection in detections:
        det_class = detection.get("class", "")
        det_conf = detection.get("confidence", 0)

        for rule in ALERT_RULES:
            # Check if this detection matches the rule
            matches = False

            if "class" in rule and det_class == rule["class"]:
                matches = True
            elif "class_list" in rule and det_class in rule["class_list"]:
                matches = True

            if matches and det_conf >= rule.get("min_confidence", 0):
                triggered.append(
                    Alert(
                        rule_name=rule["name"],
                        device_id=device_id,
                        message=f"{rule['message']} — {det_class} ({det_conf:.0%})",
                        severity=rule["severity"],
                        triggered_at=now,
                    )
                )

    return triggered
