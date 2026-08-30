"""Tests for the alert rule engine."""

from src.processor.alerts import check_alerts, Alert


class TestAlertRules:
    """Test detection → alert triggering."""

    def test_person_detected_triggers_alert(self):
        detections = [{"class": "person", "confidence": 0.85}]
        alerts = check_alerts("cam-01", detections)
        assert len(alerts) >= 1
        assert any(a.rule_name == "person_detected" for a in alerts)

    def test_low_confidence_no_alert(self):
        """Below threshold should not trigger."""
        detections = [{"class": "person", "confidence": 0.3}]
        alerts = check_alerts("cam-01", detections)
        person_alerts = [a for a in alerts if a.rule_name == "person_detected"]
        assert len(person_alerts) == 0

    def test_vehicle_triggers_warning(self):
        detections = [{"class": "car", "confidence": 0.75}]
        alerts = check_alerts("cam-01", detections)
        vehicle_alerts = [a for a in alerts if a.rule_name == "vehicle_in_zone"]
        assert len(vehicle_alerts) == 1
        assert vehicle_alerts[0].severity == "warning"

    def test_animal_detection(self):
        detections = [{"class": "dog", "confidence": 0.6}]
        alerts = check_alerts("cam-02", detections)
        assert any(a.rule_name == "animal_detected" for a in alerts)

    def test_unknown_class_no_alert(self):
        """Classes not in any rule should not trigger."""
        detections = [{"class": "laptop", "confidence": 0.99}]
        alerts = check_alerts("cam-01", detections)
        assert len(alerts) == 0

    def test_multiple_detections_multiple_alerts(self):
        detections = [
            {"class": "person", "confidence": 0.9},
            {"class": "car", "confidence": 0.8},
            {"class": "dog", "confidence": 0.7},
        ]
        alerts = check_alerts("cam-01", detections)
        assert len(alerts) >= 3

    def test_empty_detections(self):
        alerts = check_alerts("cam-01", [])
        assert alerts == []

    def test_alert_has_device_id(self):
        detections = [{"class": "person", "confidence": 0.9}]
        alerts = check_alerts("cam-99", detections)
        assert all(a.device_id == "cam-99" for a in alerts)
