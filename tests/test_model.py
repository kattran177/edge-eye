"""Tests for the edge model wrapper."""

from unittest.mock import patch, MagicMock
from src.edge.model import EdgeModel, Detection


class TestDetection:
    def test_detection_dataclass(self):
        d = Detection(class_name="person", confidence=0.95, bbox=(10, 20, 100, 200))
        assert d.class_name == "person"
        assert d.confidence == 0.95
        assert d.bbox == (10, 20, 100, 200)


class TestEdgeModel:
    @patch("src.edge.model.YOLO")
    def test_detect_returns_detections(self, mock_yolo_cls):
        """Model wrapper should parse YOLO output into Detection objects."""
        # Mock the YOLO model and its results
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_box = MagicMock()
        mock_box.cls = [MagicMock(__int__=lambda s: 0)]
        mock_box.conf = [MagicMock(__float__=lambda s: 0.92)]
        mock_box.xyxy = [MagicMock(tolist=lambda: [10, 20, 100, 200])]
        mock_box.__len__ = lambda s: 1

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        mock_result.names = {0: "person"}

        mock_model.return_value = [mock_result]

        model = EdgeModel()
        detections = model.detect("fake_image.jpg")

        assert len(detections) == 1
        assert detections[0].class_name == "person"

    @patch("src.edge.model.YOLO")
    def test_detect_no_boxes(self, mock_yolo_cls):
        """Should return empty list when no objects detected."""
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]

        model = EdgeModel()
        detections = model.detect("empty.jpg")
        assert detections == []
