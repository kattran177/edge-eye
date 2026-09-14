"""
YOLOv8 model wrapper for edge inference.

WHAT IS YOLO?
YOLO (You Only Look Once) is a family of object detection models.
Unlike image classifiers that say "this image contains a cat",
YOLO detects MULTIPLE objects and their LOCATIONS in one pass:
  - "person at (100, 200, 300, 400) with 95% confidence"
  - "car at (500, 100, 800, 300) with 87% confidence"

YOLOv8 NANO is the smallest variant:
  - ~6MB model file
  - Runs on CPU at ~30fps on modern hardware
  - 80 object classes (person, car, dog, chair, etc.)
  - Perfect for edge devices (Raspberry Pi, Jetson Nano)

WHY NOT GPT-4 VISION?
  - Edge AI runs LOCALLY — no internet needed
  - Sub-10ms latency (vs 500ms+ for cloud APIs)
  - No per-image costs
  - Works in privacy-sensitive environments
  - Runs on a $35 Raspberry Pi
"""

from dataclasses import dataclass

from ultralytics import YOLO

from src.config import CONFIDENCE_THRESHOLD


@dataclass
class Detection:
    """A single object detection result."""

    class_name: str  # e.g., "person", "car", "dog"
    confidence: float  # 0.0 to 1.0
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) bounding box


class EdgeModel:
    """
    Wraps YOLOv8 for edge inference.

    Loads the model once, then runs inference on images.
    The model auto-downloads on first use (~6MB for nano).
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        """
        Load the YOLO model.

        Args:
            model_name: Model variant. Options:
                - yolov8n.pt: Nano (~6MB, fastest, least accurate)
                - yolov8s.pt: Small (~22MB)
                - yolov8m.pt: Medium (~52MB)
                - yolov8l.pt: Large (~87MB, slowest, most accurate)
        """
        self.model = YOLO(model_name)
        self.model_name = model_name

    def detect(
        self,
        image_path: str,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> list[Detection]:
        """
        Run object detection on an image.

        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence to include a detection

        Returns:
            List of Detection objects
        """
        # verbose=False suppresses YOLO's built-in logging
        results = self.model(image_path, verbose=False, conf=confidence_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                bbox = tuple(int(x) for x in boxes.xyxy[i].tolist())

                detections.append(
                    Detection(
                        class_name=result.names[cls_id],
                        confidence=round(conf, 4),
                        bbox=bbox,
                    )
                )

        return detections
