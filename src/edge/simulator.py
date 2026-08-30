"""
Edge device simulator — runs YOLO inference and publishes detections to MQTT.

WHAT IS MQTT?
MQTT (Message Queuing Telemetry Transport) is a lightweight messaging protocol
designed for IoT. It uses publish/subscribe pattern:

  Publisher (sensor) → Broker (Mosquitto) → Subscriber (server)

Key concepts:
- TOPICS: Hierarchical paths like "edge/cam-01/detections" (like URL paths)
- QoS levels: 0 (fire and forget), 1 (at least once), 2 (exactly once)
- RETAIN: Broker keeps the last message so new subscribers get it immediately
- WILL: Message sent if a device disconnects unexpectedly (dead man's switch)

Why MQTT over HTTP for IoT?
- 10x less bandwidth (minimal headers vs HTTP's verbose ones)
- Persistent connections (no reconnection overhead)
- Built-in "last will" for device offline detection
- Push model (server doesn't have to poll devices)
- Works on unreliable networks (3G, satellite)

AWS IoT Core, Azure IoT Hub, and Google Cloud IoT all use MQTT.
"""

import argparse
import json
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from glob import glob

import paho.mqtt.client as mqtt

from src.config import (
    MQTT_BROKER,
    MQTT_PORT,
    DETECTION_INTERVAL,
    CONFIDENCE_THRESHOLD,
)
from src.edge.model import EdgeModel


shutdown_requested = False


def handle_signal(signum, frame):
    global shutdown_requested
    print("\n[INFO] Shutting down edge device...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def main():
    parser = argparse.ArgumentParser(description="Edge device simulator")
    parser.add_argument(
        "--device-id",
        default=os.getenv("DEVICE_ID", "cam-01"),
        help="Unique device identifier",
    )
    parser.add_argument(
        "--image-dir",
        default=os.getenv("IMAGE_DIR", "sample_images"),
        help="Directory containing images to process",
    )
    args = parser.parse_args()

    device_id = args.device_id
    image_dir = args.image_dir

    print(f"[INFO] Edge device '{device_id}' starting...")

    # Load the ML model
    model = EdgeModel("yolov8n.pt")
    print("[INFO] YOLOv8 Nano model loaded")

    # Connect to MQTT broker
    client = mqtt.Client(client_id=device_id, protocol=mqtt.MQTTv311)

    # Last Will and Testament — broker publishes this if we disconnect unexpectedly
    will_payload = json.dumps({
        "device_id": device_id,
        "status": "offline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    client.will_set(f"edge/{device_id}/status", will_payload, qos=1, retain=True)

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except Exception as e:
        print(f"[FATAL] Cannot connect to MQTT broker: {e}")
        sys.exit(1)

    client.loop_start()  # Start background thread for MQTT
    print(f"[INFO] Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")

    # Publish online status
    client.publish(
        f"edge/{device_id}/status",
        json.dumps({"device_id": device_id, "status": "online"}),
        qos=1,
        retain=True,
    )

    # Find images to process
    images = glob(os.path.join(image_dir, "*.jpg")) + \
             glob(os.path.join(image_dir, "*.png")) + \
             glob(os.path.join(image_dir, "*.jpeg"))

    if not images:
        # If no sample images, use YOLO's built-in test images
        print("[INFO] No sample images found, using YOLO bus.jpg for demo")
        from ultralytics.utils import ASSETS
        images = [str(ASSETS / "bus.jpg")]

    print(f"[INFO] Processing {len(images)} images in rotation")
    cycle = 0

    while not shutdown_requested:
        cycle += 1
        image_path = images[cycle % len(images)]

        # Run inference
        start = time.time()
        detections = model.detect(image_path, confidence_threshold=CONFIDENCE_THRESHOLD)
        inference_ms = (time.time() - start) * 1000

        # Build MQTT payload
        payload = {
            "device_id": device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image": os.path.basename(image_path),
            "inference_ms": round(inference_ms, 1),
            "detection_count": len(detections),
            "detections": [
                {
                    "class": d.class_name,
                    "confidence": d.confidence,
                    "bbox": list(d.bbox),
                }
                for d in detections
            ],
        }

        # Publish to MQTT
        topic = f"edge/{device_id}/detections"
        client.publish(topic, json.dumps(payload), qos=1)

        print(
            f"[{device_id}] Cycle {cycle}: "
            f"{len(detections)} objects detected in {inference_ms:.0f}ms "
            f"({os.path.basename(image_path)})"
        )

        # Also publish a heartbeat
        heartbeat = {
            "device_id": device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inference_ms": round(inference_ms, 1),
            "model": "yolov8n",
            "cycle": cycle,
        }
        client.publish(f"edge/{device_id}/heartbeat", json.dumps(heartbeat), qos=0)

        # Wait before next detection
        for _ in range(DETECTION_INTERVAL):
            if shutdown_requested:
                break
            time.sleep(1)

    # Clean disconnect
    client.publish(
        f"edge/{device_id}/status",
        json.dumps({"device_id": device_id, "status": "offline"}),
        qos=1,
        retain=True,
    )
    client.loop_stop()
    client.disconnect()
    print(f"[INFO] Edge device '{device_id}' stopped (processed {cycle} cycles)")


if __name__ == "__main__":
    main()
