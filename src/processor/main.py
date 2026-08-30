"""
Processor service — subscribes to MQTT detections, stores in PostgreSQL, triggers alerts.
"""

import json
import signal
import sys
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
import psycopg2
from psycopg2.extras import execute_values

from src.config import MQTT_BROKER, MQTT_PORT, DATABASE_URL
from src.processor.alerts import check_alerts

shutdown_requested = False


def handle_signal(signum, frame):
    global shutdown_requested
    shutdown_requested = True


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def init_db(conn):
    """Create tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id BIGSERIAL PRIMARY KEY,
                device_id TEXT NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox_x1 INT, bbox_y1 INT, bbox_x2 INT, bbox_y2 INT,
                image TEXT,
                inference_ms REAL,
                detected_at TIMESTAMPTZ NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_detections_device ON detections(device_id);
            CREATE INDEX IF NOT EXISTS idx_detections_time ON detections(detected_at);
            CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name);

            CREATE TABLE IF NOT EXISTS alerts (
                id BIGSERIAL PRIMARY KEY,
                rule_name TEXT NOT NULL,
                device_id TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                triggered_at TIMESTAMPTZ NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(triggered_at);

            CREATE TABLE IF NOT EXISTS heartbeats (
                id BIGSERIAL PRIMARY KEY,
                device_id TEXT NOT NULL,
                inference_ms REAL,
                model TEXT,
                received_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_heartbeats_device ON heartbeats(device_id);
        """)
        conn.commit()


def store_detections(conn, device_id, timestamp, image, inference_ms, detections):
    """Bulk insert detections into PostgreSQL."""
    if not detections:
        return

    rows = [
        (
            device_id,
            d["class"],
            d["confidence"],
            d["bbox"][0] if d.get("bbox") else None,
            d["bbox"][1] if d.get("bbox") else None,
            d["bbox"][2] if d.get("bbox") else None,
            d["bbox"][3] if d.get("bbox") else None,
            image,
            inference_ms,
            timestamp,
        )
        for d in detections
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            """INSERT INTO detections
               (device_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                image, inference_ms, detected_at)
               VALUES %s""",
            rows,
        )
    conn.commit()


def store_alerts(conn, alerts):
    """Store triggered alerts."""
    if not alerts:
        return

    rows = [(a.rule_name, a.device_id, a.message, a.severity, a.triggered_at) for a in alerts]

    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO alerts (rule_name, device_id, message, severity, triggered_at) VALUES %s",
            rows,
        )
    conn.commit()


def store_heartbeat(conn, device_id, inference_ms, model_name):
    """Store a device heartbeat."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO heartbeats (device_id, inference_ms, model) VALUES (%s, %s, %s)",
            (device_id, inference_ms, model_name),
        )
    conn.commit()


def main():
    print("[INFO] EdgeEye Processor starting...")

    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(DATABASE_URL)
        init_db(conn)
        print("[INFO] Connected to PostgreSQL")
    except Exception as e:
        print(f"[FATAL] Database connection failed: {e}")
        sys.exit(1)

    total_detections = 0
    total_alerts = 0

    # MQTT message handlers
    def on_connect(client, userdata, flags, rc, properties=None):
        print(f"[INFO] Connected to MQTT broker (rc={rc})")
        # Subscribe to all edge device topics
        client.subscribe("edge/+/detections", qos=1)
        client.subscribe("edge/+/heartbeat", qos=0)
        print("[INFO] Subscribed to edge/+/detections and edge/+/heartbeat")

    def on_message(client, userdata, msg):
        nonlocal total_detections, total_alerts

        try:
            payload = json.loads(msg.payload.decode())
            topic_parts = msg.topic.split("/")

            if "detections" in msg.topic:
                device_id = payload.get("device_id", topic_parts[1])
                timestamp = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
                detections = payload.get("detections", [])

                # Store detections
                store_detections(
                    conn, device_id, timestamp,
                    payload.get("image", ""),
                    payload.get("inference_ms", 0),
                    detections,
                )
                total_detections += len(detections)

                # Check alert rules
                alerts = check_alerts(device_id, detections)
                if alerts:
                    store_alerts(conn, alerts)
                    total_alerts += len(alerts)
                    for a in alerts:
                        print(f"[ALERT] [{a.severity.upper()}] {a.device_id}: {a.message}")

                print(
                    f"[INFO] {device_id}: {len(detections)} detections stored "
                    f"(total: {total_detections}, alerts: {total_alerts})"
                )

            elif "heartbeat" in msg.topic:
                device_id = payload.get("device_id", topic_parts[1])
                store_heartbeat(
                    conn,
                    device_id,
                    payload.get("inference_ms", 0),
                    payload.get("model", "unknown"),
                )

        except Exception as e:
            print(f"[ERROR] Failed to process message: {e}")

    # Connect to MQTT
    client = mqtt.Client(client_id="processor", protocol=mqtt.MQTTv311, callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except Exception as e:
        print(f"[FATAL] MQTT connection failed: {e}")
        sys.exit(1)

    print("[INFO] Processor ready — waiting for edge device messages...")

    # Run until shutdown
    while not shutdown_requested:
        client.loop(timeout=1.0)

    client.disconnect()
    conn.close()
    print(f"[INFO] Processor stopped (total: {total_detections} detections, {total_alerts} alerts)")


if __name__ == "__main__":
    main()
