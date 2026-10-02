"""EdgeEye Dashboard — Hugging Face Spaces demo with simulated data."""

import random
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EdgeEye — Edge AI Monitor", page_icon="👁️", layout="wide")

# ---------------------------------------------------------------------------
# Simulated data generation
# ---------------------------------------------------------------------------

DEVICE_IDS = ["cam-01", "cam-02", "cam-03"]
MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8n.pt"]
CLASSES = [
    "person", "car", "truck", "bicycle", "dog",
    "cat", "chair", "bottle", "laptop", "phone",
]
SEVERITIES = ["info", "warning", "critical"]
ALERT_MESSAGES = [
    "Person detected in restricted zone",
    "High detection rate — possible crowd",
    "Device inference latency spike",
    "Unknown object class detected",
    "Low confidence detections — check lighting",
]

SEVERITY_ICONS = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}


def generate_detections(hours: int, n: int = 300) -> pd.DataFrame:
    """Generate fake detection records over the given time window."""
    now = datetime.now(timezone.utc)
    records = []
    for _ in range(n):
        offset = random.uniform(0, hours * 3600)
        records.append({
            "device_id": random.choice(DEVICE_IDS),
            "class_name": random.choice(CLASSES),
            "confidence": round(random.uniform(0.5, 0.99), 2),
            "inference_ms": random.randint(8, 45),
            "detected_at": now - timedelta(seconds=offset),
        })
    return pd.DataFrame(records).sort_values("detected_at", ascending=False)


def generate_devices() -> pd.DataFrame:
    """Generate fake device health records."""
    now = datetime.now(timezone.utc)
    records = []
    for device_id, model in zip(DEVICE_IDS, MODELS):
        last_seen_seconds_ago = random.randint(5, 200)
        received_at = now - timedelta(seconds=last_seen_seconds_ago)
        records.append({
            "device_id": device_id,
            "model": model,
            "inference_ms": random.randint(8, 45),
            "received_at": received_at,
            "status": "🟢 Online" if last_seen_seconds_ago < 120 else "🔴 Offline",
        })
    return pd.DataFrame(records)


def generate_alerts(hours: int, n: int = 15) -> pd.DataFrame:
    """Generate fake alert records."""
    now = datetime.now(timezone.utc)
    records = []
    for _ in range(n):
        severity = random.choice(SEVERITIES)
        offset = random.uniform(0, hours * 3600)
        records.append({
            "severity": severity,
            "icon": SEVERITY_ICONS[severity],
            "device_id": random.choice(DEVICE_IDS),
            "message": random.choice(ALERT_MESSAGES),
            "triggered_at": now - timedelta(seconds=offset),
        })
    return pd.DataFrame(records).sort_values("triggered_at", ascending=False)


# ---------------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------------

st.title("👁️ EdgeEye — Real-Time Edge AI Monitor")
st.caption("Live object detection from edge devices · **Demo mode** — simulated data")

st.info(
    "🛠️ This is a Hugging Face Spaces demo using simulated data. "
    "The full system uses YOLOv8 + MQTT + PostgreSQL + Docker. "
    "See the [GitHub repo](https://github.com/kattran177/edge-eye) for the complete setup.",
    icon="ℹ️",
)

col_refresh, col_hours = st.columns([1, 1])
with col_refresh:
    auto = st.toggle("Auto-refresh (10s)", value=False)
with col_hours:
    hours = st.slider("Time window (hours)", 1, 48, 6)

if auto:
    time.sleep(10)
    st.rerun()

# Generate data scoped to selected time window
detections_df = generate_detections(hours)
devices_df = generate_devices()
alerts_df = generate_alerts(hours)

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------

total_dets = len(detections_df)
active_devices = devices_df[devices_df["status"] == "🟢 Online"].shape[0]
total_alerts = len(alerts_df)
avg_ms = detections_df["inference_ms"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Detections", f"{total_dets:,}")
c2.metric("Active Devices", f"{active_devices}/{len(DEVICE_IDS)}")
c3.metric("Alerts", total_alerts)
c4.metric("Avg Inference", f"{avg_ms:.0f}ms")

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Object Distribution")
    class_counts = (
        detections_df.groupby("class_name")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    fig = px.bar(
        class_counts, x="class_name", y="count",
        color="count", color_continuous_scale="Viridis",
    )
    fig.update_layout(margin=dict(t=20, b=20), height=350,
                      xaxis_title="", yaxis_title="Count", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Detections Over Time")
    detections_df["hour"] = detections_df["detected_at"].dt.floor("h")
    hourly = (
        detections_df.groupby("hour")
        .size()
        .reset_index(name="count")
        .sort_values("hour")
    )
    fig2 = px.line(hourly, x="hour", y="count", markers=True)
    fig2.update_layout(margin=dict(t=20, b=20), height=350,
                       xaxis_title="", yaxis_title="Detections")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Detection breakdown by device
# ---------------------------------------------------------------------------

st.subheader("Detections by Device")
device_counts = (
    detections_df.groupby("device_id")
    .size()
    .reset_index(name="count")
)
fig3 = px.pie(device_counts, names="device_id", values="count",
              color_discrete_sequence=px.colors.qualitative.Set2)
fig3.update_layout(margin=dict(t=20, b=20), height=300)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Device health
# ---------------------------------------------------------------------------

st.subheader("Device Health")
st.dataframe(
    devices_df[["device_id", "status", "model", "inference_ms", "received_at"]],
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Recent alerts
# ---------------------------------------------------------------------------

st.subheader("Recent Alerts")
st.dataframe(
    alerts_df[["icon", "severity", "device_id", "message", "triggered_at"]].head(20),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Latest detections
# ---------------------------------------------------------------------------

st.subheader("Latest Detections")
st.dataframe(
    detections_df[["device_id", "class_name", "confidence", "inference_ms", "detected_at"]].head(30),
    use_container_width=True,
    hide_index=True,
    height=400,
)

st.divider()
st.caption("Built with YOLOv8 · MQTT · PostgreSQL · Streamlit · Docker")
