"""EdgeEye Dashboard — real-time edge AI monitoring."""

from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2
from psycopg2.extras import RealDictCursor

from src.config import DATABASE_URL

st.set_page_config(page_title="EdgeEye — Edge AI Monitor", page_icon="👁️", layout="wide")


def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


st.title("👁️ EdgeEye — Real-Time Edge AI Monitor")
st.caption("Live object detection from edge devices")

col_refresh, col_hours = st.columns([1, 1])
with col_refresh:
    auto = st.toggle("Auto-refresh (10s)", value=False)
with col_hours:
    hours = st.slider("Time window", 1, 48, 6)

if auto:
    import time; time.sleep(10); st.rerun()

try:
    conn = get_conn()
    cur = conn.cursor()

    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Key metrics
    cur.execute("SELECT COUNT(*) as c FROM detections WHERE detected_at >= %s", (since,))
    total_dets = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(DISTINCT device_id) as c FROM heartbeats WHERE received_at >= %s", (since,))
    active_devices = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) as c FROM alerts WHERE triggered_at >= %s", (since,))
    total_alerts = cur.fetchone()["c"]

    cur.execute("SELECT AVG(inference_ms) as avg_ms FROM heartbeats WHERE received_at >= %s", (since,))
    avg_ms = cur.fetchone()["avg_ms"] or 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detections", f"{total_dets:,}")
    c2.metric("Active Devices", active_devices)
    c3.metric("Alerts", total_alerts)
    c4.metric("Avg Inference", f"{avg_ms:.0f}ms")

    st.divider()

    if total_dets == 0:
        st.info("🔄 No detections yet. Start the edge simulators and processor.")
    else:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Object Distribution")
            cur.execute(
                """SELECT class_name, COUNT(*) as count FROM detections
                   WHERE detected_at >= %s GROUP BY class_name ORDER BY count DESC LIMIT 15""",
                (since,),
            )
            rows = cur.fetchall()
            if rows:
                df = pd.DataFrame(rows)
                fig = px.bar(df, x="class_name", y="count", color="count",
                             color_continuous_scale="Viridis")
                fig.update_layout(margin=dict(t=20, b=20), height=350, xaxis_title="",
                                  yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Detections Over Time")
            cur.execute(
                """SELECT date_trunc('hour', detected_at) as hour, COUNT(*) as count
                   FROM detections WHERE detected_at >= %s
                   GROUP BY hour ORDER BY hour""",
                (since,),
            )
            rows = cur.fetchall()
            if rows:
                df = pd.DataFrame(rows)
                fig = px.line(df, x="hour", y="count", markers=True)
                fig.update_layout(margin=dict(t=20, b=20), height=350, xaxis_title="",
                                  yaxis_title="Detections")
                st.plotly_chart(fig, use_container_width=True)

        # Device health
        st.subheader("Device Health")
        cur.execute(
            """SELECT DISTINCT ON (device_id) device_id, inference_ms, model, received_at
               FROM heartbeats ORDER BY device_id, received_at DESC"""
        )
        devices = cur.fetchall()
        if devices:
            df_dev = pd.DataFrame(devices)
            df_dev["status"] = df_dev["received_at"].apply(
                lambda t: "🟢 Online" if (datetime.now(timezone.utc) - t).seconds < 120 else "🔴 Offline"
            )
            st.dataframe(df_dev[["device_id", "status", "model", "inference_ms", "received_at"]],
                         use_container_width=True, hide_index=True)

        # Recent alerts
        st.subheader("Recent Alerts")
        cur.execute(
            """SELECT severity, device_id, message, triggered_at FROM alerts
               ORDER BY triggered_at DESC LIMIT 20"""
        )
        alerts = cur.fetchall()
        if alerts:
            df_alerts = pd.DataFrame(alerts)
            df_alerts["icon"] = df_alerts["severity"].map(
                {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
            )
            st.dataframe(df_alerts[["icon", "severity", "device_id", "message", "triggered_at"]],
                         use_container_width=True, hide_index=True)
        else:
            st.text("No alerts triggered yet.")

        # Recent detections
        st.subheader("Latest Detections")
        cur.execute(
            """SELECT device_id, class_name, confidence, inference_ms, image, detected_at
               FROM detections ORDER BY detected_at DESC LIMIT 30"""
        )
        recent = cur.fetchall()
        if recent:
            st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True, height=400)

    conn.close()

except Exception as e:
    st.error(f"Database connection error: {e}")

st.divider()
st.caption("Built with YOLOv8 · MQTT · PostgreSQL · Streamlit · Docker")
