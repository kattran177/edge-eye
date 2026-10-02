---
title: Edge Eye
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.41.1"
app_file: app.py
pinned: false
---

# 👁️ EdgeEye — Real-Time Edge AI Object Detection Platform

A real-time edge computing system that runs lightweight ML models on simulated IoT devices, streams detections over MQTT, and visualises results on a live dashboard. Built with YOLOv8, MQTT (Mosquitto), PostgreSQL, and Streamlit.

**Why this exists:** Edge AI is one of the fastest-growing fields in tech — autonomous vehicles, smart factories, drones, AR/VR. This project demonstrates deploying ML models to constrained devices, IoT messaging protocols, and real-time monitoring dashboards.

---

## 🏗️ Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  Edge Device 1  │──MQTT──▶│                 │──MQTT──▶│   Processor     │
│  (YOLO Nano)    │         │   Mosquitto     │         │   Service       │
├─────────────────┤         │   MQTT Broker   │         │                 │
│  Edge Device 2  │──MQTT──▶│                 │         │  • Stores in DB │
│  (YOLO Nano)    │         │  Lightweight    │         │  • Triggers     │
├─────────────────┤         │  pub/sub broker │         │    alerts       │
│  Edge Device N  │──MQTT──▶│                 │         │  • Aggregates   │
└─────────────────┘         └─────────────────┘         └────────┬────────┘
                                                                 │
                                                          ┌──────▼───────┐
                                                          │  PostgreSQL  │
                                                          └──────┬───────┘
                                                                 │
                                                          ┌──────▼───────┐
                                                          │  Streamlit   │
                                                          │  Dashboard   │
                                                          │              │
                                                          │ • Live feed  │
                                                          │ • Device map │
                                                          │ • Alerts     │
                                                          └──────────────┘
```

**Data flow:**
1. **Edge simulators** run YOLOv8 Nano on sample images every few seconds
2. Detection results are published to **MQTT topics** (`edge/{device_id}/detections`)
3. **Mosquitto broker** routes messages to subscribers
4. **Processor** consumes detections, stores in PostgreSQL, and checks alert rules
5. **Dashboard** queries the database and displays live metrics

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| ML Model | YOLOv8 Nano (Ultralytics) | 6MB model, real-time object detection on CPU |
| IoT Protocol | MQTT (Mosquitto) | Industry-standard lightweight pub/sub messaging |
| Processing | Python | Detection storage, alerting, aggregation |
| Database | PostgreSQL | Time-series detection storage |
| Dashboard | Streamlit + Plotly | Real-time monitoring and device management |
| Containers | Docker Compose | One-command deployment |
| CI/CD | GitHub Actions | Automated testing |

---

## 🚀 Quick Start

### Run everything

```bash
docker compose up --build
```

This starts:
- Mosquitto MQTT broker (port 1883)
- PostgreSQL (port 5432)
- 2 edge device simulators
- Processor service
- Dashboard (http://localhost:8501)

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start infrastructure
docker compose up mosquitto postgres -d

# Run services
python -m src.edge.simulator --device-id cam-01    # Terminal 1
python -m src.processor.main                        # Terminal 2
streamlit run src/dashboard/app.py                   # Terminal 3
```

---

## 📊 Dashboard Features

- **Live Detection Feed** — latest objects detected across all devices
- **Device Health** — which devices are online, last heartbeat, detection rate
- **Detection Trends** — objects detected over time (line chart)
- **Object Breakdown** — what types of objects are being detected (bar chart)
- **Alert Log** — triggered alerts (e.g., "person detected in restricted zone")

---

## 📁 Project Structure

```
edge-eye/
├── src/
│   ├── edge/               # Edge device simulator
│   │   ├── simulator.py    # Runs YOLO model, publishes to MQTT
│   │   └── model.py        # YOLOv8 model wrapper
│   ├── processor/          # Central processing service
│   │   ├── main.py         # MQTT subscriber + DB writer
│   │   └── alerts.py       # Alert rule engine
│   └── dashboard/          # Monitoring dashboard
│       └── app.py          # Streamlit application
├── infra/docker/           # Dockerfiles
├── sample_images/          # Test images for the simulator
├── tests/                  # Test suite
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🔑 Key Concepts

- **Edge AI**: Running ML inference on devices close to the data source, not in the cloud
- **MQTT Protocol**: Lightweight publish/subscribe messaging designed for IoT (used by AWS IoT, Azure IoT Hub, home automation)
- **YOLOv8 Nano**: Smallest YOLO variant — proves you can deploy real ML on constrained hardware
- **Device Fleet Management**: Monitoring multiple devices from a central dashboard
- **Time-Series Data**: Storing and querying detection events over time

---

## 📈 What I Learned

- How to deploy ML models outside of notebooks for real-time inference
- MQTT pub/sub patterns and topic design for IoT systems
- The tradeoffs of edge vs cloud inference (latency, bandwidth, privacy)
- Building monitoring dashboards for device fleets
- Simulating IoT devices for development and testing

---

## License

MIT
