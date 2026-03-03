# 🛡️ ITERA Smart Sentinel

**Sistem Deteksi & Analitik Pelanggaran Helm Real-time**

Real-time helmet violation detection and analytics system for motorcycle riders, combining YOLOv8 object detection, ByteTrack multi-object tracking, and Apache Kafka/Spark streaming.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Workflow](#pipeline-workflow)
- [Database Schema](#database-schema)
- [Grafana Dashboard](#grafana-dashboard)
- [Testing](#testing)
- [Documentation](#documentation)

## Overview

ITERA Smart Sentinel detects motorcycle riders without helmets from CCTV/RTSP camera feeds in real time. The system processes video frames through a YOLOv8 detection model, tracks individuals across frames with ByteTrack, confirms violations using temporal consistency logic, streams events through Apache Kafka, and stores results in PostgreSQL for visualization in Grafana dashboards.

The model classifies three object types:

| Class ID | Class Name          | Description                          |
|----------|---------------------|--------------------------------------|
| 0        | `DRIVER_HELMET`     | Motorcycle rider wearing a helmet    |
| 1        | `DRIVER_NO_HELMET`  | Motorcycle rider without a helmet    |
| 2        | `MOTORCYCLE`        | Motorcycle (vehicle)                 |

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Video Source    │────▶│  YOLOv8      │────▶│  ByteTrack  │
│  (RTSP / File)  │     │  Detector    │     │  Tracker    │
└─────────────────┘     └──────────────┘     └──────┬──────┘
                                                    │
                                                    ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  PostgreSQL     │◀────│  Spark       │◀────│  Kafka      │
│  (Supabase)     │     │  Streaming   │     │  Producer   │
└────────┬────────┘     └──────────────┘     └──────┬──────┘
         │                                          │
         ▼                                          │
┌─────────────────┐                          ┌─────────────┐
│  Grafana        │                          │  Violation  │
│  Dashboard      │                          │  Detector   │
└─────────────────┘                          └─────────────┘
```

**Data flow:**

1. **Video Capture** — Reads frames from an RTSP stream or local video file
2. **YOLOv8 Inference** — Detects helmets, no-helmets, and motorcycles per frame
3. **ByteTrack Tracking** — Assigns persistent track IDs to detections across frames
4. **Violation Detection** — Confirms violations after N consecutive frames of `DRIVER_NO_HELMET` (temporal filtering)
5. **Kafka Streaming** — Sends confirmed violation events to a Kafka topic
6. **Spark Processing** — Consumes from Kafka, deduplicates, enriches with time dimensions
7. **PostgreSQL Storage** — Writes to a Star Schema (fact + dimension tables)
8. **Grafana Visualization** — Displays real-time dashboards with KPIs, heatmaps, and trends

## Features

- **Real-time YOLOv8 detection** — 3-class helmet/no-helmet/motorcycle detection
- **Multi-object tracking** — ByteTrack maintains unique IDs across frames, handles occlusion
- **Temporal violation confirmation** — N-frame consecutive detection rule reduces false positives
- **Per-track deduplication** — Each track ID emits at most one violation per session
- **Kafka event streaming** — High-throughput message delivery (3,700+ messages/sec)
- **Spark Structured Streaming** — Micro-batch processing with watermark-based deduplication
- **Star Schema analytics** — Denormalized time dimensions for fast Grafana queries
- **RTSP reconnection** — Automatic reconnection on stream disconnection
- **Configurable pipeline** — All parameters tunable via environment variables
- **Optional display mode** — Annotated video output with bounding boxes and FPS overlay

## Tech Stack

| Component            | Technology                                  |
|----------------------|---------------------------------------------|
| Object Detection     | YOLOv8 (Ultralytics)                        |
| Multi-Object Tracking| ByteTrack (via supervision library)          |
| Video Processing     | OpenCV                                      |
| Message Broker       | Apache Kafka (confluent-kafka)               |
| Stream Processing    | Apache Spark Structured Streaming            |
| Database             | PostgreSQL (Supabase)                        |
| Dashboard            | Grafana                                      |
| Configuration        | Pydantic Settings                            |
| Logging              | Loguru                                       |
| Package Manager      | uv                                           |

## Project Structure

```
├── config/
│   └── settings.py               # Centralized configuration (Pydantic BaseSettings)
├── docs/
│   ├── bab4_draft.tex            # Technical documentation (LaTeX)
│   └── WORKFLOW.md               # Detailed pipeline workflow documentation
├── grafana/
│   ├── dashboards/               # Grafana dashboard JSON definitions
│   └── provisioning/             # Grafana provisioning configuration
├── models/                       # YOLOv8 model weights (best.pt, gitignored)
├── scripts/
│   ├── run_pipeline.sh           # Full system startup script
│   └── setup_kafka_topics.sh     # Kafka topic initialization
├── spark/
│   └── stream_processor.py       # Spark Structured Streaming consumer
├── sql/
│   ├── 001_create_dimensions.sql # Dimension tables (camera, violation type)
│   ├── 002_create_facts.sql      # Fact table with indexes
│   └── 003_create_views.sql      # Analytical views for Grafana
├── src/
│   ├── detector/
│   │   ├── model.py              # YOLOv8 helmet detector wrapper
│   │   ├── tracker.py            # ByteTrack multi-object tracker
│   │   └── violation.py          # Violation detection logic
│   ├── pipeline/
│   │   └── main.py               # Main pipeline orchestrator
│   ├── streaming/
│   │   ├── producer.py           # Kafka producer for violation events
│   │   └── consumer.py           # Kafka consumer (debugging/testing)
│   └── utils/
│       └── schemas.py            # Pydantic data models
├── tests/
│   ├── test_detector.py          # Unit tests
│   ├── benchmark_pipeline.py     # Performance benchmarking
│   └── resource_benchmark.py     # Resource usage analysis
├── .env.example                  # Environment variable template
├── pyproject.toml                # Project metadata and dependencies
└── uv.lock                       # Dependency lock file
```

## Prerequisites

- **Python** ≥ 3.11
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux) — for infrastructure services
- **NVIDIA GPU** with CUDA ≥ 7.0 (optional, falls back to CPU)

## Installation (Windows — Docker Compose)

> **Recommended for Windows users.** Infrastructure services (Kafka, Spark, PostgreSQL, Grafana) run in Docker. Python pipeline (YOLOv8) runs natively via `uv`.

### 1. Install prerequisites

- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Clone and install

```powershell
git clone https://github.com/EgiStr/sistem-cerdas-yolo-bytrack.git
cd sistem-cerdas-yolo-bytrack
uv sync
```

### 3. Configure environment

```powershell
Copy-Item .env.example .env
# Edit .env with your settings (video source, credentials)
```

### 4. Start all infrastructure

```powershell
# Start all Docker services (Kafka, Spark, PostgreSQL, Grafana)
docker compose up -d

# OR use the all-in-one script:
.\scripts\run_pipeline.ps1
```

### 5. Setup database schema

```powershell
.\scripts\init-db.ps1
```

### 6. Setup Kafka topics

```powershell
.\scripts\setup_kafka_topics.ps1
```

### 7. Place the YOLOv8 model

Place your trained `best.pt` model file in the `models/` directory.

### 8. Run the system

```powershell
# Terminal 1 — Start Spark stream processor
.\scripts\spark_submit.ps1

# Terminal 2 — Start detection pipeline
uv run python -m src.pipeline.main --source sample_video.mp4

# With video display
uv run python -m src.pipeline.main --source sample_video.mp4 --display

# Without Kafka (testing only)
uv run python -m src.pipeline.main --source sample_video.mp4 --no-kafka --display
```

### Web UIs

| Service | URL | Credentials |
|---|---|---|
| Grafana Dashboard | http://localhost:3000 | admin / admin |
| Spark Master UI | http://localhost:8080 | — |
| Spark Worker UI | http://localhost:8081 | — |

### Docker management

```powershell
docker compose ps          # Check service status
docker compose logs -f     # Follow all logs
docker compose down        # Stop services
docker compose down -v     # Stop + remove data volumes (reset)
```

## Installation (Linux — Native)

> Original setup for native Linux installations.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/EgiStr/sistem-cerdas-yolo-bytrack.git
   cd sistem-cerdas-yolo-bytrack
   ```

2. **Install Python dependencies:**

   ```bash
   uv sync
   ```

3. **Configure environment variables:**

   ```bash
   cp .env.example .env
   # Edit .env with your settings (video source, Kafka, database credentials)
   ```

4. **Place the YOLOv8 model:**

   Place your trained `best.pt` model file in the `models/` directory.

5. **Set up the database:**

   Run the SQL scripts in order against your PostgreSQL instance:

   ```bash
   psql -h <host> -U <user> -d <database> -f sql/001_create_dimensions.sql
   psql -h <host> -U <user> -d <database> -f sql/002_create_facts.sql
   psql -h <host> -U <user> -d <database> -f sql/003_create_views.sql
   ```

6. **Set up Kafka topics:**

   ```bash
   bash scripts/setup_kafka_topics.sh
   ```

## Configuration

All settings are managed via environment variables loaded from a `.env` file. See [`.env.example`](.env.example) for all available options.

Key settings:

| Variable                    | Default            | Description                             |
|-----------------------------|--------------------|-----------------------------------------|
| `VIDEO_SOURCE`              | `sample_video.mp4` | RTSP URL or local video file path       |
| `MODEL_PATH`                | `models/best.pt`   | Path to YOLOv8 model weights            |
| `MODEL_CONFIDENCE`          | `0.5`              | Detection confidence threshold           |
| `MODEL_DEVICE`              | `0`                | CUDA device index or `cpu`               |
| `VIOLATION_CONFIRM_FRAMES`  | `3`                | Consecutive frames required to confirm   |
| `KAFKA_BOOTSTRAP_SERVERS`   | `localhost:9092`   | Kafka broker address                     |
| `KAFKA_TOPIC_VIOLATIONS`    | `video.violations` | Kafka topic for violation events         |

## Usage

### Run the detection pipeline

```bash
# Basic usage (with Kafka)
uv run python -m src.pipeline.main --source sample_video.mp4

# With video display window
uv run python -m src.pipeline.main --source sample_video.mp4 --display

# Without Kafka (testing/logging only)
uv run python -m src.pipeline.main --source sample_video.mp4 --no-kafka --display

# Save annotated output to file
uv run python -m src.pipeline.main --source sample_video.mp4 --save-output output.mp4

# RTSP camera stream
uv run python -m src.pipeline.main --source rtsp://admin:password@192.168.1.100:554/stream1
```

### Run Spark stream processor

```bash
/opt/spark/bin/spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8,org.postgresql:postgresql:42.7.1 \
    spark/stream_processor.py
```

### Run the full system

Use the provided startup script:

```bash
bash scripts/run_pipeline.sh
```

This checks prerequisites (Kafka, `.env`), creates Kafka topics, and prints instructions for starting each component.

### Debug Kafka messages

```bash
uv run python -m src.streaming.consumer
```

## Pipeline Workflow

For a detailed description of the pipeline workflow, data flow, and each component, see [docs/WORKFLOW.md](docs/WORKFLOW.md).

### Summary

The pipeline follows a **Detect → Track → Confirm → Stream → Store → Visualize** pattern:

1. **HelmetDetector** (`src/detector/model.py`) — Runs YOLOv8 inference on each frame, returns `supervision.Detections`
2. **ObjectTracker** (`src/detector/tracker.py`) — Updates ByteTrack with detections, assigns persistent `tracker_id` values
3. **ViolationDetector** (`src/detector/violation.py`) — Checks for `DRIVER_NO_HELMET` detections sustained over N consecutive frames, deduplicates by `track_id`
4. **ViolationProducer** (`src/streaming/producer.py`) — Serializes `ViolationEvent` to JSON and publishes to Kafka with snappy compression
5. **Spark StreamProcessor** (`spark/stream_processor.py`) — Consumes from Kafka, deduplicates within a 30-second watermark window, enriches with time dimensions, writes to PostgreSQL
6. **Grafana Dashboard** — Queries PostgreSQL views for real-time KPIs, heatmaps, trends, and violation logs

## Database Schema

The system uses a **Star Schema** design for analytical queries:

**Dimension tables:**
- `dim_camera` — Camera metadata (ID, name, location, gate)
- `dim_violation_type` — Violation type definitions

**Fact table:**
- `fact_violations` — Violation events with embedded time dimensions (hour, day_of_week, time_period) for fast aggregation

**Analytical views:**
- `vw_hourly_violations` — Hourly violation counts by camera (heatmap)
- `vw_daily_trend` — Daily violation trends (time series)
- `vw_camera_stats` — Per-camera statistics (bar chart)
- `vw_peak_hours` — Peak hours analysis (hour × day_of_week heatmap)
- `vw_today_summary` — Today's KPI summary
- `vw_recent_violations` — Latest 100 violations (detail table)

## Grafana Dashboard

The Grafana Safety Dashboard provides six main panels:

1. **KPI Overview** — Total violations today, average confidence, average processing latency
2. **Temporal Trend** — Time-series chart of hourly violations over 24 hours
3. **Activity Heatmap** — Hour × Day matrix showing violation hotspots
4. **Location Distribution** — Bar chart comparing violations across cameras/gates
5. **Time Period Distribution** — Pie chart of violations by period (Pagi/Siang/Sore/Malam)
6. **Violation Log** — Table of recent violations with confidence and track details

Dashboard definitions are stored in `grafana/dashboards/` and can be provisioned automatically via `grafana/provisioning/`.

## Testing

```bash
# Run unit tests
uv run pytest tests/

# Run benchmarks
uv run python tests/benchmark_pipeline.py
uv run python tests/resource_benchmark.py
```

## Documentation

- [Pipeline Workflow Details](docs/WORKFLOW.md) — Complete data flow and component documentation
- [Technical Report (LaTeX)](docs/bab4_draft.tex) — Experimental results and analysis
- [Environment Configuration](.env.example) — All available configuration variables
