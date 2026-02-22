# Pipeline Workflow Documentation

This document provides a detailed description of the ITERA Smart Sentinel pipeline workflow, covering each component, data flow, and processing stage.

## Table of Contents

- [System Overview](#system-overview)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1: Video Capture](#stage-1-video-capture)
  - [Stage 2: Object Detection (YOLOv8)](#stage-2-object-detection-yolov8)
  - [Stage 3: Multi-Object Tracking (ByteTrack)](#stage-3-multi-object-tracking-bytetrack)
  - [Stage 4: Violation Detection](#stage-4-violation-detection)
  - [Stage 5: Event Streaming (Kafka)](#stage-5-event-streaming-kafka)
  - [Stage 6: Stream Processing (Spark)](#stage-6-stream-processing-spark)
  - [Stage 7: Data Storage (PostgreSQL)](#stage-7-data-storage-postgresql)
  - [Stage 8: Visualization (Grafana)](#stage-8-visualization-grafana)
- [Data Schemas](#data-schemas)
- [Configuration Reference](#configuration-reference)
- [Performance Characteristics](#performance-characteristics)

## System Overview

The ITERA Smart Sentinel system follows a **Detect → Track → Confirm → Stream → Store → Visualize** processing pattern. The pipeline is split into two independently running processes:

1. **Detection Pipeline** (`src/pipeline/main.py`) — Video processing, detection, tracking, and Kafka publishing
2. **Stream Processor** (`spark/stream_processor.py`) — Kafka consumption, deduplication, enrichment, and PostgreSQL storage

This decoupled architecture enables independent scaling of the detection and analytics components.

```
Detection Pipeline (Python/uv)          Stream Processor (Spark)
┌────────────────────────────┐          ┌────────────────────────────┐
│ Video → Detect → Track     │          │ Kafka → Deduplicate        │
│   → Violations → Kafka     │ ───────▶ │   → Enrich → PostgreSQL    │
└────────────────────────────┘          └────────────────────────────┘
                                                     │
                                                     ▼
                                        ┌────────────────────────────┐
                                        │ Grafana Dashboard          │
                                        └────────────────────────────┘
```

## Pipeline Stages

### Stage 1: Video Capture

**Module:** `src/pipeline/main.py` → `SentinelPipeline._open_video()`

The pipeline reads frames from either a local video file or an RTSP camera stream using OpenCV.

- **Input:** Video source path or RTSP URL (configured via `VIDEO_SOURCE`)
- **Output:** BGR image frames as NumPy arrays (`H × W × 3`)
- **RTSP reconnection:** If the RTSP stream disconnects, the pipeline waits 2 seconds and attempts to reconnect automatically
- **Frame metadata:** Resolution, FPS, and total frame count are logged on startup

### Stage 2: Object Detection (YOLOv8)

**Module:** `src/detector/model.py` → `HelmetDetector`

Each frame is passed through a YOLOv8 model trained on three classes:

| Class ID | Class Name          | Description                         |
|----------|---------------------|-------------------------------------|
| 0        | `DRIVER_HELMET`     | Motorcycle rider wearing a helmet   |
| 1        | `DRIVER_NO_HELMET`  | Motorcycle rider without a helmet   |
| 2        | `MOTORCYCLE`        | Motorcycle vehicle                  |

**Processing steps:**

1. The model is loaded from `models/best.pt` and warmed up with a dummy inference
2. If the GPU is incompatible (CUDA Compute Capability < 7.0), inference falls back to CPU automatically
3. Each frame is processed with configurable confidence and IoU thresholds
4. Results are converted to `supervision.Detections` objects with class names in metadata

**Key parameters:**
- `MODEL_CONFIDENCE` (default: 0.5) — Minimum detection confidence
- `MODEL_IOU_THRESHOLD` (default: 0.45) — NMS IoU threshold
- `MODEL_DEVICE` (default: `0`) — CUDA device or `cpu`
- `MODEL_IMGSZ` (default: 640) — Input image size

### Stage 3: Multi-Object Tracking (ByteTrack)

**Module:** `src/detector/tracker.py` → `ObjectTracker`

ByteTrack assigns persistent `tracker_id` values to detections across consecutive frames. This enables the violation detector to track individual riders over time.

**How ByteTrack works:**

1. **High-confidence matching:** Detections above the activation threshold are matched to existing tracks using IoU
2. **Low-confidence recovery:** Remaining unmatched detections are matched against recently lost tracks
3. **Track lifecycle:** New tracks are created for unmatched detections; lost tracks are kept alive for `lost_track_buffer` frames before removal

**Key parameters:**
- `track_activation_threshold` (default: 0.25) — Minimum confidence for track activation
- `lost_track_buffer` (default: 30 frames) — Frames to keep lost tracks (~2 seconds at 15 FPS)
- `minimum_matching_threshold` (default: 0.8) — Minimum IoU for detection-track matching
- `frame_rate` (default: 15) — Expected video frame rate

### Stage 4: Violation Detection

**Module:** `src/detector/violation.py` → `ViolationDetector`

The violation detector uses temporal consistency logic to confirm helmet violations, reducing false positives from momentary misclassifications.

**Algorithm:**

```
For each tracked detection in the current frame:
  1. Filter for DRIVER_NO_HELMET detections
  2. Skip if this track_id already emitted a violation (deduplication)
  3. Increment the consecutive-frame counter for this track_id
  4. Accumulate the confidence score
  5. If counter ≥ N (confirm_frames):
     → Create a ViolationEvent with the average confidence
     → Mark the track_id as violated (no further events for this ID)
  6. For tracks NOT classified as DRIVER_NO_HELMET in this frame:
     → Reset their streak counter to 0 (decay)
```

**Key parameters:**
- `VIOLATION_CONFIRM_FRAMES` (default: 3) — Consecutive frames required for confirmation
- `ASSOCIATION_IOU_THRESHOLD` (default: 0.3) — IoU threshold for detection association

**Output:** `ViolationEvent` objects containing:
- `event_id` — Unique UUID
- `track_id` — ByteTrack track identifier
- `timestamp` — UTC timestamp
- `camera_id` — Source camera identifier
- `confidence` — Average confidence over the confirmation window
- `bbox` — Bounding box coordinates (x1, y1, x2, y2)
- `frame_number` — Frame number where violation was confirmed

### Stage 5: Event Streaming (Kafka)

**Module:** `src/streaming/producer.py` → `ViolationProducer`

Confirmed violation events are serialized to JSON and published to a Kafka topic.

**Producer configuration:**
- **Compression:** Snappy compression for bandwidth efficiency
- **Batching:** Messages are batched within 5ms (`linger.ms=5`) with up to 100 messages per batch
- **Reliability:** `acks=all` with 3 retries for guaranteed delivery
- **Partitioning:** Messages are keyed by `track_id` for ordered processing per track

**Kafka topics:**
- `video.detections` — All detection events (optional, high volume, 1-hour retention)
- `video.violations` — Confirmed violation events (consumed by Spark, 24-hour retention)

### Stage 6: Stream Processing (Spark)

**Module:** `spark/stream_processor.py`

Apache Spark Structured Streaming consumes violation events from Kafka, processes them in micro-batches, and writes to PostgreSQL.

**Processing steps:**

1. **Read from Kafka** — Subscribe to `video.violations` topic, parse JSON into structured schema
2. **Time enrichment** — Extract time dimensions from the event timestamp:
   - `hour`, `minute`, `day_of_week`, `date`, `week_of_year`, `month`, `year`
   - `time_period`: Mapped to Indonesian labels (`pagi`, `siang`, `sore`, `malam`)
3. **Deduplication** — Apply a 30-second watermark and drop duplicates by `(track_id, camera_id)`
4. **Write to PostgreSQL** — Insert into `dim_time` and `fact_violations` tables via JDBC

**Configuration:**
- **Trigger interval:** 5 seconds (micro-batch processing)
- **Watermark:** 30 seconds (handles late-arriving events)
- **Checkpoint:** `/tmp/sentinel-checkpoints` (for fault tolerance)

### Stage 7: Data Storage (PostgreSQL)

**SQL schemas:** `sql/001_create_dimensions.sql`, `sql/002_create_facts.sql`, `sql/003_create_views.sql`

The database uses a **Star Schema** design optimized for analytical queries:

**Dimension tables:**

- **`dim_camera`** — Camera metadata
  - `camera_id` (PK), `camera_name`, `location`, `gate_name`, `latitude`, `longitude`, `is_active`

- **`dim_violation_type`** — Violation type definitions
  - `violation_type_id` (PK), `type_code`, `type_name`, `description`, `severity`

**Fact table:**

- **`fact_violations`** — Core violation events
  - `violation_id` (PK), `camera_id` (FK), `track_id`, `violation_type`, `confidence`
  - Bounding box: `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
  - Performance: `frame_number`, `processing_latency_ms`
  - Denormalized time: `hour`, `minute`, `day_of_week`, `date`, `week_of_year`, `month`, `year`, `time_period`

**Analytical views:**

| View                    | Purpose                              | Dashboard Panel  |
|-------------------------|--------------------------------------|------------------|
| `vw_hourly_violations`  | Hourly violation counts by camera    | Heatmap          |
| `vw_daily_trend`        | Daily trend with confidence stats    | Time series      |
| `vw_camera_stats`       | Per-camera statistics                | Bar chart        |
| `vw_peak_hours`         | Hour × day-of-week violation matrix  | Heatmap          |
| `vw_today_summary`      | Today's KPI summary                  | KPI panel        |
| `vw_recent_violations`  | Latest 100 violations                | Table            |

### Stage 8: Visualization (Grafana)

**Configuration:** `grafana/dashboards/`, `grafana/provisioning/`

The Grafana Safety Dashboard provides real-time monitoring with six main panels:

1. **KPI Overview** — Total violations today, average confidence, average processing latency
2. **Temporal Trend** — Time-series chart of hourly violations over 24 hours
3. **Activity Heatmap** — Hour × Day matrix showing violation hotspots (identifies peak violation times)
4. **Location Distribution** — Bar chart comparing violations across cameras/gates
5. **Time Period Distribution** — Pie chart of violations by period (Pagi/Siang/Sore/Malam)
6. **Violation Log** — Table of recent violations with confidence, track ID, and timing details

## Data Schemas

### DetectionEvent

Represents a single YOLOv8 detection within a frame.

```python
class DetectionEvent:
    class_id: int           # 0=HELMET, 1=NO_HELMET, 2=MOTORCYCLE
    class_name: str         # Human-readable class name
    confidence: float       # Detection confidence (0.0–1.0)
    bbox: BoundingBox       # Bounding box (x1, y1, x2, y2)
    track_id: int | None    # ByteTrack ID (assigned after tracking)
```

### ViolationEvent

Represents a confirmed helmet violation, ready for Kafka/database.

```python
class ViolationEvent:
    event_id: str                     # Unique UUID
    track_id: int                     # ByteTrack track identifier
    timestamp: datetime               # UTC timestamp
    camera_id: str                    # Camera identifier
    violation_type: str               # "no_helmet"
    confidence: float                 # Average confidence over N frames
    bbox: BoundingBox                 # Bounding box at confirmation
    frame_number: int                 # Frame number at confirmation
    processing_latency_ms: float      # Pipeline processing latency
```

### Kafka Message Format (JSON)

```json
{
    "event_id": "550e8400-e29b-41d4-a716-446655440000",
    "track_id": 42,
    "timestamp": "2025-01-15T08:30:45.123456+00:00",
    "camera_id": "gate_utama_01",
    "violation_type": "no_helmet",
    "confidence": 0.87,
    "bbox_x1": 120,
    "bbox_y1": 200,
    "bbox_x2": 280,
    "bbox_y2": 450,
    "frame_number": 1542,
    "processing_latency_ms": 125.3
}
```

## Configuration Reference

All settings are loaded from environment variables via Pydantic Settings. See [`.env.example`](../.env.example) for the full template.

### Video Source
| Variable        | Default            | Description                        |
|-----------------|--------------------|------------------------------------|
| `VIDEO_SOURCE`  | `sample_video.mp4` | RTSP URL or local video file path  |

### Camera Metadata
| Variable          | Default                                    | Description         |
|-------------------|--------------------------------------------|---------------------|
| `CAMERA_ID`       | `gate_utama_01`                            | Camera identifier   |
| `CAMERA_NAME`     | `Gerbang Utama ITERA`                     | Camera display name |
| `CAMERA_LOCATION` | `Jalan Terusan Ryacudu, Gerbang Utama`    | Camera location     |

### YOLOv8 Model
| Variable              | Default         | Description                          |
|-----------------------|-----------------|--------------------------------------|
| `MODEL_PATH`          | `models/best.pt`| Path to YOLOv8 model weights         |
| `MODEL_CONFIDENCE`    | `0.5`           | Minimum detection confidence (0–1)   |
| `MODEL_IOU_THRESHOLD` | `0.45`          | NMS IoU threshold (0–1)              |
| `MODEL_DEVICE`        | `0`             | CUDA device index or `cpu`           |

### Violation Logic
| Variable                    | Default | Description                                   |
|-----------------------------|---------|-----------------------------------------------|
| `VIOLATION_CONFIRM_FRAMES`  | `3`     | Consecutive frames required for confirmation  |
| `ASSOCIATION_IOU_THRESHOLD` | `0.3`   | IoU threshold for detection association       |

### Kafka
| Variable                   | Default            | Description                    |
|----------------------------|--------------------|--------------------------------|
| `KAFKA_BOOTSTRAP_SERVERS`  | `localhost:9092`   | Kafka broker address           |
| `KAFKA_TOPIC_DETECTIONS`   | `video.detections` | Topic for detection events     |
| `KAFKA_TOPIC_VIOLATIONS`   | `video.violations` | Topic for violation events     |

### Database (Supabase/PostgreSQL)
| Variable              | Default     | Description              |
|-----------------------|-------------|--------------------------|
| `SUPABASE_DB_HOST`    | `localhost` | PostgreSQL host          |
| `SUPABASE_DB_PORT`    | `5432`      | PostgreSQL port          |
| `SUPABASE_DB_NAME`    | `postgres`  | Database name            |
| `SUPABASE_DB_USER`    | `postgres`  | Database user            |
| `SUPABASE_DB_PASSWORD`| (empty)     | Database password        |

### Spark
| Variable              | Default     | Description              |
|-----------------------|-------------|--------------------------|
| `SPARK_MASTER`        | `local[2]`  | Spark master URL         |
| `SPARK_DRIVER_MEMORY` | `3g`        | Spark driver memory      |

## Performance Characteristics

Benchmark results from testing on 3,000 frames (1280×720, 25 FPS) with CPU inference:

### Detection Pipeline

| Metric             | Value       |
|--------------------|-------------|
| Average FPS        | 8.08        |
| Median FPS         | 8.39        |
| P95 FPS            | 8.91        |
| Avg latency/frame  | 128 ms      |
| YOLOv8 inference   | 126.5 ms (98.8% of total) |
| ByteTrack tracking | 1.5 ms (1.1% of total)    |
| Violation logic    | 0.03 ms (negligible)       |

### Kafka Streaming

| Metric            | Value             |
|-------------------|-------------------|
| Throughput        | 3,728 messages/sec|
| Avg latency/msg   | 0.24 ms           |
| P99 latency/msg   | 2.7 ms            |

### Spark Processing

| Metric                | Value          |
|-----------------------|----------------|
| Micro-batch interval  | 5 seconds      |
| Batch processing time | 3–5 seconds    |
| End-to-end latency    | 5–10 seconds   |
| Watermark window      | 30 seconds     |

> **Note:** GPU inference (CUDA ≥ 7.0) is expected to improve detection FPS by 5–10× (estimated 40–80 FPS).
