# Copilot Instructions — ITERA Smart Sentinel

## Architecture Overview

Two decoupled processes form the pipeline: **Detection Pipeline** (Python/uv) and **Stream Processor** (Spark).

1. **Detection Pipeline** (`src/pipeline/main.py` → `SentinelPipeline`): Video → YOLOv8 (`src/detector/model.py`) → ByteTrack (`src/detector/tracker.py`) → Violation confirmation (`src/detector/violation.py`) → Kafka (`src/streaming/producer.py`)
2. **Stream Processor** (`spark/stream_processor.py`): Kafka → Spark Structured Streaming → dedup + time-dimension enrichment → PostgreSQL Star Schema → Grafana

All config flows through a **single Pydantic `Settings` singleton** in `config/settings.py`, loaded from `.env`. Never hardcode config values—add new fields there.

## Key Conventions

- **Package manager:** `uv` (not pip). Run with `uv run python -m ...`, install with `uv sync`.
- **Module execution:** The pipeline is invoked as `uv run python -m src.pipeline.main`, not as a script. All `src/` imports use absolute paths from the project root (e.g., `from src.detector.model import HelmetDetector`).
- **Detection output type:** All detectors return `supervision.Detections` objects (from the `supervision` library). Tracker and violation logic consume these directly—never convert to plain dicts mid-pipeline.
- **Schemas:** Event data models live in `src/utils/schemas.py` as Pydantic models (`ViolationEvent`, `BoundingBox`, `DetectionEvent`). Kafka serialization uses `ViolationEvent.to_kafka_dict()`.
- **Logging:** Use `loguru.logger` everywhere (not stdlib `logging`).

## YOLOv8 Model — Class IDs

The model outputs 3 classes. These IDs are canonical and defined in `src/detector/model.py::CLASS_NAMES`:

| ID | Name | Meaning |
|----|------|---------|
| 0 | `DRIVER_HELMET` | Rider wearing helmet |
| 1 | `DRIVER_NO_HELMET` | Rider without helmet |
| 2 | `MOTORCYCLE` | Motorcycle vehicle |

> ⚠️ **Testing mode swap:** `src/detector/violation.py` currently has `HELMET = 1` and `NO_HELMET = 0` (swapped from production values) to generate more violation events during testing. Revert `NO_HELMET = 1` and `HELMET = 0` for production.

## Violation Detection Logic

`ViolationDetector.check()` in `src/detector/violation.py` implements:
- **N-frame temporal confirmation:** A track must be classified as no-helmet for `confirm_frames` (default 3) consecutive frames before emitting a violation.
- **Per-track deduplication:** Each `track_id` emits at most one `ViolationEvent` per session (stored in `_violated_ids`).
- **Patience-based decay:** Missing tracks get `patience_frames` (2) grace frames before streak resets.
- **IoA spatial filter:** Driver boxes must overlap ≥20% with a motorcycle box to qualify (filters pedestrians).

When modifying violation logic, preserve these four invariants.

## Spark Stream Processor

`spark/stream_processor.py` runs outside the uv venv (system Python under Spark). It reads `.env` manually if `python-dotenv` is unavailable. Config uses `os.getenv()` directly—not the Pydantic `Settings` class.

Key processing: 30-second watermark dedup by `(track_id, camera_id)`, Indonesian time-period labels (`pagi/siang/sore/malam`), writes to `fact_violations` and `dim_time` tables via JDBC.

## Database Schema

Star Schema defined in `sql/` (run in order: `001` → `002` → `003`):
- **Dimensions:** `dim_camera`, `dim_violation_type`
- **Fact:** `fact_violations` — has embedded (denormalized) time fields for fast Grafana queries
- **Views:** `sql/003_create_views.sql` defines 6 analytical views (`vw_hourly_violations`, `vw_daily_trend`, etc.) consumed by Grafana

## Developer Commands

```bash
# Install dependencies
uv sync

# Run detection pipeline (no Kafka, with display)
uv run python -m src.pipeline.main --source sample_video.mp4 --no-kafka --display

# Run detection pipeline (with Kafka)
uv run python -m src.pipeline.main --source sample_video.mp4

# Run Spark stream processor
/opt/spark/bin/spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8,org.postgresql:postgresql:42.7.1 \
    spark/stream_processor.py

# Debug Kafka messages
uv run python -m src.streaming.consumer

# Run tests
uv run pytest tests/

# Run benchmarks (for Bab 4 metrics)
uv run python tests/benchmark_pipeline.py --source test.mp4 --frames 3000

# Local PostgreSQL via Docker
docker compose -f docker-compose.db.yml up -d
```

## File Roles Quick Reference

| Path | Role |
|------|------|
| `config/settings.py` | Single source of truth for all config (Pydantic Settings) |
| `src/detector/model.py` | YOLOv8 wrapper → returns `sv.Detections` |
| `src/detector/tracker.py` | ByteTrack wrapper via `supervision.ByteTrack` |
| `src/detector/violation.py` | Temporal N-frame confirmation + dedup + IoA spatial filter |
| `src/pipeline/main.py` | Orchestrator — wires detect→track→violate→kafka loop |
| `src/streaming/producer.py` | Kafka producer (confluent-kafka, snappy compression) |
| `src/utils/schemas.py` | Pydantic models for all event data |
| `spark/stream_processor.py` | Standalone Spark job (reads .env directly, not Settings) |
| `sql/001-003_*.sql` | Star Schema DDL — run sequentially |
| `grafana/dashboards/` | Grafana dashboard JSON (auto-provisioned) |
