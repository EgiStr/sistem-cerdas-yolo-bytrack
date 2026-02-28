"""
ITERA Smart Sentinel - Centralized Configuration

Uses Pydantic Settings to load config from .env file with validation.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Video Source ---
    video_source: str = Field(default="sample_video.mp4", description="RTSP URL or local video path")

    # --- Camera Info ---
    camera_id: str = Field(default="gate_utama_01")
    camera_name: str = Field(default="Gerbang Utama ITERA")
    camera_location: str = Field(default="Jalan Terusan Ryacudu, Gerbang Utama")

    # --- YOLOv8 Model ---
    model_path: str = Field(default="models/best.pt")
    model_confidence: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="YOLO confidence threshold. Set LOW (0.25) so partially-occluded "
                    "detections reach ByteTrack's second-tier association instead of "
                    "being discarded. Violation logic has its own temporal filter."
    )
    model_iou_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="YOLO NMS IoU threshold. Set HIGH (0.7) to keep overlapping "
                    "boxes for rider + passenger on the same motorcycle. "
                    "Lower values aggressively suppress nearby detections."
    )
    model_device: str = Field(default="0", description="CUDA device index or 'cpu'")
    model_imgsz: int = Field(default=640, description="Input image size for YOLO")

    # --- Pipeline ---
    pipeline_frame_stride: int = Field(
        default=1, ge=1,
        description="Process every Nth frame, skip the rest. "
                    "1 = process all frames (default). "
                    "3 = process 1 of 3 (3x faster, good for slow CPU devices). "
                    "Skipped frames use cheap cap.grab() instead of full decode+inference"
    )

    # --- ByteTrack Tracker ---
    tracker_activation_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Splits detections into HIGH (> this) and LOW (YOLO conf .. this) tiers. "
                    "HIGH-conf detections do normal matching (tier 1). "
                    "LOW-conf detections rescue lost tracks via second-chance matching (tier 2). "
                    "Set BETWEEN model_confidence and desired high-conf cutoff. "
                    "E.g., model_confidence=0.25, activation=0.4 → tier2 covers 0.25-0.4"
    )
    tracker_lost_buffer: int = Field(
        default=120, ge=1,
        description="Frames to keep lost tracks alive. "
                    "Higher = better occlusion/re-ID handling (120 ≈ 4.8s @25fps)"
    )
    tracker_matching_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Cost threshold for matching detections to existing tracks. "
                    "HIGHER = more lenient (accepts lower IoU matches, fewer ID switches). "
                    "LOWER = stricter (requires higher IoU, more ID switches). "
                    "0.85 accepts IoU*conf > 0.15; recommended for slow devices"
    )
    tracker_frame_rate: int = Field(
        default=0, ge=0,
        description="Frame rate for ByteTrack's max_time_lost = frame_rate/30 * lost_buffer. "
                    "0 = auto-calculate from video FPS / pipeline_frame_stride. "
                    "Manual override: set to effective FPS after stride"
    )
    tracker_min_consecutive_frames: int = Field(
        default=1, ge=1,
        description="Consecutive frames before a track gets a visible ID. "
                    "Use 1 to track immediately (violation logic already has its own "
                    "temporal confirm_frames filter). Higher values cause detection "
                    "gaps that break violation streak counting."
    )

    # --- Violation Logic ---
    violation_confirm_frames: int = Field(default=3, ge=1)
    violation_patience_frames: int = Field(
        default=2, ge=1,
        description="Grace frames before resetting no-helmet streak. "
                    "Higher = more tolerant of intermittent detection gaps"
    )
    association_iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Kafka ---
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    kafka_topic_detections: str = Field(default="video.detections")
    kafka_topic_violations: str = Field(default="video.violations")

    # --- Supabase PostgreSQL ---
    supabase_db_host: str = Field(default="localhost")
    supabase_db_port: int = Field(default=5432)
    supabase_db_name: str = Field(default="postgres")
    supabase_db_user: str = Field(default="postgres")
    supabase_db_password: str = Field(default="")

    # --- Spark ---
    spark_master: str = Field(default="local[2]")
    spark_driver_memory: str = Field(default="3g")

    # --- Grafana ---
    grafana_url: str = Field(default="http://localhost:3000")

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.supabase_db_user}:{self.supabase_db_password}"
            f"@{self.supabase_db_host}:{self.supabase_db_port}/{self.supabase_db_name}"
        )

    @property
    def jdbc_url(self) -> str:
        """Construct JDBC URL for Spark."""
        return (
            f"jdbc:postgresql://{self.supabase_db_host}:{self.supabase_db_port}"
            f"/{self.supabase_db_name}"
        )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton instance
settings = Settings()
