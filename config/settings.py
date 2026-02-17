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
    model_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    model_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    model_device: str = Field(default="0", description="CUDA device index or 'cpu'")
    model_imgsz: int = Field(default=640, description="Input image size for YOLO")

    # --- Violation Logic ---
    violation_confirm_frames: int = Field(default=3, ge=1)
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
