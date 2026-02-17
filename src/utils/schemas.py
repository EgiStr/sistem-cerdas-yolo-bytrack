"""
Pydantic schemas for event data flowing through the pipeline.
"""

from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4


class BoundingBox(BaseModel):
    """Bounding box coordinates (xyxy format)."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class DetectionEvent(BaseModel):
    """A single detection from YOLOv8 in a frame."""
    class_id: int
    class_name: str  # 'Helmet', 'NoHelmet', 'Motorbike'
    confidence: float
    bbox: BoundingBox
    track_id: Optional[int] = None


class ViolationEvent(BaseModel):
    """A confirmed violation event ready for Kafka/DB."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    track_id: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    camera_id: str
    violation_type: str = "no_helmet"
    confidence: float
    bbox: BoundingBox
    frame_number: int
    processing_latency_ms: Optional[float] = None

    def to_kafka_dict(self) -> dict:
        """Serialize for Kafka producer."""
        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "timestamp": self.timestamp.isoformat(),
            "camera_id": self.camera_id,
            "violation_type": self.violation_type,
            "confidence": self.confidence,
            "bbox_x1": int(self.bbox.x1),
            "bbox_y1": int(self.bbox.y1),
            "bbox_x2": int(self.bbox.x2),
            "bbox_y2": int(self.bbox.y2),
            "frame_number": self.frame_number,
            "processing_latency_ms": self.processing_latency_ms,
        }


class FrameResult(BaseModel):
    """Result of processing a single video frame."""
    frame_number: int
    timestamp: datetime
    total_detections: int
    helmets: int
    no_helmets: int
    motorbikes: int
    violations: list[ViolationEvent] = []
    fps: Optional[float] = None
