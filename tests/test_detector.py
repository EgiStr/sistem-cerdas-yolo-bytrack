"""
Test suite for the detection module.

Run: uv run pytest tests/test_detector.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.utils.schemas import BoundingBox, ViolationEvent, DetectionEvent
from src.detector.violation import ViolationDetector, compute_iou, compute_centroid_distance


class TestBoundingBox:
    """Tests for the BoundingBox schema."""

    def test_center_calculation(self):
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        assert bbox.center == (200.0, 300.0)

    def test_area_calculation(self):
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert bbox.area == 10000.0


class TestIoU:
    """Tests for IoU computation."""

    def test_perfect_overlap(self):
        box = np.array([0, 0, 100, 100])
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        box_a = np.array([0, 0, 50, 50])
        box_b = np.array([100, 100, 200, 200])
        assert compute_iou(box_a, box_b) == 0.0

    def test_partial_overlap(self):
        box_a = np.array([0, 0, 100, 100])
        box_b = np.array([50, 50, 150, 150])
        iou = compute_iou(box_a, box_b)
        assert 0.0 < iou < 1.0
        # Intersection: 50x50 = 2500, Union: 10000+10000-2500 = 17500
        expected = 2500 / 17500
        assert abs(iou - expected) < 1e-6


class TestViolationDetector:
    """Tests for the ViolationDetector logic."""

    def _make_detections(self, class_ids, tracker_ids, boxes, confidences=None):
        """Helper to create mock supervision Detections."""
        import supervision as sv

        n = len(class_ids)
        if confidences is None:
            confidences = [0.9] * n

        detections = sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
            confidence=np.array(confidences, dtype=np.float32),
            tracker_id=np.array(tracker_ids, dtype=int),
        )
        detections.data["class_name"] = np.array([
            {0: "Helmet", 1: "NoHelmet", 2: "Motorbike"}[c]
            for c in class_ids
        ])
        return detections

    def test_no_violation_with_helmet(self):
        """No violation when rider has helmet."""
        detector = ViolationDetector(confirm_frames=1)

        # Helmet(0) near Motorbike(2), but no NoHelmet
        detections = self._make_detections(
            class_ids=[0, 2],
            tracker_ids=[1, 2],
            boxes=[[100, 100, 200, 200], [90, 90, 210, 300]],
        )

        violations = detector.check(detections, frame_number=1)
        assert len(violations) == 0

    def test_violation_after_n_frames(self):
        """Violation only fires after N consecutive frames."""
        N = 3
        detector = ViolationDetector(confirm_frames=N, iou_threshold=0.1)

        for frame in range(1, N + 1):
            detections = self._make_detections(
                class_ids=[1, 2],  # NoHelmet + Motorbike
                tracker_ids=[10, 20],
                boxes=[[100, 50, 200, 200], [80, 100, 220, 350]],  # Overlapping
            )
            violations = detector.check(detections, frame_number=frame)

            if frame < N:
                assert len(violations) == 0, f"Should not fire at frame {frame}"
            else:
                assert len(violations) == 1, f"Should fire at frame {frame}"
                assert violations[0].track_id == 10

    def test_deduplication(self):
        """Same track_id should not generate multiple violations."""
        detector = ViolationDetector(confirm_frames=1, iou_threshold=0.1)

        detections = self._make_detections(
            class_ids=[1, 2],
            tracker_ids=[42, 99],
            boxes=[[100, 50, 200, 200], [80, 100, 220, 350]],
        )

        v1 = detector.check(detections, frame_number=1)
        assert len(v1) == 1

        v2 = detector.check(detections, frame_number=2)
        assert len(v2) == 0, "Should not fire again for same track_id"

    def test_no_motorbike_no_violation(self):
        """NoHelmet without motorbike should not trigger violation."""
        detector = ViolationDetector(confirm_frames=1)

        detections = self._make_detections(
            class_ids=[1],  # Only NoHelmet, no Motorbike
            tracker_ids=[5],
            boxes=[[100, 100, 200, 200]],
        )

        violations = detector.check(detections, frame_number=1)
        assert len(violations) == 0


class TestViolationEvent:
    """Tests for ViolationEvent schema."""

    def test_kafka_serialization(self):
        event = ViolationEvent(
            track_id=42,
            camera_id="gate_utama_01",
            violation_type="no_helmet",
            confidence=0.87,
            bbox=BoundingBox(x1=100, y1=200, x2=300, y2=400),
            frame_number=1523,
        )

        kafka_dict = event.to_kafka_dict()
        assert kafka_dict["track_id"] == 42
        assert kafka_dict["camera_id"] == "gate_utama_01"
        assert kafka_dict["confidence"] == 0.87
        assert kafka_dict["bbox_x1"] == 100
        assert "event_id" in kafka_dict
        assert "timestamp" in kafka_dict
