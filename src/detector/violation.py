"""
Violation Detection Logic

Implements the core logic for determining helmet violations:
1. DRIVER_NO_HELMET detection = direct violation signal (model-level classification)
2. Track DRIVER_NO_HELMET state per track_id over N consecutive frames
3. Emit violation event only after confirmation (N-frame rule)
4. Deduplicate: one violation per track_id per session

Note: The YOLOv8 model uses classes {0: DRIVER_HELMET, 1: DRIVER_NO_HELMET, 2: MOTORCYCLE}.
Since DRIVER_NO_HELMET already encodes "rider without helmet on motorcycle",
no geometric association between person and motorcycle is needed.
The N-frame temporal confirmation rule prevents false positives from flickering detections.
"""

import numpy as np
import supervision as sv
from datetime import datetime, timezone
from collections import defaultdict
from loguru import logger
from config.settings import settings
from src.utils.schemas import BoundingBox, ViolationEvent


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute IoU between two boxes in xyxy format.

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def compute_ioa(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection over Area (IoA) of box_a with respect to box_b.
    Useful for checking if a person (box_a) is riding a motorbike (box_b).

    Args:
        box_a: [x1, y1, x2, y2] (e.g., person/driver)
        box_b: [x1, y1, x2, y2] (e.g., motorbike)
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])

    return intersection / area_a if area_a > 0 else 0.0


def compute_centroid_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Euclidean distance between centroids of two boxes."""
    cx_a = (box_a[0] + box_a[2]) / 2
    cy_a = (box_a[1] + box_a[3]) / 2
    cx_b = (box_b[0] + box_b[2]) / 2
    cy_b = (box_b[1] + box_b[3]) / 2
    return float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))


class ViolationDetector:
    """
    Detects helmet violations based on tracked detections.

    Logic:
    - A "DRIVER_NO_HELMET" detection = direct violation signal
    - Must persist for N consecutive frames to confirm (temporal filtering)
    - Each track_id emits at most one violation event (deduplication)
    """

    # Class IDs (match model: {0: DRIVER_HELMET, 1: DRIVER_NO_HELMET, 2: MOTORCYCLE})
    # ⚠️ TESTING MODE: Swapped — HELMET treated as violation to generate more events
    HELMET = 1       # DRIVER_NO_HELMET (original: 0)
    NO_HELMET = 0    # DRIVER_HELMET as violation trigger (original: 1)
    MOTORBIKE = 2    # MOTORCYCLE

    def __init__(
        self,
        confirm_frames: int | None = None,
        iou_threshold: float | None = None,
        camera_id: str | None = None,
    ):
        self.confirm_frames = confirm_frames or settings.violation_confirm_frames
        self.iou_threshold = iou_threshold or settings.association_iou_threshold
        self.camera_id = camera_id or settings.camera_id
        # Patience for transient occlusion (number of frames a track can be missing without resetting streak)
        self.patience_frames = 2 

        # State tracking: track_id -> consecutive NoHelmet frame count
        self._no_helmet_streak: dict[int, int] = defaultdict(int)

        # State tracking: track_id -> consecutive frames missing from NoHelmet detections
        self._missing_count: dict[int, int] = defaultdict(int)

        # Set of track_ids that already emitted a violation (dedup)
        self._violated_ids: set[int] = set()

        # Track confidence scores for averaging
        self._confidence_accumulator: dict[int, list[float]] = defaultdict(list)

        logger.info(
            f"ViolationDetector: confirm_frames={self.confirm_frames}, "
            f"iou_threshold={self.iou_threshold}, camera_id={self.camera_id}"
        )

    def check(
        self,
        detections: sv.Detections,
        frame_number: int,
    ) -> list[ViolationEvent]:
        """
        Check tracked detections for helmet violations.

        Since DRIVER_NO_HELMET class directly indicates a helmetless rider,
        this method applies N-frame temporal confirmation and deduplication.

        Args:
            detections: Tracked detections with tracker_id populated
            frame_number: Current frame number

        Returns:
            List of confirmed ViolationEvent(s) for this frame
        """
        if len(detections) == 0:
            return []

        violations = []

        # Separate detections by class
        class_ids = detections.class_id
        tracker_ids = detections.tracker_id
        xyxy = detections.xyxy
        confidences = detections.confidence

        # Get indices for DRIVER_NO_HELMET
        no_helmet_mask = class_ids == self.NO_HELMET
        no_helmet_indices = np.where(no_helmet_mask)[0]

        # Get indices for MOTORCYCLE
        motorcycle_mask = class_ids == self.MOTORBIKE
        motorcycle_indices = np.where(motorcycle_mask)[0]

        if len(no_helmet_indices) == 0:
            # Decay streak for tracks not seen as NoHelmet
            self._decay_streaks(tracker_ids, no_helmet_indices)
            return []

        # Process each DRIVER_NO_HELMET detection
        for nh_idx in no_helmet_indices:
            nh_box = xyxy[nh_idx]
            nh_track_id = tracker_ids[nh_idx] if tracker_ids is not None else None
            nh_confidence = confidences[nh_idx]

            if nh_track_id is None:
                continue

            # Skip if this track already generated a violation
            if nh_track_id in self._violated_ids:
                continue

            # Geometric Spatial Association (IoA) for Pillion Riders (Boncengan)
            # Check if this NO_HELMET driver is geometrically associated with any MOTORCYCLE
            is_riding = False
            for m_idx in motorcycle_indices:
                m_box = xyxy[m_idx]
                ioa = compute_ioa(nh_box, m_box)
                # If >= 20% of the person's bounding box overlaps with a motorcycle, consider them riding
                if ioa >= 0.2:
                    is_riding = True
                    break

            if not is_riding and len(motorcycle_indices) > 0:
                # If there are motorcycles but this person doesn't overlap with any, they might be a pedestrian (False Positive)
                # We skip incrementing their violation streak.
                continue

            # DRIVER_NO_HELMET confirmed — increment streak and reset missing count
            self._no_helmet_streak[nh_track_id] += 1
            self._missing_count[nh_track_id] = 0
            self._confidence_accumulator[nh_track_id].append(float(nh_confidence))

            # Check if confirmed (N consecutive frames)
            if self._no_helmet_streak[nh_track_id] >= self.confirm_frames:
                avg_confidence = np.mean(self._confidence_accumulator[nh_track_id])

                violation = ViolationEvent(
                    track_id=int(nh_track_id),
                    timestamp=datetime.now(timezone.utc),
                    camera_id=self.camera_id,
                    violation_type="no_helmet",
                    confidence=float(avg_confidence),
                    bbox=BoundingBox(
                        x1=float(nh_box[0]),
                        y1=float(nh_box[1]),
                        x2=float(nh_box[2]),
                        y2=float(nh_box[3]),
                    ),
                    frame_number=frame_number,
                )
                violations.append(violation)
                self._violated_ids.add(nh_track_id)

                logger.info(
                    f"🚨 VIOLATION CONFIRMED: track_id={nh_track_id}, "
                    f"confidence={avg_confidence:.2f}, frame={frame_number}"
                )

        # Decay streaks for tracks not in current NoHelmet detections
        self._decay_streaks(tracker_ids, no_helmet_indices)

        return violations

    def _decay_streaks(
        self, all_tracker_ids: np.ndarray | None, no_helmet_indices: np.ndarray
    ):
        """Reset streaks for tracks that are no longer classified as NoHelmet."""
        if all_tracker_ids is None:
            return

        current_no_helmet_ids = set()
        for idx in no_helmet_indices:
            if all_tracker_ids[idx] is not None:
                current_no_helmet_ids.add(int(all_tracker_ids[idx]))

        # Reset streak for tracks not in current NoHelmet only if patience exceeded
        for track_id in list(self._no_helmet_streak.keys()):
            if track_id not in current_no_helmet_ids:
                self._missing_count[track_id] += 1
                
                if self._missing_count[track_id] > self.patience_frames:
                    self._no_helmet_streak[track_id] = 0
                    self._missing_count[track_id] = 0
                    self._confidence_accumulator[track_id].clear()

    def reset(self):
        """Reset all violation state. Used when switching video sources."""
        self._no_helmet_streak.clear()
        self._missing_count.clear()
        self._violated_ids.clear()
        self._confidence_accumulator.clear()
        logger.info("ViolationDetector state reset")

    @property
    def stats(self) -> dict:
        """Get current violation statistics."""
        return {
            "total_violations": len(self._violated_ids),
            "active_streaks": len(self._no_helmet_streak),
            "violated_track_ids": list(self._violated_ids),
        }
