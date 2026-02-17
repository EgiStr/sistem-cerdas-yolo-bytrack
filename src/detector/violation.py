"""
Violation Detection Logic

Implements the core logic for determining helmet violations:
1. Associate Person/NoHelmet detections with nearby Motorbike detections
2. Track NoHelmet state per track_id over N consecutive frames
3. Emit violation event only after confirmation (N-frame rule)
4. Deduplicate: one violation per track_id per session
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
    - A "NoHelmet" detection near a "Motorbike" = potential violation
    - Must persist for N consecutive frames to confirm
    - Each track_id emits at most one violation event
    """

    # Class IDs (match your YOLO model)
    HELMET = 0
    NO_HELMET = 1
    MOTORBIKE = 2

    def __init__(
        self,
        confirm_frames: int | None = None,
        iou_threshold: float | None = None,
        camera_id: str | None = None,
    ):
        self.confirm_frames = confirm_frames or settings.violation_confirm_frames
        self.iou_threshold = iou_threshold or settings.association_iou_threshold
        self.camera_id = camera_id or settings.camera_id

        # State tracking: track_id -> consecutive NoHelmet frame count
        self._no_helmet_streak: dict[int, int] = defaultdict(int)

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

        # Get indices for each class
        no_helmet_mask = class_ids == self.NO_HELMET
        motorbike_mask = class_ids == self.MOTORBIKE

        no_helmet_indices = np.where(no_helmet_mask)[0]
        motorbike_indices = np.where(motorbike_mask)[0]

        # If no motorbikes or no-helmet detections, nothing to do
        if len(no_helmet_indices) == 0 or len(motorbike_indices) == 0:
            # Decay streak for tracks not seen as NoHelmet
            self._decay_streaks(tracker_ids, no_helmet_indices)
            return []

        # Associate NoHelmet with nearby Motorbike
        for nh_idx in no_helmet_indices:
            nh_box = xyxy[nh_idx]
            nh_track_id = tracker_ids[nh_idx] if tracker_ids is not None else None
            nh_confidence = confidences[nh_idx]

            if nh_track_id is None:
                continue

            # Skip if this track already generated a violation
            if nh_track_id in self._violated_ids:
                continue

            # Check if NoHelmet is near any Motorbike
            is_riding = self._is_near_motorbike(nh_box, xyxy[motorbike_indices])

            if is_riding:
                # Increment streak
                self._no_helmet_streak[nh_track_id] += 1
                self._confidence_accumulator[nh_track_id].append(float(nh_confidence))

                # Check if confirmed
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
            else:
                # Not near motorbike, reset streak
                self._no_helmet_streak[nh_track_id] = 0
                self._confidence_accumulator[nh_track_id].clear()

        # Decay streaks for tracks not in current NoHelmet detections
        self._decay_streaks(tracker_ids, no_helmet_indices)

        return violations

    def _is_near_motorbike(
        self, person_box: np.ndarray, motorbike_boxes: np.ndarray
    ) -> bool:
        """
        Check if a person/no-helmet box is associated with any motorbike.

        Uses IoU-based proximity. Person riding a motorbike typically has
        significant vertical overlap with the motorbike bounding box.
        """
        for mb_box in motorbike_boxes:
            iou = compute_iou(person_box, mb_box)
            if iou >= self.iou_threshold:
                return True

            # Also check centroid proximity as fallback
            # (useful when IoU is low but person is clearly on motorbike)
            dist = compute_centroid_distance(person_box, mb_box)
            box_diag = np.sqrt(
                (mb_box[2] - mb_box[0]) ** 2 + (mb_box[3] - mb_box[1]) ** 2
            )
            if box_diag > 0 and dist / box_diag < 0.5:
                return True

        return False

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

        # Reset streak for tracks not in current NoHelmet
        for track_id in list(self._no_helmet_streak.keys()):
            if track_id not in current_no_helmet_ids:
                self._no_helmet_streak[track_id] = 0
                self._confidence_accumulator[track_id].clear()

    def reset(self):
        """Reset all violation state. Used when switching video sources."""
        self._no_helmet_streak.clear()
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
