"""
ByteTrack Object Tracker Wrapper

Uses supervision's ByteTrack implementation for multi-object tracking.
Maintains unique track IDs across frames, handling occlusion gracefully.
"""

import supervision as sv
from loguru import logger


class ObjectTracker:
    """ByteTrack-based multi-object tracker using supervision library."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 15,
    ):
        """
        Initialize ByteTrack tracker.

        Args:
            track_activation_threshold: Detection confidence threshold for track activation.
                                        Lower = track more, may increase false positives.
            lost_track_buffer: Number of frames to keep lost tracks alive before removal.
                               Higher = better occlusion handling, more memory.
            minimum_matching_threshold: Minimum IoU for matching detections to existing tracks.
            frame_rate: Expected video frame rate for track age estimation.
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        logger.info(
            f"ByteTrack initialized: activation={track_activation_threshold}, "
            f"lost_buffer={lost_track_buffer}, matching={minimum_matching_threshold}, "
            f"fps={frame_rate}"
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections for current frame.

        Args:
            detections: supervision.Detections from YOLOv8 model

        Returns:
            supervision.Detections with tracker_id field populated
        """
        if len(detections) == 0:
            return detections

        tracked = self.tracker.update_with_detections(detections)
        return tracked

    def reset(self):
        """Reset all tracks. Used when switching video sources."""
        self.tracker.reset()
        logger.info("Tracker reset - all tracks cleared")
