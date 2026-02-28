"""
ByteTrack Object Tracker Wrapper

Uses supervision's ByteTrack implementation for multi-object tracking.
Maintains unique track IDs across frames, handling occlusion gracefully.
"""

import supervision as sv
from loguru import logger
from config.settings import settings


class ObjectTracker:
    """ByteTrack-based multi-object tracker using supervision library."""

    def __init__(
        self,
        track_activation_threshold: float | None = None,
        lost_track_buffer: int | None = None,
        minimum_matching_threshold: float | None = None,
        frame_rate: int | None = None,
        minimum_consecutive_frames: int | None = None,
    ):
        """
        Initialize ByteTrack tracker.

        All parameters default to values from config/settings.py (loaded from .env).
        These defaults are tuned for slow devices (~200ms/frame ≈ 5 FPS effective).

        Args:
            track_activation_threshold: Detection confidence threshold for track activation.
                Lower = track more objects including low-confidence ones.
            lost_track_buffer: Frames to keep lost tracks alive before removal.
                Higher = better occlusion & re-ID handling. 90 ≈ 3s @30fps.
            minimum_matching_threshold: Cost threshold for matching detections to tracks.
                HIGHER = more lenient matching (fewer ID switches, recommended 0.8).
                LOWER = stricter matching (more ID switches).
                The fused cost = 1 - IoU*confidence; matches with cost > thresh are rejected.
            frame_rate: Frame rate for computing max_time_lost (= frame_rate/30 * lost_buffer).
                For video files processed frame-by-frame: use video FPS (e.g., 30).
                For RTSP with frame skipping: use effective processing FPS.
            minimum_consecutive_frames: Consecutive detections needed before assigning
                a visible track ID. Prevents noisy false-positive tracks.
        """
        # Load all params from settings (single source of truth)
        activation = track_activation_threshold if track_activation_threshold is not None else settings.tracker_activation_threshold
        lost_buffer = lost_track_buffer if lost_track_buffer is not None else settings.tracker_lost_buffer
        matching = minimum_matching_threshold if minimum_matching_threshold is not None else settings.tracker_matching_threshold
        fps = frame_rate if frame_rate is not None else settings.tracker_frame_rate
        min_frames = minimum_consecutive_frames if minimum_consecutive_frames is not None else settings.tracker_min_consecutive_frames

        # frame_rate=0 should not reach ByteTrack; pipeline auto-calculates before calling
        if fps <= 0:
            fps = 30
            logger.warning("tracker_frame_rate=0 but no override provided; defaulting to 30")

        self.tracker = sv.ByteTrack(
            track_activation_threshold=activation,
            lost_track_buffer=lost_buffer,
            minimum_matching_threshold=matching,
            frame_rate=fps,
            minimum_consecutive_frames=min_frames,
        )

        # Internal: max_time_lost = int(frame_rate / 30.0 * lost_track_buffer)
        max_time_lost = int(fps / 30.0 * lost_buffer)
        logger.info(
            f"ByteTrack initialized: activation={activation}, "
            f"lost_buffer={lost_buffer}, matching={matching}, "
            f"fps={fps}, min_consecutive={min_frames}, "
            f"max_time_lost={max_time_lost} frames"
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
