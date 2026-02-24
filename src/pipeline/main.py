"""
Main Pipeline Orchestrator

Reads video frames from RTSP/file source, runs YOLOv8 detection,
tracks objects with ByteTrack, checks for helmet violations,
and sends confirmed violations to Kafka.

Usage:
    uv run python -m src.pipeline.main --source sample_video.mp4
    uv run python -m src.pipeline.main --source rtsp://camera_url
"""

import cv2
import time
import argparse
import numpy as np
import supervision as sv
from datetime import datetime
from loguru import logger

from config.settings import settings
from src.detector.model import HelmetDetector
from src.detector.tracker import ObjectTracker
from src.detector.violation import ViolationDetector
from src.streaming.producer import ViolationProducer


class SentinelPipeline:
    """
    Main pipeline: Video → Detect → Track → Violation Check → Kafka

    Handles video capture, inference, tracking, violation logic,
    and event streaming in a single main loop.
    """

    def __init__(
        self,
        video_source: str | None = None,
        enable_kafka: bool = True,
        show_display: bool = False,
        save_output: str | None = None,
    ):
        self.video_source = video_source or settings.video_source
        self.enable_kafka = enable_kafka
        self.show_display = show_display
        self.save_output = save_output

        # Initialize components
        logger.info("=" * 60)
        logger.info("🛡️  ITERA Smart Sentinel - Pipeline Starting")
        logger.info("=" * 60)

        self.detector = HelmetDetector()
        # Tracker is initialized after opening video to use actual FPS
        self.tracker: ObjectTracker | None = None
        self.violation_checker = ViolationDetector()

        if self.enable_kafka:
            self.producer = ViolationProducer()
        else:
            self.producer = None
            logger.warning("Kafka disabled - violations will only be logged")

        # Annotators for visualization
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

        # Stats
        self._frame_count = 0
        self._total_violations = 0
        self._start_time = None
        self._fps_history: list[float] = []

    def _open_video(self) -> cv2.VideoCapture:
        """Open video source (file or RTSP stream)."""
        logger.info(f"Opening video source: {self.video_source}")

        cap = cv2.VideoCapture(self.video_source)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.video_source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

        return cap

    # ── Colour palette (BGR) ──────────────────────────────────────────
    # Chosen for high contrast on outdoor traffic footage and
    # distinguishable even with common colour-vision deficiencies.
    CLR_SAFE       = (80, 200, 80)    # muted green  – helmet OK
    CLR_MOTORCYCLE = (200, 180, 60)   # teal/cyan    – motorcycle
    CLR_NO_ASSOC   = (100, 160, 200)  # warm grey    – rider, no motor
    CLR_SUSPECT    = (0, 165, 255)    # orange       – building streak
    CLR_VIOLATION  = (60, 60, 230)    # strong red   – confirmed violation
    CLR_HUD_BG     = (30, 30, 30)     # dark grey    – HUD background
    CLR_WHITE      = (255, 255, 255)

    def _annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame for visualization."""
        annotated = frame.copy()

        if len(detections) == 0:
            self._draw_hud(annotated)
            return annotated

        from src.detector.violation import compute_ioa

        # Violation detector state
        violated_ids = self.violation_checker._violated_ids
        streak_map = self.violation_checker._no_helmet_streak
        confirm_needed = self.violation_checker.confirm_frames

        class_ids = detections.class_id
        tracker_ids = detections.tracker_id
        xyxy = detections.xyxy
        confidences = detections.confidence
        class_names = detections.data.get("class_name", [None] * len(detections))

        # Pre-compute rider ↔ motorcycle IoA map
        rider_mask = (class_ids == 0) | (class_ids == 1)
        motorbike_mask = (class_ids == 2)
        rider_indices = np.where(rider_mask)[0]
        motorbike_indices = np.where(motorbike_mask)[0]
        ioa_threshold = settings.association_iou_threshold

        riding_indices: set[int] = set()
        for r_idx in rider_indices:
            for m_idx in motorbike_indices:
                if compute_ioa(xyxy[r_idx], xyxy[m_idx]) >= ioa_threshold:
                    riding_indices.add(int(r_idx))
                    break

        # ── Draw each detection ──────────────────────────────────────
        for i in range(len(detections)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            track_id = (
                int(tracker_ids[i])
                if tracker_ids is not None and tracker_ids[i] is not None
                else None
            )
            class_id = class_ids[i]
            conf = confidences[i]
            cls_name = class_names[i] if class_names[i] is not None else ""
            is_rider = class_id in (0, 1)
            has_motor = i in riding_indices

            # ── Resolve colour, badge text, badge icon ───────────────
            if class_id == 2:
                color = self.CLR_MOTORCYCLE
                badge = ""
            elif track_id is not None and track_id in violated_ids:
                color = self.CLR_VIOLATION
                badge = "!! VIOLATION"
            elif (
                track_id is not None
                and streak_map.get(track_id, 0) > 0
                and has_motor
            ):
                streak = streak_map[track_id]
                color = self.CLR_SUSPECT
                badge = f"SUSPECT {streak}/{confirm_needed}"
            elif is_rider and not has_motor:
                color = self.CLR_NO_ASSOC
                badge = "NO MOTOR"
            else:
                color = self.CLR_SAFE
                badge = ""

            # ── Box ──────────────────────────────────────────────────
            thick = 3 if badge.startswith("!!") else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)

            # Corner accents for VIOLATION (makes it instantly visible)
            if badge.startswith("!!"):
                accent_len = min(20, (x2 - x1) // 3, (y2 - y1) // 3)
                for cx, cy, dx, dy in [
                    (x1, y1, 1, 1), (x2, y1, -1, 1),
                    (x1, y2, 1, -1), (x2, y2, -1, -1),
                ]:
                    cv2.line(annotated, (cx, cy), (cx + dx * accent_len, cy), color, 4)
                    cv2.line(annotated, (cx, cy), (cx, cy + dy * accent_len), color, 4)

            # ── Label pill (track + class + conf) ────────────────────
            id_str = f"#{track_id} " if track_id is not None else ""
            label = f"{id_str}{cls_name} {conf:.0%}"
            self._draw_pill(annotated, label, x1, y1 - 4, color, above=True)

            # ── Badge pill (status) ──────────────────────────────────
            if badge:
                # Use white text on coloured pill for status
                self._draw_pill(
                    annotated, badge, x1, y1 - 26, color, above=True,
                    font_scale=0.50, bold=True,
                )

        # ── Association lines (dashed style) ─────────────────────────
        for m_idx in motorbike_indices:
            m_box = xyxy[m_idx]
            mc = (int((m_box[0] + m_box[2]) / 2), int((m_box[1] + m_box[3]) / 2))
            for r_idx in rider_indices:
                if int(r_idx) not in riding_indices:
                    continue
                r_box = xyxy[r_idx]
                rc = (int((r_box[0] + r_box[2]) / 2), int((r_box[1] + r_box[3]) / 2))
                if compute_ioa(r_box, m_box) >= ioa_threshold:
                    self._draw_dashed_line(annotated, rc, mc, self.CLR_SAFE, 2, 8)
                    cv2.circle(annotated, rc, 4, self.CLR_SAFE, -1)
                    cv2.circle(annotated, mc, 4, self.CLR_MOTORCYCLE, -1)

        self._draw_hud(annotated)
        return annotated

    # ── Drawing helpers ──────────────────────────────────────────────

    @staticmethod
    def _draw_pill(
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple,
        above: bool = True,
        font_scale: float = 0.45,
        bold: bool = False,
    ) -> None:
        """Draw a rounded-rectangle 'pill' label with white text."""
        thick = 2 if bold else 1
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick
        )
        pad_x, pad_y = 6, 4
        if above:
            box_y1 = y - th - 2 * pad_y
            box_y2 = y
        else:
            box_y1 = y
            box_y2 = y + th + 2 * pad_y

        # Clamp to frame
        box_y1 = max(box_y1, 0)
        box_x2 = x + tw + 2 * pad_x

        # Semi-transparent filled rectangle
        overlay = img.copy()
        cv2.rectangle(overlay, (x, box_y1), (box_x2, box_y2), color, -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # White text
        text_y = box_y2 - pad_y
        cv2.putText(
            img, text, (x + pad_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_dashed_line(
        img: np.ndarray,
        pt1: tuple, pt2: tuple,
        color: tuple, thickness: int = 1, gap: int = 8,
    ) -> None:
        """Draw a dashed line between two points."""
        dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist == 0:
            return
        dx = (pt2[0] - pt1[0]) / dist
        dy = (pt2[1] - pt1[1]) / dist
        steps = int(dist / gap)
        for j in range(0, steps, 2):
            s = (int(pt1[0] + dx * j * gap), int(pt1[1] + dy * j * gap))
            e = (
                int(pt1[0] + dx * min((j + 1) * gap, dist)),
                int(pt1[1] + dy * min((j + 1) * gap, dist)),
            )
            cv2.line(img, s, e, color, thickness, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Draw heads-up display with stats panel and colour legend."""
        h, w = frame.shape[:2]

        # ── Top-left stats panel (semi-transparent) ──────────────────
        panel_h, panel_w = 80, 280
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), self.CLR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        fps_val = f"{self._fps_history[-1]:.1f}" if self._fps_history else "--"
        cv2.putText(
            frame, f"FPS  {fps_val}", (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.CLR_SAFE, 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"VIOLATIONS  {self._total_violations}", (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.CLR_VIOLATION, 2, cv2.LINE_AA,
        )

        # ── Bottom-left legend (semi-transparent) ────────────────────
        legend_items = [
            ("\u25CF Helmet OK",            self.CLR_SAFE),
            ("\u25CF Motorcycle",           self.CLR_MOTORCYCLE),
            ("\u25CF No Motor (ignored)",   self.CLR_NO_ASSOC),
            ("\u25CF Suspect (streak)",     self.CLR_SUSPECT),
            ("\u25CF VIOLATION",            self.CLR_VIOLATION),
        ]
        line_h = 22
        legend_h = len(legend_items) * line_h + 12
        legend_w = 260
        ly = h - legend_h

        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, ly), (legend_w, h), self.CLR_HUD_BG, -1)
        cv2.addWeighted(overlay2, 0.60, frame, 0.40, 0, frame)

        for idx, (text, color) in enumerate(legend_items):
            ty = ly + 18 + idx * line_h
            cv2.putText(
                frame, text, (10, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

    def run(self):
        """
        Main pipeline loop.

        Reads frames, processes them through detect → track → violation check,
        sends violations to Kafka, and optionally displays annotated output.
        """
        cap = self._open_video()
        writer = None

        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 15
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(self.save_output, fourcc, fps, (w, h))
            logger.info(f"Saving output to: {self.save_output}")

        # Initialize tracker with actual video FPS for correct max_time_lost
        video_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        self.tracker = ObjectTracker(frame_rate=video_fps)

        self._start_time = time.time()
        logger.info("🚀 Pipeline running...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("Video source ended or disconnected")
                    # For RTSP: try reconnecting
                    if self.video_source.startswith("rtsp"):
                        logger.warning("Attempting RTSP reconnection...")
                        time.sleep(2)
                        cap.release()
                        cap = self._open_video()
                        continue
                    break

                frame_start = time.time()
                self._frame_count += 1

                # Step 1: Detect
                detections = self.detector.predict(frame)

                # Step 2: Track
                tracked = self.tracker.update(detections)

                # Step 3: Check violations
                violations = self.violation_checker.check(
                    tracked, self._frame_count
                )

                # Step 4: Send to Kafka
                if violations:
                    self._total_violations += len(violations)
                    if self.producer:
                        processing_latency = (time.time() - frame_start) * 1000
                        for v in violations:
                            v.processing_latency_ms = processing_latency
                        self.producer.send_batch(violations)

                # Calculate FPS
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self._fps_history.append(fps)
                if len(self._fps_history) > 100:
                    self._fps_history.pop(0)

                # Step 5: Visualize (optional)
                if self.show_display or writer:
                    annotated = self._annotate_frame(frame, tracked)
                    if writer:
                        writer.write(annotated)
                    if self.show_display:
                        cv2.imshow("ITERA Smart Sentinel", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            logger.info("User pressed 'q' - stopping")
                            break

                # Log stats periodically
                if self._frame_count % 100 == 0:
                    avg_fps = np.mean(self._fps_history[-100:])
                    elapsed = time.time() - self._start_time
                    logger.info(
                        f"📊 Frame {self._frame_count} | "
                        f"FPS: {avg_fps:.1f} | "
                        f"Violations: {self._total_violations} | "
                        f"Elapsed: {elapsed:.1f}s"
                    )

        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")

        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if self.show_display:
                cv2.destroyAllWindows()
            if self.producer:
                self.producer.close()

            self._print_summary()

    def _print_summary(self):
        """Print final pipeline statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        avg_fps = np.mean(self._fps_history) if self._fps_history else 0

        logger.info("=" * 60)
        logger.info("📋 PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total frames processed: {self._frame_count}")
        logger.info(f"Total violations detected: {self._total_violations}")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        logger.info(f"Total runtime: {elapsed:.1f}s")
        logger.info(f"Violation stats: {self.violation_checker.stats}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ITERA Smart Sentinel Pipeline")
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video source (file path or RTSP URL)"
    )
    parser.add_argument(
        "--no-kafka", action="store_true",
        help="Disable Kafka (violations logged only)"
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show annotated video window"
    )
    parser.add_argument(
        "--save-output", type=str, default=None,
        help="Save annotated output to video file"
    )
    args = parser.parse_args()

    pipeline = SentinelPipeline(
        video_source=args.source,
        enable_kafka=not args.no_kafka,
        show_display=args.display,
        save_output=args.save_output,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
