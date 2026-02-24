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
        self.tracker = ObjectTracker(frame_rate=15)
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

    def _annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame for visualization."""
        if len(detections) == 0:
            return frame

        # Build labels
        labels = []
        for i in range(len(detections)):
            class_name = detections.data.get("class_name", [None])[i]
            tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None
            confidence = detections.confidence[i]

            label = f"#{tracker_id} " if tracker_id is not None else ""
            label += f"{class_name} {confidence:.2f}"
            labels.append(label)

        annotated = self.box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated = self.label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        # Draw One-to-Many Association Lines (Rider to Motorbike via IoA)
        # Import compute_ioa here to avoid circular imports if any, or assume it's available
        from src.detector.violation import compute_ioa
        
        # Class IDs (match model: {0: DRIVER_HELMET, 1: DRIVER_NO_HELMET, 2: MOTORCYCLE})
        # Note: In violation.py NO_HELMET is 0 and HELMET is 1 due to swapping.
        # We will use the same indices used in violation.py logic.
        class_ids = detections.class_id
        xyxy = detections.xyxy
        
        # Masks for riders (Helmet/NoHelmet) and Motorbikes
        rider_mask = (class_ids == 0) | (class_ids == 1)
        motorbike_mask = (class_ids == 2)
        
        rider_indices = np.where(rider_mask)[0]
        motorbike_indices = np.where(motorbike_mask)[0]

        iou_threshold = settings.association_iou_threshold  # typically 0.3
        
        # For each motorbike, find all riders associated with it (One-to-Many)
        for m_idx in motorbike_indices:
            m_box = xyxy[m_idx]
            m_center = (int((m_box[0] + m_box[2]) / 2), int((m_box[1] + m_box[3]) / 2))
            
            for r_idx in rider_indices:
                r_box = xyxy[r_idx]
                r_center = (int((r_box[0] + r_box[2]) / 2), int((r_box[1] + r_box[3]) / 2))
                
                ioa = compute_ioa(r_box, m_box)
                if ioa >= iou_threshold:
                    # Draw a line connecting rider to motorbike
                    # Green line for successful spatial association
                    cv2.line(annotated, r_center, m_center, (0, 255, 0), 2)
                    cv2.circle(annotated, r_center, 4, (255, 0, 0), -1) # Blue dot for rider
                    cv2.circle(annotated, m_center, 4, (0, 0, 255), -1) # Red dot for motor

        # Add FPS and violation count overlay
        fps_text = f"FPS: {self._fps_history[-1]:.1f}" if self._fps_history else "FPS: --"
        cv2.putText(
            annotated, fps_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )
        cv2.putText(
            annotated,
            f"Violations: {self._total_violations}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
        )

        return annotated

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
