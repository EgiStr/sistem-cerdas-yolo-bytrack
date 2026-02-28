"""
MOT Challenge Format Exporter

Exports pipeline tracking results to MOT Challenge format for
standardized evaluation with MOTA/MOTP/IDF1 metrics.

MOT Format (per line):
    <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

Where x, y, z are -1 (unused in 2D tracking).

Usage:
    # During pipeline run:
    uv run python -m src.pipeline.main --source video.mp4 --export-mot output_dir --no-kafka

    # Standalone on existing video:
    uv run python -m src.evaluation.mot_exporter --source video.mp4 --output output_dir
"""

import os
import cv2
import csv
import time
import argparse
import numpy as np
import supervision as sv
from pathlib import Path
from loguru import logger

from config.settings import settings
from src.detector.model import HelmetDetector
from src.detector.tracker import ObjectTracker


class MOTExporter:
    """
    Collects per-frame tracking results and writes them
    to MOT Challenge format text files.

    Output structure:
        output_dir/
            det.txt          - raw detections (no tracker IDs)
            pred.txt         - tracking predictions (with IDs)
            gt.txt           - placeholder for ground truth (user fills)
            seqinfo.ini      - sequence metadata
            frames/          - extracted video frames (optional)
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "frames").mkdir(exist_ok=True)

        self._pred_rows: list[list] = []  # tracking predictions
        self._det_rows: list[list] = []   # raw detections
        self._frame_count = 0
        self._img_width = 0
        self._img_height = 0
        self._fps = 30.0
        self._save_frames = False

        logger.info(f"MOT Exporter initialized: {self.output_dir}")

    def set_video_info(self, width: int, height: int, fps: float):
        """Store video metadata for seqinfo.ini."""
        self._img_width = width
        self._img_height = height
        self._fps = fps

    def enable_frame_save(self, enable: bool = True):
        """Enable saving individual frames as images (for annotation tool)."""
        self._save_frames = enable

    def record_frame(
        self,
        frame_id: int,
        frame: np.ndarray,
        raw_detections: sv.Detections,
        tracked_detections: sv.Detections,
    ):
        """
        Record one frame's detections and tracking results.

        Args:
            frame_id: 1-based frame number
            frame: the raw BGR frame
            raw_detections: detections BEFORE tracking
            tracked_detections: detections AFTER tracking (with tracker_id)
        """
        self._frame_count = max(self._frame_count, frame_id)

        # Save frame image if enabled
        if self._save_frames:
            frame_path = self.output_dir / "frames" / f"{frame_id:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

        # Record raw detections
        if len(raw_detections) > 0:
            for i in range(len(raw_detections)):
                x1, y1, x2, y2 = raw_detections.xyxy[i]
                bb_w = x2 - x1
                bb_h = y2 - y1
                conf = float(raw_detections.confidence[i])
                cls_id = int(raw_detections.class_id[i])
                # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
                self._det_rows.append([
                    frame_id, -1,
                    round(float(x1), 2), round(float(y1), 2),
                    round(float(bb_w), 2), round(float(bb_h), 2),
                    round(conf, 4), cls_id, -1, -1
                ])

        # Record tracked detections
        if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                tid = tracked_detections.tracker_id[i]
                if tid is None:
                    continue
                x1, y1, x2, y2 = tracked_detections.xyxy[i]
                bb_w = x2 - x1
                bb_h = y2 - y1
                conf = float(tracked_detections.confidence[i])
                cls_id = int(tracked_detections.class_id[i])
                self._pred_rows.append([
                    frame_id, int(tid),
                    round(float(x1), 2), round(float(y1), 2),
                    round(float(bb_w), 2), round(float(bb_h), 2),
                    round(conf, 4), cls_id, -1, -1
                ])

    def save(self):
        """Write all collected data to MOT format files."""
        # Write predictions (tracking output)
        pred_path = self.output_dir / "pred.txt"
        with open(pred_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in sorted(self._pred_rows, key=lambda r: (r[0], r[1])):
                writer.writerow(row)
        logger.info(f"Saved {len(self._pred_rows)} tracking predictions -> {pred_path}")

        # Write raw detections
        det_path = self.output_dir / "det.txt"
        with open(det_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in sorted(self._det_rows, key=lambda r: r[0]):
                writer.writerow(row)
        logger.info(f"Saved {len(self._det_rows)} raw detections -> {det_path}")

        # Write placeholder GT file
        gt_path = self.output_dir / "gt.txt"
        if not gt_path.exists():
            gt_path.touch()
            logger.info(f"Created empty GT file -> {gt_path}")
            logger.info("  Use gt_annotator.py to annotate ground truth")

        # Write seqinfo.ini
        seq_path = self.output_dir / "seqinfo.ini"
        with open(seq_path, "w") as f:
            f.write("[Sequence]\n")
            f.write(f"name={self.output_dir.name}\n")
            f.write(f"imDir=frames\n")
            f.write(f"frameRate={int(self._fps)}\n")
            f.write(f"seqLength={self._frame_count}\n")
            f.write(f"imWidth={self._img_width}\n")
            f.write(f"imHeight={self._img_height}\n")
            f.write(f"imExt=.jpg\n")
        logger.info(f"Saved sequence info -> {seq_path}")

        # Summary
        unique_tracks = len(set(r[1] for r in self._pred_rows))
        logger.info(
            f"MOT Export Summary: {self._frame_count} frames, "
            f"{unique_tracks} unique tracks, "
            f"{len(self._pred_rows)} track entries"
        )

    @property
    def stats(self) -> dict:
        return {
            "frames": self._frame_count,
            "prediction_entries": len(self._pred_rows),
            "detection_entries": len(self._det_rows),
            "unique_tracks": len(set(r[1] for r in self._pred_rows)),
        }


def run_export(source: str, output_dir: str, save_frames: bool = True):
    """
    Standalone MOT export: run detection + tracking and export results.

    Args:
        source: video file path
        output_dir: directory to write MOT files
        save_frames: save individual frames for annotation
    """
    logger.info(f"Running standalone MOT export: {source} -> {output_dir}")

    detector = HelmetDetector()
    exporter = MOTExporter(output_dir)
    exporter.enable_frame_save(save_frames)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    exporter.set_video_info(width, height, fps)
    tracker = ObjectTracker(frame_rate=int(fps))

    frame_id = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        detections = detector.predict(frame)
        tracked = tracker.update(detections)
        exporter.record_frame(frame_id, frame, detections, tracked)

        if frame_id % 100 == 0:
            elapsed = time.time() - start
            pct = frame_id / total * 100 if total > 0 else 0
            logger.info(f"Processed {frame_id}/{total} ({pct:.1f}%) - {elapsed:.1f}s")

    cap.release()
    exporter.save()

    elapsed = time.time() - start
    logger.info(f"MOT export complete in {elapsed:.1f}s")
    return exporter.stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export tracking results to MOT format")
    parser.add_argument("--source", type=str, required=True, help="Video file path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--no-frames", action="store_true",
        help="Skip saving individual frames (faster, less disk space)"
    )
    args = parser.parse_args()

    run_export(args.source, args.output, save_frames=not args.no_frames)
