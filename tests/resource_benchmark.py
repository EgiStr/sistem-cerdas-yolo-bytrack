"""
ITERA Smart Sentinel — Resource Benchmark (CPU & Memory Profiler)

Measures CPU and memory utilization during the detection pipeline execution.
Uses psutil to monitor the CURRENT process (in-process measurement for accuracy).

Usage:
    uv run python tests/resource_benchmark.py --source test.mp4 --frames 3000
    uv run python tests/resource_benchmark.py --source test.mp4 --frames 0  # all frames
"""

import cv2
import time
import json
import argparse
import threading
import numpy as np
import psutil
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from loguru import logger

from config.settings import settings
from src.detector.model import HelmetDetector
from src.detector.tracker import ObjectTracker
from src.detector.violation import ViolationDetector


class ResourceMonitor:
    """Background thread that samples CPU and memory usage of the current process."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.cpu_samples: list[float] = []
        self.mem_samples_mb: list[float] = []
        self._process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._process.cpu_percent()  # prime first call
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        # We primed it with interval=None on start
        while not self._stop_event.is_set():
            try:
                self._stop_event.wait(self.interval)
                if self._stop_event.is_set() and len(self.cpu_samples) > 0:
                    break
                
                cpu = self._process.cpu_percent(interval=None)
                mem = self._process.memory_info().rss / (1024 * 1024)
                self.cpu_samples.append(cpu)
                self.mem_samples_mb.append(mem)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

        if self.cpu_samples:
            cpu_avg = round(float(np.mean(self.cpu_samples)), 2)
            cpu_peak = round(float(max(self.cpu_samples)), 2)
        else:
            cpu_avg = cpu_peak = 0.0

        if self.mem_samples_mb:
            mem_avg = round(float(np.mean(self.mem_samples_mb)), 2)
            mem_peak = round(float(max(self.mem_samples_mb)), 2)
        else:
            mem_avg = mem_peak = 0.0

        return {
            "samples_collected": len(self.cpu_samples),
            "cpu_percent": {"avg": cpu_avg, "peak": cpu_peak},
            "memory_mb": {"avg": mem_avg, "peak": mem_peak},
        }


def run_pipeline_with_resource_monitoring(video_source: str, max_frames: int) -> dict:
    """Run the detection pipeline in-process while monitoring CPU/memory."""
    logger.info("=" * 70)
    logger.info("🖥️  ITERA Smart Sentinel — Resource Benchmark")
    logger.info("=" * 70)

    # Start resource monitor
    monitor = ResourceMonitor(interval=0.5)
    monitor.start()

    # Initialize components
    detector = HelmetDetector()
    tracker = ObjectTracker(frame_rate=15)
    violation_checker = ViolationDetector(confirm_frames=3)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        monitor.stop()
        raise RuntimeError(f"Cannot open: {video_source}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    process_frames = max_frames if max_frames > 0 else total_video_frames
    logger.info(f"Video: {width}x{height} @ {video_fps} FPS, {total_video_frames} total frames")
    logger.info(f"Processing: {process_frames} frames")
    logger.info(f"Model: {settings.model_path} | Device: {detector.device}")
    logger.info("=" * 70)

    frame_num = 0
    start_time = time.time()

    while frame_num < process_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        detections = detector.predict(frame)
        tracked = tracker.update(detections)
        _ = violation_checker.check(tracked, frame_num)

        if frame_num % 500 == 0:
            elapsed = time.time() - start_time
            avg_fps = frame_num / elapsed
            logger.info(f"[{frame_num}/{process_frames}] FPS: {avg_fps:.1f}")

    cap.release()
    total_time = time.time() - start_time

    # Stop monitor and collect results
    resource_stats = monitor.stop()

    results = {
        "benchmark_info": {
            "video_source": video_source,
            "frames": frame_num,
            "duration_seconds": round(total_time, 2)
        },
        "cpu_percent": resource_stats["cpu_percent"],
        "memory_mb": resource_stats["memory_mb"]
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="ITERA Smart Sentinel — Resource Benchmark")
    parser.add_argument("--source", type=str, default="test.mp4", help="Video source")
    parser.add_argument("--frames", type=int, default=3000, help="Max frames (0=all)")
    parser.add_argument("--output", type=str, default="resource_results.json", help="Output JSON file")
    args = parser.parse_args()

    results = run_pipeline_with_resource_monitoring(args.source, args.frames)

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    info = results["benchmark_info"]
    cpu = results["cpu_percent"]
    mem = results["memory_mb"]
    print(f"\n{'=' * 60}")
    print(f"📊 RESOURCE BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"Video:   {info['video_source']}")
    print(f"Frames:  {info['frames']} | Time: {info['duration_seconds']}s")
    print(f"\n🖥️  CPU Usage:    Avg {cpu['avg']}%  |  Peak {cpu['peak']}%")
    print(f"💾 Memory Usage: Avg {mem['avg']} MB  |  Peak {mem['peak']} MB")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
