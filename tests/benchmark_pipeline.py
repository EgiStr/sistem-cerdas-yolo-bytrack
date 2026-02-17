"""
ITERA Smart Sentinel — Comprehensive Benchmark Suite

Runs the full pipeline on test.mp4 and collects ALL metrics needed for Bab 4:
  1. Per-component latency (inference, tracking, violation logic)
  2. FPS throughput over time
  3. Per-class detection distribution
  4. Violation detection count & confidence
  5. Tracking stability (unique track IDs, active tracks over time)
  6. N-frame threshold comparison (N=1, N=3, N=5)

Usage:
    uv run python tests/benchmark_pipeline.py --source test.mp4 --frames 3000
    uv run python tests/benchmark_pipeline.py --source test.mp4 --frames 0  # all frames
"""

import cv2
import time
import json
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from loguru import logger

from config.settings import settings
from src.detector.model import HelmetDetector
from src.detector.tracker import ObjectTracker
from src.detector.violation import ViolationDetector


class PipelineBenchmark:
    """Benchmarks the full detection + tracking + violation pipeline."""

    def __init__(self, video_source: str, max_frames: int = 3000):
        self.video_source = video_source
        self.max_frames = max_frames

        # ── Metrics storage ──
        self.latency_inference: list[float] = []
        self.latency_tracking: list[float] = []
        self.latency_violation: list[float] = []
        self.latency_total: list[float] = []
        self.fps_per_frame: list[float] = []

        # Per-class counts per frame
        self.class_counts: dict[str, list[int]] = defaultdict(list)

        # Detection counts per frame
        self.detections_per_frame: list[int] = []

        # Track ID tracking
        self.all_track_ids: set[int] = set()
        self.active_tracks_per_frame: list[int] = []

        # Violation data
        self.violations: list[dict] = []

    def run(self) -> dict:
        """Run benchmark and return complete results dict."""
        logger.info("=" * 70)
        logger.info("🔬 ITERA Smart Sentinel — Pipeline Benchmark")
        logger.info("=" * 70)

        # Initialize components
        detector = HelmetDetector()
        tracker = ObjectTracker(frame_rate=15)
        violation_checker = ViolationDetector(confirm_frames=3)

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {self.video_source}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        process_frames = self.max_frames if self.max_frames > 0 else total_video_frames
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
            t_total_start = time.perf_counter()

            # ── Step 1: Inference ──
            t0 = time.perf_counter()
            detections = detector.predict(frame)
            t_inference = (time.perf_counter() - t0) * 1000

            # ── Step 2: Tracking ──
            t0 = time.perf_counter()
            tracked = tracker.update(detections)
            t_tracking = (time.perf_counter() - t0) * 1000

            # ── Step 3: Violation check ──
            t0 = time.perf_counter()
            violations = violation_checker.check(tracked, frame_num)
            t_violation = (time.perf_counter() - t0) * 1000

            t_total = (time.perf_counter() - t_total_start) * 1000
            fps = 1000.0 / t_total if t_total > 0 else 0

            # ── Record metrics ──
            self.latency_inference.append(t_inference)
            self.latency_tracking.append(t_tracking)
            self.latency_violation.append(t_violation)
            self.latency_total.append(t_total)
            self.fps_per_frame.append(fps)

            # Count detections per class
            self.detections_per_frame.append(len(detections))
            class_names_in_frame = defaultdict(int)
            if len(detections) > 0 and "class_name" in detections.data:
                for cn in detections.data["class_name"]:
                    class_names_in_frame[cn] += 1
            for cls in ["DRIVER_HELMET", "DRIVER_NO_HELMET", "MOTORCYCLE"]:
                self.class_counts[cls].append(class_names_in_frame.get(cls, 0))

            # Track IDs
            if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
                current_ids = set(int(tid) for tid in tracked.tracker_id if tid is not None)
                self.all_track_ids.update(current_ids)
                self.active_tracks_per_frame.append(len(current_ids))
            else:
                self.active_tracks_per_frame.append(0)

            # Violations
            for v in violations:
                self.violations.append(v.to_kafka_dict())

            # Progress log
            if frame_num % 500 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_num / elapsed
                logger.info(
                    f"[{frame_num}/{process_frames}] "
                    f"FPS: {avg_fps:.1f} | "
                    f"Inf: {np.mean(self.latency_inference[-500:]):.1f}ms | "
                    f"Track: {np.mean(self.latency_tracking[-500:]):.1f}ms | "
                    f"Violations: {len(self.violations)}"
                )

        cap.release()
        total_time = time.time() - start_time

        # ── Compile results ──
        results = self._compile_results(frame_num, total_time, video_fps, width, height, detector.device)
        results["violation_checker_stats"] = violation_checker.stats

        return results

    def _compile_results(self, frames_processed, total_time, video_fps, width, height, device) -> dict:
        """Compile all metrics into a structured results dict."""
        def percentile(data, p):
            return float(np.percentile(data, p)) if data else 0

        def mean(data):
            return float(np.mean(data)) if data else 0

        def stdev(data):
            return float(np.std(data)) if data else 0

        results = {
            "benchmark_info": {
                "video_source": self.video_source,
                "resolution": f"{width}x{height}",
                "video_fps": video_fps,
                "frames_processed": frames_processed,
                "total_time_seconds": round(total_time, 2),
                "device": device,
                "timestamp": datetime.now().isoformat(),
            },
            "throughput": {
                "fps_mean": round(mean(self.fps_per_frame), 2),
                "fps_median": round(percentile(self.fps_per_frame, 50), 2),
                "fps_min": round(min(self.fps_per_frame) if self.fps_per_frame else 0, 2),
                "fps_max": round(max(self.fps_per_frame) if self.fps_per_frame else 0, 2),
                "fps_p5": round(percentile(self.fps_per_frame, 5), 2),
                "fps_p95": round(percentile(self.fps_per_frame, 95), 2),
                "fps_stdev": round(stdev(self.fps_per_frame), 2),
            },
            "latency_ms": {
                "inference": {
                    "mean": round(mean(self.latency_inference), 2),
                    "median": round(percentile(self.latency_inference, 50), 2),
                    "p95": round(percentile(self.latency_inference, 95), 2),
                    "p99": round(percentile(self.latency_inference, 99), 2),
                    "min": round(min(self.latency_inference) if self.latency_inference else 0, 2),
                    "max": round(max(self.latency_inference) if self.latency_inference else 0, 2),
                    "stdev": round(stdev(self.latency_inference), 2),
                },
                "tracking": {
                    "mean": round(mean(self.latency_tracking), 2),
                    "median": round(percentile(self.latency_tracking, 50), 2),
                    "p95": round(percentile(self.latency_tracking, 95), 2),
                    "p99": round(percentile(self.latency_tracking, 99), 2),
                },
                "violation_check": {
                    "mean": round(mean(self.latency_violation), 2),
                    "median": round(percentile(self.latency_violation, 50), 2),
                    "p95": round(percentile(self.latency_violation, 95), 2),
                },
                "total_pipeline": {
                    "mean": round(mean(self.latency_total), 2),
                    "median": round(percentile(self.latency_total, 50), 2),
                    "p95": round(percentile(self.latency_total, 95), 2),
                    "p99": round(percentile(self.latency_total, 99), 2),
                },
            },
            "detections": {
                "total_detections": int(sum(self.detections_per_frame)),
                "avg_detections_per_frame": round(mean(self.detections_per_frame), 2),
                "per_class_total": {
                    cls: int(sum(counts)) for cls, counts in self.class_counts.items()
                },
                "per_class_avg_per_frame": {
                    cls: round(mean(counts), 2) for cls, counts in self.class_counts.items()
                },
            },
            "tracking": {
                "total_unique_track_ids": len(self.all_track_ids),
                "avg_active_tracks_per_frame": round(mean(self.active_tracks_per_frame), 2),
                "max_active_tracks": max(self.active_tracks_per_frame) if self.active_tracks_per_frame else 0,
            },
            "violations": {
                "total_confirmed": len(self.violations),
                "avg_confidence": round(
                    mean([v["confidence"] for v in self.violations]), 3
                ) if self.violations else 0,
                "details": self.violations,
            },
        }

        return results


class ThresholdComparison:
    """Compare violation detection with different N-frame thresholds."""

    def __init__(self, video_source: str, max_frames: int = 3000):
        self.video_source = video_source
        self.max_frames = max_frames

    def run(self) -> dict:
        """Test N=1, N=3, N=5 on the same video segment."""
        logger.info("=" * 70)
        logger.info("🔬 Threshold Temporal Comparison (N=1, N=3, N=5)")
        logger.info("=" * 70)

        results = {}

        for n_threshold in [1, 3, 5]:
            logger.info(f"\n--- Testing N={n_threshold} ---")

            detector = HelmetDetector()
            tracker = ObjectTracker(frame_rate=15)
            violation_checker = ViolationDetector(confirm_frames=n_threshold)

            cap = cv2.VideoCapture(self.video_source)
            process_frames = self.max_frames if self.max_frames > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            violations = []
            frame_num = 0

            while frame_num < process_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1

                detections = detector.predict(frame)
                tracked = tracker.update(detections)
                viol = violation_checker.check(tracked, frame_num)
                violations.extend(viol)

            cap.release()

            results[f"N={n_threshold}"] = {
                "frames_processed": frame_num,
                "total_violations": len(violations),
                "avg_confidence": round(
                    float(np.mean([v.confidence for v in violations])), 3
                ) if violations else 0,
                "unique_track_ids": len(violation_checker._violated_ids),
                "violation_details": [
                    {"track_id": v.track_id, "confidence": round(v.confidence, 3), "frame": v.frame_number}
                    for v in violations
                ],
            }

            logger.info(
                f"N={n_threshold}: {len(violations)} violations, "
                f"avg_conf={results[f'N={n_threshold}']['avg_confidence']}"
            )

        return results


class KafkaBenchmark:
    """Benchmark Kafka producer throughput and latency."""

    def __init__(self, num_messages: int = 1000):
        self.num_messages = num_messages

    def run(self) -> dict:
        """Send test messages to Kafka and measure throughput."""
        from src.streaming.producer import ViolationProducer
        from src.utils.schemas import ViolationEvent, BoundingBox

        logger.info("=" * 70)
        logger.info(f"🔬 Kafka Producer Benchmark ({self.num_messages} messages)")
        logger.info("=" * 70)

        try:
            producer = ViolationProducer()
        except Exception as e:
            logger.warning(f"Kafka not available: {e}")
            return {"error": str(e), "kafka_available": False}

        # Generate test events
        events = []
        for i in range(self.num_messages):
            event = ViolationEvent(
                track_id=i,
                camera_id="benchmark_test",
                violation_type="no_helmet",
                confidence=0.85 + np.random.random() * 0.1,
                bbox=BoundingBox(x1=100, y1=100, x2=300, y2=300),
                frame_number=i * 10,
            )
            events.append(event)

        # Benchmark send
        latencies = []
        t_start = time.time()

        for event in events:
            t0 = time.perf_counter()
            producer.send(event)
            latencies.append((time.perf_counter() - t0) * 1000)

        # Flush remaining
        t_flush_start = time.perf_counter()
        producer.flush(timeout=10)
        flush_time = (time.perf_counter() - t_flush_start) * 1000

        total_time = time.time() - t_start
        producer.close()

        return {
            "kafka_available": True,
            "messages_sent": self.num_messages,
            "total_time_seconds": round(total_time, 3),
            "throughput_msgs_per_sec": round(self.num_messages / total_time, 1),
            "per_message_latency_ms": {
                "mean": round(float(np.mean(latencies)), 3),
                "median": round(float(np.percentile(latencies, 50)), 3),
                "p95": round(float(np.percentile(latencies, 95)), 3),
                "p99": round(float(np.percentile(latencies, 99)), 3),
            },
            "flush_time_ms": round(flush_time, 2),
        }


def print_results_table(results: dict):
    """Pretty-print results for terminal and latex reference."""
    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    info = results.get("benchmark_info", {})
    print(f"\n📹 Video: {info.get('video_source')} ({info.get('resolution')} @ {info.get('video_fps')} FPS)")
    print(f"   Frames: {info.get('frames_processed')} | Device: {info.get('device')}")
    print(f"   Time: {info.get('total_time_seconds')}s")

    # Throughput
    tp = results.get("throughput", {})
    print(f"\n⚡ Throughput:")
    print(f"   FPS Mean:   {tp.get('fps_mean')}")
    print(f"   FPS Median: {tp.get('fps_median')}")
    print(f"   FPS Min:    {tp.get('fps_min')}")
    print(f"   FPS Max:    {tp.get('fps_max')}")
    print(f"   FPS P5:     {tp.get('fps_p5')}")
    print(f"   FPS P95:    {tp.get('fps_p95')}")

    # Latency
    lat = results.get("latency_ms", {})
    print(f"\n⏱️  Latency Breakdown (ms):")
    print(f"   {'Component':<20} {'Mean':>8} {'Median':>8} {'P95':>8} {'P99':>8}")
    print(f"   {'-'*56}")
    for comp_name, comp_key in [
        ("YOLOv8 Inference", "inference"),
        ("ByteTrack", "tracking"),
        ("Violation Logic", "violation_check"),
        ("Total Pipeline", "total_pipeline"),
    ]:
        c = lat.get(comp_key, {})
        print(f"   {comp_name:<20} {c.get('mean', 0):>8.2f} {c.get('median', 0):>8.2f} {c.get('p95', 0):>8.2f} {c.get('p99', 0):>8.2f}")

    # Detections
    det = results.get("detections", {})
    print(f"\n🎯 Detections:")
    print(f"   Total: {det.get('total_detections')} | Avg/frame: {det.get('avg_detections_per_frame')}")
    for cls, count in det.get("per_class_total", {}).items():
        avg = det.get("per_class_avg_per_frame", {}).get(cls, 0)
        print(f"   {cls}: {count} total ({avg} avg/frame)")

    # Tracking
    trk = results.get("tracking", {})
    print(f"\n🔍 Tracking:")
    print(f"   Unique Track IDs: {trk.get('total_unique_track_ids')}")
    print(f"   Avg Active/Frame: {trk.get('avg_active_tracks_per_frame')}")
    print(f"   Max Active:       {trk.get('max_active_tracks')}")

    # Violations
    viol = results.get("violations", {})
    print(f"\n🚨 Violations:")
    print(f"   Total Confirmed: {viol.get('total_confirmed')}")
    print(f"   Avg Confidence:  {viol.get('avg_confidence')}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="ITERA Smart Sentinel — Benchmark Suite")
    parser.add_argument("--source", type=str, default="test.mp4", help="Video source")
    parser.add_argument("--frames", type=int, default=3000, help="Max frames (0=all)")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")
    parser.add_argument("--skip-kafka", action="store_true", help="Skip Kafka benchmark")
    parser.add_argument("--skip-threshold", action="store_true", help="Skip threshold comparison")
    args = parser.parse_args()

    all_results = {}

    # ── 1. Pipeline Benchmark ──
    bench = PipelineBenchmark(args.source, args.frames)
    pipeline_results = bench.run()
    all_results["pipeline"] = pipeline_results
    print_results_table(pipeline_results)

    # ── 2. Threshold Comparison ──
    if not args.skip_threshold:
        thresh = ThresholdComparison(args.source, args.frames)
        threshold_results = thresh.run()
        all_results["threshold_comparison"] = threshold_results

        print("\n" + "=" * 70)
        print("📊 THRESHOLD COMPARISON (N-frame rule)")
        print("=" * 70)
        for key, val in threshold_results.items():
            print(f"   {key}: {val['total_violations']} violations, avg_conf={val['avg_confidence']}")

    # ── 3. Kafka Benchmark ──
    if not args.skip_kafka:
        kafka = KafkaBenchmark(num_messages=500)
        kafka_results = kafka.run()
        all_results["kafka"] = kafka_results

        if kafka_results.get("kafka_available"):
            print(f"\n📨 Kafka: {kafka_results['throughput_msgs_per_sec']} msg/s, "
                  f"latency={kafka_results['per_message_latency_ms']['mean']:.2f}ms")
        else:
            print(f"\n📨 Kafka: Not available ({kafka_results.get('error', 'unknown')})")

    # ── Save results ──
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\n✅ Results saved to: {output_path}")
    logger.info("Use these results to fill in the Bab 4 tables!")


if __name__ == "__main__":
    main()
