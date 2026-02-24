"""
ITERA Smart Sentinel — System Resource Benchmark (CPU & Memory Profiler)

Measures CPU and memory utilization during the detection pipeline execution
across three separate components:
1. The Pipeline itself (Python)
2. Apache Kafka (Java)
3. Apache Spark (Java)

Usage:
    uv run python tests/system_resource_benchmark.py --source test.mp4 --frames 3000
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


def find_process_by_cmdline(keywords: list[str]) -> psutil.Process | None:
    """Find a running process whose cmdline contains all the given keywords."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline:
                # Convert cmdline list to a single string for easier searching
                cmd_str = " ".join(cmdline).lower()
                if all(k.lower() in cmd_str for k in keywords):
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None


class SystemResourceMonitor:
    """Background thread that samples CPU and memory usage of multiple processes."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        
        # Identify processes
        self.procs = {}
        
        # 1. Pipeline (Current Process)
        self.procs['pipeline'] = psutil.Process(os.getpid())
        
        # 2. Kafka (Look for 'kafka.Kafka' in java cmdline)
        kafka_proc = find_process_by_cmdline(['java', 'kafka.kafka'])
        if kafka_proc:
            self.procs['kafka'] = kafka_proc
            logger.info(f"Found Kafka process: PID {kafka_proc.pid}")
        else:
            logger.warning("Kafka process not found! Make sure Kafka is running locally.")
            
        # 3. Spark (Look for 'SparkSubmit' in java cmdline)
        spark_proc = find_process_by_cmdline(['java', 'sparksubmit'])
        if spark_proc:
            self.procs['spark'] = spark_proc
            logger.info(f"Found Spark process: PID {spark_proc.pid}")
        else:
            logger.warning("Spark process not found! Make sure stream_processor.py is running via spark-submit.")

        # Initialize data stores
        self.samples = {
            name: {"cpu": [], "mem": []} for name in ['pipeline', 'kafka', 'spark']
        }

    def start(self):
        # Prime CPU percent for all available processes
        for name, proc in self.procs.items():
            try:
                proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                self._stop_event.wait(self.interval)
                if self._stop_event.is_set() and len(self.samples['pipeline']['cpu']) > 0:
                    break
                
                # Sample each tracked process
                for name, proc in list(self.procs.items()):
                    try:
                        cpu = proc.cpu_percent(interval=None)
                        mem = proc.memory_info().rss / (1024 * 1024)
                        
                        self.samples[name]['cpu'].append(cpu)
                        self.samples[name]['mem'].append(mem)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        # Process died during benchmark
                        logger.warning(f"{name.capitalize()} process died or became inaccessible.")
                        del self.procs[name]
                        
            except Exception as e:
                logger.error(f"Error in resource sampling loop: {e}")
                break

    def stop(self) -> dict:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

        results = {}
        for name in ['pipeline', 'kafka', 'spark']:
            cpu_samples = self.samples[name]['cpu']
            mem_samples = self.samples[name]['mem']
            
            if cpu_samples:
                cpu_avg = round(float(np.mean(cpu_samples)), 2)
                cpu_peak = round(float(max(cpu_samples)), 2)
            else:
                cpu_avg = cpu_peak = 0.0

            if mem_samples:
                mem_avg = round(float(np.mean(mem_samples)), 2)
                mem_peak = round(float(max(mem_samples)), 2)
            else:
                mem_avg = mem_peak = 0.0
                
            results[name] = {
                "samples_collected": len(cpu_samples),
                "cpu_percent": {"avg": cpu_avg, "peak": cpu_peak},
                "memory_mb": {"avg": mem_avg, "peak": mem_peak},
            }

        return results


def run_pipeline_with_resource_monitoring(video_source: str, max_frames: int) -> dict:
    """Run the detection pipeline in-process while monitoring overall system CPU/memory."""
    logger.info("=" * 70)
    logger.info("🖥️  ITERA Smart Sentinel — SYSTEM Resource Benchmark")
    logger.info("=" * 70)

    # Start resource monitor BEFORE heavy components load
    monitor = SystemResourceMonitor(interval=0.5)
    monitor.start()

    # Initialize components
    detector = HelmetDetector()
    tracker = ObjectTracker(frame_rate=15)
    violation_checker = ViolationDetector(confirm_frames=3)
    try:
        from src.streaming.producer import ViolationProducer
        producer = ViolationProducer()
    except Exception as e:
        logger.warning(f"Could not load Kafka Producer: {e}")
        producer = None

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

        frame_start_time = time.time()
        detections = detector.predict(frame)
        tracked = tracker.update(detections)
        violations = violation_checker.check(tracked, frame_num)
        
        # Stream to Kafka just like the main pipeline
        if violations and producer:
            processing_latency = (time.time() - frame_start_time) * 1000
            for v in violations:
                v.processing_latency_ms = processing_latency
            producer.send_batch(violations)

        if frame_num % 500 == 0:
            elapsed = time.time() - start_time
            avg_fps = frame_num / elapsed
            logger.info(f"[{frame_num}/{process_frames}] FPS: {avg_fps:.1f}")

    cap.release()
    if producer:
        producer.close()

    total_time = time.time() - start_time

    # Stop monitor and collect results
    resource_stats = monitor.stop()

    results = {
        "benchmark_info": {
            "video_source": video_source,
            "frames": frame_num,
            "duration_seconds": round(total_time, 2),
            "timestamp": datetime.now().isoformat()
        },
        "resource_usage": resource_stats,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="ITERA Smart Sentinel — System Resource Benchmark")
    parser.add_argument("--source", type=str, default="test.mp4", help="Video source")
    parser.add_argument("--frames", type=int, default=3000, help="Max frames (0=all)")
    parser.add_argument("--output", type=str, default="system_resource_results.json", help="Output JSON file")
    args = parser.parse_args()

    results = run_pipeline_with_resource_monitoring(args.source, args.frames)

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    info = results["benchmark_info"]
    res = results["resource_usage"]
    
    print(f"\n{'=' * 75}")
    print(f"📊 SYSTEM RESOURCE BENCHMARK RESULTS")
    print(f"{'=' * 75}")
    print(f"Video:   {info['video_source']}")
    print(f"Frames:  {info['frames']} | Time: {info['duration_seconds']}s")
    print(f"{'-' * 75}")
    
    for component in ['pipeline', 'kafka', 'spark']:
        stats = res[component]
        cpu = stats["cpu_percent"]
        mem = stats["memory_mb"]
        
        status = "✅ Active" if stats["samples_collected"] > 0 else "❌ Not Found/Tracked"
        
        print(f"🔹 {component.upper()} [{status}] ({stats['samples_collected']} samples)")
        if stats["samples_collected"] > 0:
            print(f"   🖥️  CPU: Avg {cpu['avg']:>6}%  |  Peak {cpu['peak']:>6}%")
            print(f"   💾 MEM: Avg {mem['avg']:>6} MB |  Peak {mem['peak']:>6} MB")
        print(f"{'-' * 75}")
        
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
