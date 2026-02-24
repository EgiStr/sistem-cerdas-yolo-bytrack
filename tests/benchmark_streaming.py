"""
ITERA Smart Sentinel — Kafka and Spark End-to-End Benchmark

This script simulates high-volume violation events sent to Kafka,
and measures the end-to-end latency until they are processed by Spark
and appear in the PostgreSQL Data Mart (fact_violations).

Usage:
    uv run python tests/benchmark_streaming.py --messages 1000
"""

import time
import uuid
import json
import argparse
import psycopg2
from datetime import datetime
from loguru import logger

from config.settings import settings
from src.streaming.producer import ViolationProducer
from src.utils.schemas import ViolationEvent, BoundingBox

def check_db_connection() -> psycopg2.extensions.connection:
    """Check if the database is accessible."""
    conn = psycopg2.connect(
        host=settings.supabase_db_host,
        port=settings.supabase_db_port,
        dbname=settings.supabase_db_name,
        user=settings.supabase_db_user,
        password=settings.supabase_db_password,
        sslmode="disable" if settings.supabase_db_host == "localhost" else "require"
    )
    return conn

def run_benchmark(num_messages: int) -> dict:
    logger.info("=" * 70)
    logger.info(f"🚀 ITERA Smart Sentinel — Streaming E2E Benchmark")
    logger.info("=" * 70)
    
    # 1. Connect to DB to clear out old benchmark data if any
    try:
        conn = check_db_connection()
        cur = conn.cursor()
        # Create a unique benchmark ID for this run
        benchmark_run_id = f"benchmark_run_{int(time.time())}"
        logger.info(f"Connected to DB. Benchmark Run ID: {benchmark_run_id}")
    except Exception as e:
        logger.error(f"Failed to connect to Datamart Database: {e}")
        return {"error": str(e)}

    # Initialize Producer
    try:
        producer = ViolationProducer()
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        return {"error": str(e)}

    # 2. Prepare Events
    logger.info(f"Preparing {num_messages} test events...")
    events = []
    # Use a specific track_id offset to avoid deduplication clipping all messages
    base_track_id = int(time.time()) % 10000 
    
    for i in range(num_messages):
        # Foreign key constraint requires a valid camera ID from dim_camera
        # We will use 'gate_utama_01' and use violation_type or confidence to identify our dummy data
        event = ViolationEvent(
            track_id=base_track_id + i,
            camera_id="gate_utama_01",
            violation_type=f"benchmark_test_{benchmark_run_id}",
            confidence=0.85,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            frame_number=i * 5
        )
        events.append(event)
        
    # 3. Fire events into Kafka quickly (measure ingestion throughput)
    logger.info("Firing events into Kafka...")
    t_start_ingest = time.perf_counter()
    
    for event in events:
        producer.send(event)
        
    producer.flush(timeout=30)
    t_end_ingest = time.perf_counter()
    
    ingest_time = t_end_ingest - t_start_ingest
    ingest_fps = num_messages / ingest_time if ingest_time > 0 else 0
    
    logger.info(f"✅ Kafka Ingestion Complete: {num_messages} msgs in {ingest_time:.2f}s ({ingest_fps:.1f} msg/s)")
    
    # 4. Wait for Spark to process and load into DB
    logger.info("Polling PostgreSQL for processed events (End-to-End Latency)...")
    
    t_start_poll = time.perf_counter()
    max_wait_time = 120  # Max wait 2 minutes
    events_processed = 0
    poll_intervals = []
    
    while (time.perf_counter() - t_start_poll) < max_wait_time:
        t_poll_start = time.perf_counter()
        
        # Check count of messages with our specific benchmark_run_id (now in violation_type)
        cur.execute("SELECT COUNT(*) FROM fact_violations WHERE violation_type = %s", (f"benchmark_test_{benchmark_run_id}",))
        count = cur.fetchone()[0]
            
        poll_intervals.append(time.perf_counter() - t_poll_start)
        
        if count > events_processed:
            logger.info(f"  Processed {count}/{num_messages} events...")
            events_processed = count
            
        if count >= num_messages:
            break
            
        time.sleep(2.0)  # check every 2 seconds
        
    t_end_poll = time.perf_counter()
    e2e_duration = t_end_poll - t_start_poll
    
    # Clean up dummy data
    logger.info("Cleaning up benchmark data from database...")
    cur.execute("DELETE FROM fact_violations WHERE violation_type = %s", (f"benchmark_test_{benchmark_run_id}",))
    conn.commit()
    cur.close()
    conn.close()

    results = {
        "benchmark_id": benchmark_run_id,
        "messages_sent": num_messages,
        "kafka_ingestion": {
            "duration_seconds": round(ingest_time, 2),
            "throughput_msg_per_sec": round(ingest_fps, 1)
        },
        "spark_processing": {
            "messages_processed": events_processed,
            "success_rate": round((events_processed / num_messages) * 100, 1) if num_messages > 0 else 0,
            "end_to_end_duration_seconds": round(e2e_duration, 2),
            "avg_e2e_latency_seconds": round(e2e_duration, 2) if events_processed >= num_messages else "> 120 (Timeout)"
        }
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ITERA Smart Sentinel Streaming E2E Benchmark")
    parser.add_argument("--messages", type=int, default=1000)
    args = parser.parse_args()
    
    res = run_benchmark(args.messages)
    
    print("\n" + "="*50)
    print("📊 STREAMING E2E BENCHMARK RESULTS")
    print("="*50)
    print(json.dumps(res, indent=2))
    print("="*50)
    
    with open("streaming_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved to streaming_results.json")
