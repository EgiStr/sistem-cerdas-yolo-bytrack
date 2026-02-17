"""
Spark Structured Streaming Processor

Consumes violation events from Kafka, deduplicates by track_id
within a time window, enriches with time dimensions, and writes
to PostgreSQL (Supabase) Star Schema.

Usage:
    /opt/spark/bin/spark-submit \
        --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8,org.postgresql:postgresql:42.7.1 \
        spark/stream_processor.py
"""

import os
import sys
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, TimestampType,
)

# Load environment variables (handle Spark's system Python without dotenv)
import pathlib
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual .env parsing for Spark (runs in system Python, not uv venv)
    env_file = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

# ─── Configuration ─────────────────────────────────────────────
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC_VIOLATIONS", "video.violations")

DB_HOST = os.getenv("SUPABASE_DB_HOST", "localhost")
DB_PORT = os.getenv("SUPABASE_DB_PORT", "5432")
DB_NAME = os.getenv("SUPABASE_DB_NAME", "postgres")
DB_USER = os.getenv("SUPABASE_DB_USER", "postgres")
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")

JDBC_URL = f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}"
CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/sentinel-checkpoints")

# ─── Event Schema ──────────────────────────────────────────────
EVENT_SCHEMA = StructType([
    StructField("event_id", StringType(), False),
    StructField("track_id", IntegerType(), False),
    StructField("timestamp", StringType(), False),
    StructField("camera_id", StringType(), False),
    StructField("violation_type", StringType(), False),
    StructField("confidence", FloatType(), False),
    StructField("bbox_x1", IntegerType(), True),
    StructField("bbox_y1", IntegerType(), True),
    StructField("bbox_x2", IntegerType(), True),
    StructField("bbox_y2", IntegerType(), True),
    StructField("frame_number", IntegerType(), True),
    StructField("processing_latency_ms", FloatType(), True),
])


def create_spark_session() -> SparkSession:
    """Create Spark session optimized for 12GB RAM constraint."""
    return (
        SparkSession.builder
        .appName("ITERA-Smart-Sentinel-StreamProcessor")
        .master(os.getenv("SPARK_MASTER", "local[2]"))
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "3g"))
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8,"
                "org.postgresql:postgresql:42.7.1")
        .getOrCreate()
    )


def read_kafka_stream(spark: SparkSession) -> DataFrame:
    """Read violation events from Kafka topic as a streaming DataFrame."""
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", 1000)
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parse JSON from Kafka value
    parsed = (
        raw_stream
        .select(
            F.from_json(
                F.col("value").cast("string"),
                EVENT_SCHEMA
            ).alias("event"),
            F.col("timestamp").alias("kafka_timestamp"),
        )
        .select("event.*", "kafka_timestamp")
        .withColumn(
            "event_timestamp",
            F.to_timestamp(F.col("timestamp"))
        )
    )

    return parsed


def enrich_with_time_dimensions(df: DataFrame) -> DataFrame:
    """
    Add time dimension fields for the Star Schema.

    Extracts hour, day_of_week, date, etc. from event timestamp.
    Also maps to Indonesian time period labels.
    """
    enriched = (
        df
        .withColumn("hour", F.hour("event_timestamp"))
        .withColumn("minute", F.minute("event_timestamp"))
        .withColumn("day_of_week", F.dayofweek("event_timestamp"))
        .withColumn("date", F.to_date("event_timestamp"))
        .withColumn("week_of_year", F.weekofyear("event_timestamp"))
        .withColumn("month", F.month("event_timestamp"))
        .withColumn("year", F.year("event_timestamp"))
        .withColumn(
            "time_period",
            F.when(F.col("hour").between(5, 10), "pagi")
            .when(F.col("hour").between(11, 14), "siang")
            .when(F.col("hour").between(15, 17), "sore")
            .otherwise("malam")
        )
    )

    return enriched


def deduplicate_events(df: DataFrame) -> DataFrame:
    """
    Deduplicate events by (track_id, camera_id) within a 30-second window.

    Uses watermark to handle late-arriving events and dropDuplicates
    for exactly-once semantics per track per camera.
    """
    deduped = (
        df
        .withWatermark("event_timestamp", "30 seconds")
        .dropDuplicatesWithinWatermark(["track_id", "camera_id"])
    )

    return deduped


def write_to_postgres(batch_df: DataFrame, batch_id: int):
    """
    Write a micro-batch to PostgreSQL using JDBC.

    Called by foreachBatch sink for each micro-batch.
    Handles both dim_time and fact_violations tables.
    """
    if batch_df.isEmpty():
        return

    print(f"[Batch {batch_id}] Writing {batch_df.count()} violations to PostgreSQL...")

    jdbc_properties = {
        "user": DB_USER,
        "password": DB_PASSWORD,
        "driver": "org.postgresql.Driver",
    }

    # ── Write to dim_time (upsert via temp table) ──
    time_df = (
        batch_df.select(
            F.col("event_timestamp").alias("full_timestamp"),
            "hour", "minute", "day_of_week", "date",
            "week_of_year", "month", "year", "time_period",
        )
        .distinct()
    )

    time_df.write.jdbc(
        url=JDBC_URL,
        table="dim_time",
        mode="append",
        properties=jdbc_properties,
    )

    # ── Write to fact_violations ──
    # Get the time_id for each event by matching timestamp
    # Since dim_time has serial PK, we write fact with NULL time_id
    # and update via a DB trigger/view, OR embed the time data directly.

    # For simplicity, we write the violation directly with embedded time data.
    fact_df = batch_df.select(
        F.col("event_id").alias("violation_id"),
        F.col("camera_id"),
        F.col("track_id"),
        F.col("violation_type"),
        F.col("confidence"),
        F.col("bbox_x1"),
        F.col("bbox_y1"),
        F.col("bbox_x2"),
        F.col("bbox_y2"),
        F.col("frame_number"),
        F.col("processing_latency_ms"),
        F.col("event_timestamp").alias("created_at"),
        # Embedded time fields (for direct Grafana queries)
        "hour", "minute", "day_of_week", "date",
        "week_of_year", "month", "year", "time_period",
    )

    fact_df.write.jdbc(
        url=JDBC_URL,
        table="fact_violations",
        mode="append",
        properties=jdbc_properties,
    )

    print(f"[Batch {batch_id}] ✅ Written to PostgreSQL successfully")


def main():
    """Entry point for Spark Structured Streaming job."""
    print("=" * 60)
    print("⚡ ITERA Smart Sentinel - Spark Stream Processor")
    print("=" * 60)
    print(f"Kafka: {KAFKA_SERVERS} / {KAFKA_TOPIC}")
    print(f"PostgreSQL: {JDBC_URL}")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print("=" * 60)

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Read from Kafka
    stream = read_kafka_stream(spark)

    # Enrich with time dimensions
    enriched = enrich_with_time_dimensions(stream)

    # Deduplicate
    deduped = deduplicate_events(enriched)

    # Write to PostgreSQL
    query = (
        deduped.writeStream
        .foreachBatch(write_to_postgres)
        .outputMode("append")
        .option("checkpointLocation", CHECKPOINT_DIR)
        .trigger(processingTime="5 seconds")
        .start()
    )

    print("🚀 Stream processor running... Press Ctrl+C to stop")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping stream processor...")
        query.stop()
        spark.stop()
        print("✅ Stream processor stopped")


if __name__ == "__main__":
    main()
