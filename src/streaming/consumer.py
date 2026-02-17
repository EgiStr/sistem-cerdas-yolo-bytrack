"""
Kafka Consumer for debugging/testing

Reads and prints violation events from Kafka topic for debugging.

Usage:
    uv run python -m src.streaming.consumer
"""

import json
import signal
import sys
from confluent_kafka import Consumer, KafkaError
from loguru import logger
from config.settings import settings


def main():
    consumer = Consumer({
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "group.id": "sentinel-debug-consumer",
        "auto.offset.reset": "latest",
    })

    topic = settings.kafka_topic_violations
    consumer.subscribe([topic])

    logger.info(f"🎧 Listening on topic: {topic}")
    logger.info("Press Ctrl+C to stop")

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    msg_count = 0
    try:
        while running:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                logger.error(f"Kafka error: {msg.error()}")
                continue

            msg_count += 1
            value = json.loads(msg.value().decode("utf-8"))
            logger.info(
                f"📩 [{msg_count}] Violation received:\n"
                f"   Track ID:   {value.get('track_id')}\n"
                f"   Type:       {value.get('violation_type')}\n"
                f"   Confidence: {value.get('confidence', 0):.2f}\n"
                f"   Camera:     {value.get('camera_id')}\n"
                f"   Timestamp:  {value.get('timestamp')}\n"
                f"   Frame:      {value.get('frame_number')}"
            )

    finally:
        consumer.close()
        logger.info(f"Consumer closed. Total messages received: {msg_count}")


if __name__ == "__main__":
    main()
