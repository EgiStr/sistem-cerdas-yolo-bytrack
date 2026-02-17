"""
Kafka Producer for Violation Events

Sends confirmed violation events as JSON messages to Kafka topics.
Uses confluent-kafka (librdkafka) for high performance.
"""

import json
from confluent_kafka import Producer
from loguru import logger
from config.settings import settings
from src.utils.schemas import ViolationEvent


class ViolationProducer:
    """Kafka producer for streaming violation events."""

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        topic: str | None = None,
    ):
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_topic_violations

        self._producer = Producer({
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": "sentinel-detector",
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 100,
            "linger.ms": 5,  # Batch within 5ms for throughput
            "batch.num.messages": 100,
            "compression.type": "snappy",
        })

        logger.info(
            f"Kafka Producer initialized: servers={self.bootstrap_servers}, "
            f"topic={self.topic}"
        )

    def _delivery_callback(self, err, msg):
        """Callback for message delivery confirmation."""
        if err is not None:
            logger.error(f"Kafka delivery failed: {err}")
        else:
            logger.debug(
                f"Kafka msg delivered: topic={msg.topic()}, "
                f"partition={msg.partition()}, offset={msg.offset()}"
            )

    def send(self, violation: ViolationEvent):
        """
        Send a violation event to Kafka.

        Args:
            violation: ViolationEvent to serialize and send
        """
        try:
            payload = json.dumps(violation.to_kafka_dict()).encode("utf-8")
            self._producer.produce(
                topic=self.topic,
                key=str(violation.track_id).encode("utf-8"),
                value=payload,
                callback=self._delivery_callback,
            )
            self._producer.poll(0)  # Trigger delivery callbacks

        except BufferError:
            logger.warning("Kafka producer queue full, flushing...")
            self._producer.flush(timeout=5)
            # Retry once
            self._producer.produce(
                topic=self.topic,
                key=str(violation.track_id).encode("utf-8"),
                value=payload,
                callback=self._delivery_callback,
            )

        except Exception as e:
            logger.error(f"Failed to send violation to Kafka: {e}")

    def send_batch(self, violations: list[ViolationEvent]):
        """Send multiple violation events."""
        for v in violations:
            self.send(v)

    def flush(self, timeout: float = 10.0):
        """Flush all pending messages."""
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning(f"{remaining} messages still in queue after flush")

    def close(self):
        """Flush and cleanup."""
        self.flush()
        logger.info("Kafka Producer closed")
