#!/usr/bin/env bash
# ============================================
# ITERA Smart Sentinel - Kafka Topic Setup
# ============================================
# Creates the required Kafka topics for the pipeline.
# Run this once before starting the pipeline.

set -euo pipefail

KAFKA_BIN="/opt/kafka/bin"
BOOTSTRAP="localhost:9092"

echo "🔧 Creating Kafka topics..."

# Topic for all detection events (high volume, optional)
${KAFKA_BIN}/kafka-topics.sh \
    --create \
    --if-not-exists \
    --topic video.detections \
    --partitions 3 \
    --replication-factor 1 \
    --bootstrap-server ${BOOTSTRAP} \
    --config retention.ms=3600000 \
    --config cleanup.policy=delete

echo "✅ Created topic: video.detections (retention: 1 hour)"

# Topic for confirmed violations (consumed by Spark)
${KAFKA_BIN}/kafka-topics.sh \
    --create \
    --if-not-exists \
    --topic video.violations \
    --partitions 3 \
    --replication-factor 1 \
    --bootstrap-server ${BOOTSTRAP} \
    --config retention.ms=86400000 \
    --config cleanup.policy=delete

echo "✅ Created topic: video.violations (retention: 24 hours)"

# List all topics
echo ""
echo "📋 Current topics:"
${KAFKA_BIN}/kafka-topics.sh --list --bootstrap-server ${BOOTSTRAP}

echo ""
echo "🎉 Kafka topics ready!"
