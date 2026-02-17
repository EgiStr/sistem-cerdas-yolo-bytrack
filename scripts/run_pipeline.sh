#!/usr/bin/env bash
# ============================================
# ITERA Smart Sentinel - Full Pipeline Runner
# ============================================
# Starts all services in the correct order.
# Make sure Kafka and Zookeeper are already running.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "🛡️  ITERA Smart Sentinel - Starting Pipeline"
echo "============================================"

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Kafka
if ! /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092 &>/dev/null; then
    echo "❌ Kafka is not running! Start it first:"
    echo "   /opt/kafka/bin/zookeeper-server-start.sh -daemon /opt/kafka/config/zookeeper.properties"
    echo "   /opt/kafka/bin/kafka-server-start.sh -daemon /opt/kafka/config/server.properties"
    exit 1
fi
echo "✅ Kafka is running"

# Check .env
if [ ! -f .env ]; then
    echo "⚠️  .env not found, copying from .env.example"
    cp .env.example .env
    echo "📝 Please edit .env with your Supabase credentials"
fi

# Setup Kafka topics (idempotent)
echo ""
echo "📨 Setting up Kafka topics..."
bash scripts/setup_kafka_topics.sh

echo ""
echo "============================================"
echo "Ready to run!"
echo ""
echo "Usage:"
echo "  Terminal 1 - Start Spark processor:"
echo "    /opt/spark/bin/spark-submit \\"
echo "      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8,org.postgresql:postgresql:42.7.1 \\"
echo "      spark/stream_processor.py"
echo ""
echo "  Terminal 2 - Start detector pipeline:"
echo "    uv run python -m src.pipeline.main --source sample_video.mp4"
echo ""
echo "  Terminal 2 (with display):"
echo "    uv run python -m src.pipeline.main --source sample_video.mp4 --display"
echo ""
echo "  Terminal 2 (without Kafka, testing only):"
echo "    uv run python -m src.pipeline.main --source sample_video.mp4 --no-kafka --display"
echo ""
echo "  Grafana dashboard:"
echo "    http://localhost:3000"
echo "============================================"
