# ============================================
# ITERA Smart Sentinel - Kafka Topic Setup (Windows)
# ============================================
# Creates the required Kafka topics using Docker.
# Run this once after `docker compose up -d`.
#
# Usage: .\scripts\setup_kafka_topics.ps1

$ErrorActionPreference = "Stop"

$CONTAINER = "sentinel-kafka"
$BOOTSTRAP = "localhost:9092"

Write-Host ""
Write-Host "Creating Kafka topics..." -ForegroundColor Cyan

# Check if Kafka container is running
$running = docker ps --filter "name=$CONTAINER" --format "{{.Names}}" 2>$null
if ($running -ne $CONTAINER) {
    Write-Host "[ERROR] Kafka container '$CONTAINER' is not running!" -ForegroundColor Red
    Write-Host "   Run: docker compose up -d" -ForegroundColor Yellow
    exit 1
}

# Wait for Kafka to be ready
Write-Host "[WAIT] Waiting for Kafka broker to be ready..." -ForegroundColor Yellow
$maxRetries = 10
$retry = 0
while ($retry -lt $maxRetries) {
    $result = docker exec $CONTAINER kafka-broker-api-versions --bootstrap-server $BOOTSTRAP 2>$null
    if ($LASTEXITCODE -eq 0) { break }
    $retry++
    Start-Sleep -Seconds 3
}
if ($retry -eq $maxRetries) {
    Write-Host "[ERROR] Kafka broker not ready after $($maxRetries * 3) seconds" -ForegroundColor Red
    exit 1
}

# Topic: video.detections (high volume, optional)
docker exec $CONTAINER kafka-topics --create `
    --if-not-exists `
    --topic video.detections `
    --partitions 3 `
    --replication-factor 1 `
    --bootstrap-server $BOOTSTRAP `
    --config retention.ms=3600000 `
    --config cleanup.policy=delete

Write-Host "[OK] Created topic: video.detections (retention: 1 hour)" -ForegroundColor Green

# Topic: video.violations (consumed by Spark)
docker exec $CONTAINER kafka-topics --create `
    --if-not-exists `
    --topic video.violations `
    --partitions 3 `
    --replication-factor 1 `
    --bootstrap-server $BOOTSTRAP `
    --config retention.ms=86400000 `
    --config cleanup.policy=delete

Write-Host "[OK] Created topic: video.violations (retention: 24 hours)" -ForegroundColor Green

# List all topics
Write-Host ""
Write-Host "Current topics:" -ForegroundColor Cyan
docker exec $CONTAINER kafka-topics --list --bootstrap-server $BOOTSTRAP

Write-Host ""
Write-Host "[DONE] Kafka topics ready!" -ForegroundColor Green
