# ============================================
# ITERA Smart Sentinel - Spark Submit (Windows)
# ============================================
# Submits the Spark stream processor job via Docker.
# Runs in foreground so you can see logs.
#
# Usage: .\scripts\spark_submit.ps1

$ErrorActionPreference = "Stop"

$CONTAINER = "sentinel-spark-master"

Write-Host ""
Write-Host "ITERA Smart Sentinel - Spark Submit" -ForegroundColor Cyan
Write-Host ("=" * 50)

# Check if Spark master container is running
$running = docker ps --filter "name=$CONTAINER" --format "{{.Names}}" 2>$null
if ($running -ne $CONTAINER) {
    Write-Host "[ERROR] Spark master container '$CONTAINER' is not running!" -ForegroundColor Red
    Write-Host "   Run: docker compose up -d" -ForegroundColor Yellow
    exit 1
}

Write-Host "[RUN] Submitting stream_processor.py to Spark..." -ForegroundColor Yellow
Write-Host "      Press Ctrl+C to stop" -ForegroundColor DarkGray

docker exec $CONTAINER spark-submit `
    --master local[2] `
    --driver-memory 2g `
    --packages "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,org.postgresql:postgresql:42.7.1" `
    --conf "spark.sql.shuffle.partitions=4" `
    --conf "spark.jars.ivy=/tmp/.ivy2" `
    /app/spark/stream_processor.py
