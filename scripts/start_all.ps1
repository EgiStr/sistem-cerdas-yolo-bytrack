# ============================================
# ITERA Smart Sentinel - Run Full Pipeline (Windows)
# ============================================
# Starts BOTH Spark stream processor and detection pipeline.
# Spark runs in background (Docker), detection runs in foreground.
#
# Usage:
#   .\scripts\start_all.ps1                          # default: test.mp4
#   .\scripts\start_all.ps1 -Video "my_video.mp4"    # custom video
#   .\scripts\start_all.ps1 -Display                  # with display window

param(
    [string]$Video = "test.mp4",
    [switch]$Display,
    [switch]$NoKafka
)

$ErrorActionPreference = "Stop"

# Add uv to PATH if needed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $env:Path = "C:\Users\egicu\.local\bin;$env:Path"
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  ITERA Smart Sentinel - Full Pipeline" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# --- Step 1: Verify Docker services ---
Write-Host "[1/4] Checking Docker services..." -ForegroundColor Yellow
$services = @("sentinel-postgres", "sentinel-kafka", "sentinel-spark-master")
foreach ($svc in $services) {
    $running = docker ps --filter "name=$svc" --format "{{.Names}}" 2>$null
    if ($running -ne $svc) {
        Write-Host "  [ERROR] $svc is not running!" -ForegroundColor Red
        Write-Host "  Run: docker compose up -d" -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "  [OK] All Docker services running" -ForegroundColor Green

# --- Step 2: Check video file ---
Write-Host "[2/4] Checking video source: $Video" -ForegroundColor Yellow
if (-not (Test-Path $Video)) {
    Write-Host "  [ERROR] Video file '$Video' not found!" -ForegroundColor Red
    exit 1
}
Write-Host "  [OK] Video file found" -ForegroundColor Green

# --- Step 3: Start Spark in background ---
Write-Host "[3/4] Starting Spark stream processor (background)..." -ForegroundColor Yellow

# Check if Spark job is already running
$sparkRunning = docker exec sentinel-spark-master ps aux 2>$null | Select-String "stream_processor.py"
if ($sparkRunning) {
    Write-Host "  [OK] Spark already running" -ForegroundColor Green
} else {
    docker exec -d sentinel-spark-master spark-submit `
        --master local[2] `
        --driver-memory 2g `
        --packages "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,org.postgresql:postgresql:42.7.1" `
        --conf "spark.sql.shuffle.partitions=4" `
        --conf "spark.jars.ivy=/tmp/.ivy2" `
        /app/spark/stream_processor.py

    Write-Host "  [OK] Spark submitted" -ForegroundColor Green
    Write-Host "  Waiting 15s for Spark to initialize..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 15
}

# --- Step 4: Start detection pipeline ---
Write-Host "[4/4] Starting detection pipeline..." -ForegroundColor Yellow
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  Pipeline starting! Press Ctrl+C to stop" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

$args_list = @("run", "python", "-m", "src.pipeline.main", "--source", $Video)

if ($Display) {
    $args_list += "--display"
}

if ($NoKafka) {
    $args_list += "--no-kafka"
}

& uv @args_list
