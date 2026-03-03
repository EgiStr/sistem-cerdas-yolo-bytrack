# ============================================
# ITERA Smart Sentinel - Full Pipeline Runner (Windows)
# ============================================
# Checks prerequisites, sets up infrastructure, and prints
# instructions for starting each component.
#
# Usage: .\scripts\run_pipeline.ps1

$ErrorActionPreference = "Stop"

Write-Host "`n🛡️  ITERA Smart Sentinel - Pipeline Setup" -ForegroundColor Cyan
Write-Host "=" * 50

# ── Step 1: Check Docker ────────────────────────────────────
Write-Host "`n🔍 Checking prerequisites..." -ForegroundColor Yellow

$dockerVersion = docker --version 2>$null
if (-not $dockerVersion) {
    Write-Host "❌ Docker is not installed or not in PATH!" -ForegroundColor Red
    Write-Host "   Install Docker Desktop: https://docs.docker.com/desktop/install/windows/" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green

# ── Step 2: Check .env ──────────────────────────────────────
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  .env not found, copying from .env.example" -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "📝 Please edit .env with your settings" -ForegroundColor Yellow
}
Write-Host "✅ .env file exists" -ForegroundColor Green

# ── Step 3: Start Docker services ───────────────────────────
Write-Host "`n🐳 Starting Docker services..." -ForegroundColor Yellow
docker compose up -d

# Wait for services to be healthy
Write-Host "`n⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "`n📊 Service status:" -ForegroundColor Cyan
docker compose ps

# ── Step 4: Setup Kafka topics ──────────────────────────────
Write-Host ""
& "$PSScriptRoot\setup_kafka_topics.ps1"

# ── Step 5: Init database ──────────────────────────────────
Write-Host ""
& "$PSScriptRoot\init-db.ps1"

# ── Step 6: Print instructions ──────────────────────────────
Write-Host "`n" -NoNewline
Write-Host "=" * 50
Write-Host "🚀 Infrastructure Ready!" -ForegroundColor Green
Write-Host "=" * 50
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Terminal 1 - Start Spark stream processor:" -ForegroundColor White
Write-Host "    .\scripts\spark_submit.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Terminal 2 - Start detector pipeline:" -ForegroundColor White
Write-Host "    uv run python -m src.pipeline.main --source sample_video.mp4" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Terminal 2 (with display):" -ForegroundColor White
Write-Host "    uv run python -m src.pipeline.main --source sample_video.mp4 --display" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Terminal 2 (without Kafka, testing only):" -ForegroundColor White
Write-Host "    uv run python -m src.pipeline.main --source sample_video.mp4 --no-kafka --display" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Grafana dashboard:" -ForegroundColor White
Write-Host "    http://localhost:3000  (admin / admin)" -ForegroundColor Yellow
Write-Host ""
Write-Host "  Spark Master UI:" -ForegroundColor White
Write-Host "    http://localhost:8080" -ForegroundColor Yellow
Write-Host ""
Write-Host "=" * 50
