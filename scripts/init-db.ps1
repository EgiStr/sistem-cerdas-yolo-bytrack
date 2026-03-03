# ============================================
# ITERA Smart Sentinel - Database Init (Windows)
# ============================================
# Runs SQL migration scripts against the PostgreSQL container.
# Safe to run multiple times (uses IF NOT EXISTS).
#
# Usage: .\scripts\init-db.ps1

$ErrorActionPreference = "Stop"

$CONTAINER = "sentinel-postgres"
$DB_USER = "postgres"
$DB_NAME = "sentinel_db"

Write-Host ""
Write-Host "ITERA Smart Sentinel - Database Setup" -ForegroundColor Cyan
Write-Host ("=" * 50)

# Check if Postgres container is running and healthy
$running = docker ps --filter "name=$CONTAINER" --filter "health=healthy" --format "{{.Names}}" 2>$null
if ($running -ne $CONTAINER) {
    $anyRunning = docker ps --filter "name=$CONTAINER" --format "{{.Names}}" 2>$null
    if ($anyRunning -ne $CONTAINER) {
        Write-Host "[ERROR] PostgreSQL container '$CONTAINER' is not running!" -ForegroundColor Red
        Write-Host "   Run: docker compose up -d" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "[WAIT] Waiting for PostgreSQL to be healthy..." -ForegroundColor Yellow
    $maxRetries = 15
    $retry = 0
    while ($retry -lt $maxRetries) {
        $healthy = docker ps --filter "name=$CONTAINER" --filter "health=healthy" --format "{{.Names}}" 2>$null
        if ($healthy -eq $CONTAINER) { break }
        $retry++
        Start-Sleep -Seconds 2
    }
    if ($retry -eq $maxRetries) {
        Write-Host "[ERROR] PostgreSQL not healthy after $($maxRetries * 2) seconds" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[OK] PostgreSQL is healthy" -ForegroundColor Green

# Run SQL migration scripts in order
$sqlFiles = @(
    "001_create_dimensions.sql",
    "002_create_facts.sql",
    "003_create_views.sql"
)

foreach ($sqlFile in $sqlFiles) {
    $localPath = "sql/$sqlFile"
    if (-not (Test-Path $localPath)) {
        Write-Host "[SKIP] $sqlFile (file not found)" -ForegroundColor Yellow
        continue
    }

    Write-Host "[RUN] $sqlFile..." -ForegroundColor Yellow
    $sqlContent = Get-Content $localPath -Raw
    $sqlContent | docker exec -i $CONTAINER psql -U $DB_USER -d $DB_NAME

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] $sqlFile applied" -ForegroundColor Green
    }
    else {
        Write-Host "  [FAIL] $sqlFile failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "[DONE] Database schema ready!" -ForegroundColor Green

# Verify tables
Write-Host ""
Write-Host "Verifying tables:" -ForegroundColor Cyan
docker exec $CONTAINER psql -U $DB_USER -d $DB_NAME -c '\dt'
