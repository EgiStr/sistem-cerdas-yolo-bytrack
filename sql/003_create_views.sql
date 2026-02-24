-- ============================================
-- ITERA Smart Sentinel - Analytical Views
-- Materialized views for Grafana dashboard
-- ============================================
-- Best Practices:
--   1. Each Grafana panel maps to exactly ONE view
--   2. All formatting (ROUND, COALESCE, CASE) lives in the view
--   3. Dashboard SQL is always: SELECT * FROM vw_<name>
--   4. Views are ordered by dashboard layout (top → bottom)
-- ============================================


-- ══════════════════════════════════════════════
-- SECTION 1: KPI Overview
-- ══════════════════════════════════════════════

-- ── Panel 1: KPI Summary (with fallback to all-time) ──
-- Single row: violations_today, avg_confidence, avg_latency, total_all_time
-- Uses COALESCE so confidence/latency fall back to all-time when today = 0
CREATE OR REPLACE VIEW vw_kpi_summary AS
SELECT
    COALESCE(today.total_violations_today, 0)           AS violations_today,
    COALESCE(today.avg_confidence_today,
             alltime.avg_confidence_alltime)             AS avg_confidence,
    COALESCE(today.avg_latency_today,
             alltime.avg_latency_alltime)                AS avg_latency_ms,
    alltime.total_violations_alltime                     AS total_violations_alltime,
    today.last_violation_time
FROM (
    SELECT
        COUNT(*)                    AS total_violations_today,
        AVG(confidence)             AS avg_confidence_today,
        AVG(processing_latency_ms)  AS avg_latency_today,
        MAX(created_at)             AS last_violation_time
    FROM fact_violations
    WHERE date = CURRENT_DATE
) today
CROSS JOIN (
    SELECT
        COUNT(*)                    AS total_violations_alltime,
        AVG(confidence)             AS avg_confidence_alltime,
        AVG(processing_latency_ms)  AS avg_latency_alltime
    FROM fact_violations
) alltime;


-- ══════════════════════════════════════════════
-- SECTION 2: Tren & Distribusi
-- ══════════════════════════════════════════════

-- ── Panel 5: Daily trend (time-series) ──
-- Aggregated per date for Grafana time_series format
CREATE OR REPLACE VIEW vw_daily_trend AS
SELECT
    date                                                AS time,
    COUNT(*)                                            AS violations,
    ROUND(AVG(confidence)::numeric, 3)                  AS avg_confidence,
    ROUND(AVG(processing_latency_ms)::numeric, 1)       AS avg_latency_ms
FROM fact_violations
GROUP BY date
ORDER BY date;


-- ── Panel 6: Camera distribution (bar chart) ──
-- Pre-formatted with camera name fallback and rounded metrics
CREATE OR REPLACE VIEW vw_camera_stats AS
SELECT
    f.camera_id,
    COALESCE(c.camera_name, f.camera_id)                AS camera_name,
    COALESCE(c.gate_name, '-')                          AS gate_name,
    COUNT(*)                                            AS total_violations,
    ROUND(AVG(f.confidence)::numeric, 3)                AS avg_confidence,
    ROUND(AVG(f.processing_latency_ms)::numeric, 1)     AS avg_latency_ms,
    MIN(f.created_at)                                   AS first_violation,
    MAX(f.created_at)                                   AS last_violation
FROM fact_violations f
LEFT JOIN dim_camera c ON f.camera_id = c.camera_id
GROUP BY f.camera_id, c.camera_name, c.gate_name
ORDER BY total_violations DESC;


-- ══════════════════════════════════════════════
-- SECTION 3: Heatmap & Pola
-- ══════════════════════════════════════════════

-- ── Helper view for heatmap (peak hours aggregation) ──
CREATE OR REPLACE VIEW vw_peak_hours AS
SELECT
    day_of_week,
    hour,
    COUNT(*)            AS violation_count,
    AVG(confidence)     AS avg_confidence
FROM fact_violations
GROUP BY day_of_week, hour
ORDER BY day_of_week, hour;


-- ── Panel 7: Heatmap grid (hour × day_of_week) ──
-- Pivoted: rows = hours 0-23, columns = Senin-Minggu
-- Uses generate_series to ensure all 24 hours appear
CREATE OR REPLACE VIEW vw_heatmap_hour_day AS
SELECT
    h.hour                                                                      AS "Jam",
    COALESCE(SUM(CASE WHEN p.day_of_week = 2 THEN p.violation_count END), 0)    AS "Senin",
    COALESCE(SUM(CASE WHEN p.day_of_week = 3 THEN p.violation_count END), 0)    AS "Selasa",
    COALESCE(SUM(CASE WHEN p.day_of_week = 4 THEN p.violation_count END), 0)    AS "Rabu",
    COALESCE(SUM(CASE WHEN p.day_of_week = 5 THEN p.violation_count END), 0)    AS "Kamis",
    COALESCE(SUM(CASE WHEN p.day_of_week = 6 THEN p.violation_count END), 0)    AS "Jumat",
    COALESCE(SUM(CASE WHEN p.day_of_week = 7 THEN p.violation_count END), 0)    AS "Sabtu",
    COALESCE(SUM(CASE WHEN p.day_of_week = 1 THEN p.violation_count END), 0)    AS "Minggu"
FROM generate_series(0, 23) AS h(hour)
LEFT JOIN vw_peak_hours p ON p.hour = h.hour
GROUP BY h.hour
ORDER BY h.hour;


-- ── Panel 8: Time period distribution (pie chart) ──
-- Pivoted single row: Pagi / Siang / Sore / Malam
CREATE OR REPLACE VIEW vw_time_period_distribution AS
SELECT
    COALESCE(SUM(CASE WHEN time_period = 'pagi'  THEN 1 END), 0) AS "Pagi",
    COALESCE(SUM(CASE WHEN time_period = 'siang' THEN 1 END), 0) AS "Siang",
    COALESCE(SUM(CASE WHEN time_period = 'sore'  THEN 1 END), 0) AS "Sore",
    COALESCE(SUM(CASE WHEN time_period = 'malam' THEN 1 END), 0) AS "Malam"
FROM fact_violations;


-- ══════════════════════════════════════════════
-- SECTION 4: Log Detail
-- ══════════════════════════════════════════════

-- ── Panel 9: Recent violations log (table) ──
-- Pre-joined with dim_camera, pre-formatted columns
CREATE OR REPLACE VIEW vw_recent_violations AS
SELECT
    f.created_at                                        AS "Waktu",
    f.camera_id                                         AS "Camera ID",
    COALESCE(c.camera_name, f.camera_id)                AS "Kamera",
    COALESCE(c.gate_name, '-')                          AS "Gate",
    f.violation_type                                    AS "Tipe",
    ROUND(f.confidence::numeric, 3)                     AS "Confidence",
    f.track_id                                          AS "Track ID",
    ROUND(f.processing_latency_ms::numeric, 1)          AS "Latency (ms)",
    f.time_period                                       AS "Periode",
    f.hour                                              AS "Jam"
FROM fact_violations f
LEFT JOIN dim_camera c ON f.camera_id = c.camera_id
ORDER BY f.created_at DESC
LIMIT 100;


-- ══════════════════════════════════════════════
-- SECTION 5: Legacy views (backward compatibility)
-- ══════════════════════════════════════════════

-- ── Hourly violations (used internally by vw_heatmap_hour_day) ──
CREATE OR REPLACE VIEW vw_hourly_violations AS
SELECT
    date,
    hour,
    day_of_week,
    time_period,
    camera_id,
    COUNT(*)                        AS violation_count,
    AVG(confidence)                 AS avg_confidence,
    AVG(processing_latency_ms)      AS avg_latency_ms
FROM fact_violations
GROUP BY date, hour, day_of_week, time_period, camera_id;


-- ── Today's summary (used internally by vw_kpi_summary) ──
CREATE OR REPLACE VIEW vw_today_summary AS
SELECT
    COUNT(*)                        AS total_violations_today,
    AVG(confidence)                 AS avg_confidence_today,
    AVG(processing_latency_ms)      AS avg_latency_today,
    MAX(created_at)                 AS last_violation_time
FROM fact_violations
WHERE date = CURRENT_DATE;
