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
    c.camera_id,
    COALESCE(c.camera_name, c.camera_id)                AS camera_name,
    COALESCE(c.location_description, '-')               AS location,
    COUNT(*)                                            AS total_violations,
    ROUND(AVG(f.confidence)::numeric, 3)                AS avg_confidence,
    ROUND(AVG(f.processing_latency_ms)::numeric, 1)     AS avg_latency_ms,
    MIN(f.created_at)                                   AS first_violation,
    MAX(f.created_at)                                   AS last_violation
FROM fact_violations f
LEFT JOIN dim_camera c ON f.camera_fk = c.camera_pk
GROUP BY c.camera_id, c.camera_name, c.location_description
ORDER BY total_violations DESC;


-- ══════════════════════════════════════════════
-- SECTION 3: Pola Pelanggaran
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


-- ── Panel: Hourly bar chart — violations per hour ──
-- Shows violation distribution across 24 hours for pattern discovery
-- Jam is TEXT (e.g. "16:00") because Grafana barchart needs a string x-axis
CREATE OR REPLACE VIEW vw_hourly_bar AS
SELECT
    lpad(h.hour::text, 2, '0') || ':00'                AS "Jam",
    COALESCE(agg.total, 0)                              AS "Pelanggaran",
    COALESCE(ROUND(agg.avg_conf::numeric, 3), 0)        AS "Avg Confidence"
FROM generate_series(0, 23) AS h(hour)
LEFT JOIN (
    SELECT hour, COUNT(*) AS total, AVG(confidence) AS avg_conf
    FROM fact_violations
    GROUP BY hour
) agg ON agg.hour = h.hour
ORDER BY h.hour;


-- ── Panel: Confidence distribution — bucketed histogram ──
-- Groups violations into confidence ranges for model quality insight
CREATE OR REPLACE VIEW vw_confidence_distribution AS
SELECT
    ranges.bucket                                       AS "Range Confidence",
    COALESCE(agg.total, 0)                              AS "Jumlah"
FROM (
    VALUES ('0.5-0.6'), ('0.6-0.7'), ('0.7-0.8'), ('0.8-0.9'), ('0.9-1.0')
) AS ranges(bucket)
LEFT JOIN (
    SELECT
        CASE
            WHEN confidence >= 0.9 THEN '0.9-1.0'
            WHEN confidence >= 0.8 THEN '0.8-0.9'
            WHEN confidence >= 0.7 THEN '0.7-0.8'
            WHEN confidence >= 0.6 THEN '0.6-0.7'
            ELSE '0.5-0.6'
        END AS conf_bucket,
        COUNT(*) AS total
    FROM fact_violations
    GROUP BY 1
) agg ON agg.conf_bucket = ranges.bucket
ORDER BY ranges.bucket;


-- ══════════════════════════════════════════════
-- SECTION 4: Log Detail
-- ══════════════════════════════════════════════

-- ── Panel 9: Recent violations log (table) ──
-- Pre-joined with dim_camera, pre-formatted columns
CREATE OR REPLACE VIEW vw_recent_violations AS
SELECT
    f.created_at                                        AS "Waktu",
    c.camera_id                                         AS "Camera ID",
    COALESCE(c.camera_name, c.camera_id)                AS "Kamera",
    COALESCE(c.location_description, '-')               AS "Lokasi",
    COALESCE(vt.type_name, '-')                         AS "Tipe",
    ROUND(f.confidence::numeric, 3)                     AS "Confidence",
    f.track_id                                          AS "Track ID",
    ROUND(f.processing_latency_ms::numeric, 1)          AS "Latency (ms)",
    f.time_period                                       AS "Periode",
    f.hour                                              AS "Jam"
FROM fact_violations f
LEFT JOIN dim_camera c ON f.camera_fk = c.camera_pk
LEFT JOIN dim_violation_type vt ON f.violation_type_fk = vt.type_pk
ORDER BY f.created_at DESC
LIMIT 100;


-- ══════════════════════════════════════════════
-- SECTION 5: Legacy views (backward compatibility)
-- ══════════════════════════════════════════════

-- ── Hourly violations (used internally by vw_heatmap_hour_day) ──
CREATE OR REPLACE VIEW vw_hourly_violations AS
SELECT
    f.date,
    f.hour,
    f.day_of_week,
    f.time_period,
    c.camera_id,
    COUNT(*)                        AS violation_count,
    AVG(f.confidence)               AS avg_confidence,
    AVG(f.processing_latency_ms)    AS avg_latency_ms
FROM fact_violations f
LEFT JOIN dim_camera c ON f.camera_fk = c.camera_pk
GROUP BY f.date, f.hour, f.day_of_week, f.time_period, c.camera_id;


-- ── Today's summary (used internally by vw_kpi_summary) ──
CREATE OR REPLACE VIEW vw_today_summary AS
SELECT
    COUNT(*)                        AS total_violations_today,
    AVG(confidence)                 AS avg_confidence_today,
    AVG(processing_latency_ms)      AS avg_latency_today,
    MAX(created_at)                 AS last_violation_time
FROM fact_violations
WHERE date = CURRENT_DATE;


-- ══════════════════════════════════════════════
-- SECTION 6: Distribusi Tipe & Geomap
-- ══════════════════════════════════════════════

-- ── Panel: Pie Chart — Violation type distribution ──
-- Pivoted single-row format like vw_time_period_distribution
-- Each column = one violation type, value = count
-- This format is natively understood by Grafana piechart
CREATE OR REPLACE VIEW vw_violation_type_distribution AS
SELECT
    COALESCE(SUM(CASE WHEN vt.type_code = 'no_helmet'           THEN 1 END), 0) AS "Tanpa Helm (Pengendara)",
    COALESCE(SUM(CASE WHEN vt.type_code = 'no_helmet_passenger' THEN 1 END), 0) AS "Tanpa Helm (Penumpang)"
FROM fact_violations f
JOIN dim_violation_type vt ON f.violation_type_fk = vt.type_pk;


-- ── Panel: Table — Violation type detail with severity ──
-- Multi-row format for detail table showing severity, confidence, latency
CREATE OR REPLACE VIEW vw_violation_type_detail AS
SELECT
    vt.type_name                                        AS "Tipe Pelanggaran",
    COUNT(f.violation_pk)                               AS "Jumlah",
    ROUND(
        COUNT(f.violation_pk) * 100.0
        / NULLIF((SELECT COUNT(*) FROM fact_violations), 0)
    , 1)                                                AS "Persentase (%)",
    CASE vt.severity_level
        WHEN 3 THEN 'Tinggi'
        WHEN 2 THEN 'Sedang'
        WHEN 1 THEN 'Rendah'
    END                                                 AS "Severity",
    ROUND(AVG(f.confidence)::numeric, 3)                AS "Avg Confidence",
    ROUND(AVG(f.processing_latency_ms)::numeric, 1)     AS "Avg Latency (ms)"
FROM dim_violation_type vt
LEFT JOIN fact_violations f ON f.violation_type_fk = vt.type_pk
GROUP BY vt.type_pk, vt.type_name, vt.type_code, vt.severity_level
ORDER BY "Jumlah" DESC;


-- ── Panel: Geomap — Camera markers with violation counts ──
-- Returns one row per camera with GPS + aggregated metrics + risk label
-- Grafana Geomap reads latitude/longitude columns automatically
CREATE OR REPLACE VIEW vw_camera_geomap AS
SELECT
    c.camera_id,
    c.camera_name,
    c.location_description,
    c.gps_latitude                                      AS latitude,
    c.gps_longitude                                     AS longitude,
    COALESCE(agg.total_violations, 0)                   AS total_violations,
    COALESCE(ROUND(agg.avg_confidence::numeric, 3), 0)  AS avg_confidence,
    COALESCE(ROUND(agg.avg_latency::numeric, 1), 0)     AS avg_latency_ms,
    -- Risk level based on violation count thresholds
    CASE
        WHEN COALESCE(agg.total_violations, 0) >= 50  THEN 'Tinggi'
        WHEN COALESCE(agg.total_violations, 0) >= 10  THEN 'Sedang'
        WHEN COALESCE(agg.total_violations, 0) >= 1   THEN 'Rendah'
        ELSE 'Aman'
    END                                                 AS risk_level,
    -- Peak hour at this camera (most violations happen at this hour)
    agg.peak_hour,
    -- Dominant time period
    agg.peak_period,
    agg.last_violation
FROM dim_camera c
LEFT JOIN (
    SELECT
        camera_fk,
        COUNT(*)                    AS total_violations,
        AVG(confidence)             AS avg_confidence,
        AVG(processing_latency_ms)  AS avg_latency,
        MAX(created_at)             AS last_violation,
        -- Find the hour with most violations
        (ARRAY_AGG(hour ORDER BY cnt DESC))[1] AS peak_hour,
        (ARRAY_AGG(time_period ORDER BY cnt DESC))[1] AS peak_period
    FROM (
        SELECT camera_fk, hour, time_period, confidence, processing_latency_ms, created_at,
               COUNT(*) OVER (PARTITION BY camera_fk, hour) AS cnt
        FROM fact_violations
    ) sub
    GROUP BY camera_fk
) agg ON agg.camera_fk = c.camera_pk
WHERE c.gps_latitude IS NOT NULL
  AND c.gps_longitude IS NOT NULL;


