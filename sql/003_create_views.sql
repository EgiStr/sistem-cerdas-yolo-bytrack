-- ============================================
-- ITERA Smart Sentinel - Analytical Views
-- Materialized views for Grafana dashboard
-- ============================================

-- ── Hourly violations (for heatmap panel) ──
CREATE OR REPLACE VIEW vw_hourly_violations AS
SELECT
    date,
    hour,
    day_of_week,
    time_period,
    camera_id,
    COUNT(*) AS violation_count,
    AVG(confidence) AS avg_confidence,
    AVG(processing_latency_ms) AS avg_latency_ms
FROM fact_violations
GROUP BY date, hour, day_of_week, time_period, camera_id;


-- ── Daily trend (for time-series panel) ──
CREATE OR REPLACE VIEW vw_daily_trend AS
SELECT
    date,
    camera_id,
    COUNT(*) AS violation_count,
    AVG(confidence) AS avg_confidence,
    MIN(confidence) AS min_confidence,
    MAX(confidence) AS max_confidence,
    AVG(processing_latency_ms) AS avg_latency_ms
FROM fact_violations
GROUP BY date, camera_id
ORDER BY date;


-- ── Camera statistics (for bar chart panel) ──
CREATE OR REPLACE VIEW vw_camera_stats AS
SELECT
    f.camera_id,
    c.camera_name,
    c.gate_name,
    COUNT(*) AS total_violations,
    AVG(f.confidence) AS avg_confidence,
    AVG(f.processing_latency_ms) AS avg_latency_ms,
    MIN(f.created_at) AS first_violation,
    MAX(f.created_at) AS last_violation
FROM fact_violations f
LEFT JOIN dim_camera c ON f.camera_id = c.camera_id
GROUP BY f.camera_id, c.camera_name, c.gate_name;


-- ── Peak hours analysis (for heatmap: hour x day_of_week) ──
CREATE OR REPLACE VIEW vw_peak_hours AS
SELECT
    day_of_week,
    hour,
    COUNT(*) AS violation_count,
    AVG(confidence) AS avg_confidence
FROM fact_violations
GROUP BY day_of_week, hour
ORDER BY day_of_week, hour;


-- ── Today's summary (for KPI panel) ──
CREATE OR REPLACE VIEW vw_today_summary AS
SELECT
    COUNT(*) AS total_violations_today,
    AVG(confidence) AS avg_confidence_today,
    AVG(processing_latency_ms) AS avg_latency_today,
    MAX(created_at) AS last_violation_time
FROM fact_violations
WHERE date = CURRENT_DATE;


-- ── Recent violations (for table panel) ──
CREATE OR REPLACE VIEW vw_recent_violations AS
SELECT
    f.violation_id,
    f.created_at,
    f.camera_id,
    c.camera_name,
    c.gate_name,
    f.violation_type,
    f.confidence,
    f.track_id,
    f.processing_latency_ms,
    f.time_period,
    f.hour
FROM fact_violations f
LEFT JOIN dim_camera c ON f.camera_id = c.camera_id
ORDER BY f.created_at DESC
LIMIT 100;
