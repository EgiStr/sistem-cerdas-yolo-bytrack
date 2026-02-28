-- ============================================
-- ITERA Smart Sentinel - Fact Table
-- Central fact table for violation events
-- Sesuai proposal Gambar 3.2 dengan tambahan
-- denormalized time fields untuk optimasi Grafana
-- ============================================

CREATE TABLE IF NOT EXISTS fact_violations (
    violation_pk BIGSERIAL PRIMARY KEY,

    -- Foreign keys (Star Schema sesuai Gambar 3.2)
    timestamp_fk INT REFERENCES dim_time(time_id),
    camera_fk INT REFERENCES dim_camera(camera_pk),
    violation_type_fk INT REFERENCES dim_violation_type(type_pk),

    -- Degenerate dimensions
    track_id INT NOT NULL,
    confidence FLOAT NOT NULL,

    -- Bounding box
    bbox_x1 INT,
    bbox_y1 INT,
    bbox_x2 INT,
    bbox_y2 INT,

    -- Measures
    frame_number INT,
    processing_latency_ms FLOAT,
    violation_count INT DEFAULT 1,

    -- Denormalized time fields (optimasi: menghindari JOIN untuk Grafana)
    hour INT,
    minute INT,
    day_of_week INT,
    date DATE,
    week_of_year INT,
    month INT,
    year INT,
    time_period VARCHAR(20),

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- Indexes for Grafana dashboard queries
-- ============================================

CREATE INDEX IF NOT EXISTS idx_fv_created_at
    ON fact_violations (created_at);

CREATE INDEX IF NOT EXISTS idx_fv_date
    ON fact_violations (date);

CREATE INDEX IF NOT EXISTS idx_fv_hour
    ON fact_violations (hour);

CREATE INDEX IF NOT EXISTS idx_fv_camera_fk
    ON fact_violations (camera_fk);

CREATE INDEX IF NOT EXISTS idx_fv_time_period
    ON fact_violations (time_period);

CREATE INDEX IF NOT EXISTS idx_fv_timestamp_fk
    ON fact_violations (timestamp_fk);

-- Composite index for common dashboard queries
CREATE INDEX IF NOT EXISTS idx_fv_date_hour_camera
    ON fact_violations (date, hour, camera_fk);
