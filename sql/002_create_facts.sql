-- ============================================
-- ITERA Smart Sentinel - Fact Table
-- Central fact table for violation events
-- ============================================

CREATE TABLE IF NOT EXISTS fact_violations (
    violation_id VARCHAR(36) PRIMARY KEY,
    camera_id VARCHAR(50) REFERENCES dim_camera(camera_id),
    track_id INT NOT NULL,
    violation_type VARCHAR(50) DEFAULT 'no_helmet',
    confidence FLOAT NOT NULL,
    
    -- Bounding box
    bbox_x1 INT,
    bbox_y1 INT,
    bbox_x2 INT,
    bbox_y2 INT,
    
    -- Performance metrics
    frame_number INT,
    processing_latency_ms FLOAT,
    
    -- Embedded time dimensions (denormalized for query speed)
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

CREATE INDEX IF NOT EXISTS idx_fact_violations_created_at 
    ON fact_violations (created_at);

CREATE INDEX IF NOT EXISTS idx_fact_violations_date 
    ON fact_violations (date);

CREATE INDEX IF NOT EXISTS idx_fact_violations_hour 
    ON fact_violations (hour);

CREATE INDEX IF NOT EXISTS idx_fact_violations_camera 
    ON fact_violations (camera_id);

CREATE INDEX IF NOT EXISTS idx_fact_violations_time_period 
    ON fact_violations (time_period);

-- Composite index for common dashboard queries
CREATE INDEX IF NOT EXISTS idx_fact_violations_date_hour_camera 
    ON fact_violations (date, hour, camera_id);
