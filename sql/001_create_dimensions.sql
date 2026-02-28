-- ============================================
-- ITERA Smart Sentinel - Dimension Tables
-- Star Schema for Violation Analytics
-- Sesuai proposal Gambar 3.2
-- ============================================

-- Dimension: Time
-- Diisi otomatis oleh Spark Stream Processor
CREATE TABLE IF NOT EXISTS dim_time (
    time_id SERIAL PRIMARY KEY,
    full_timestamp TIMESTAMP NOT NULL UNIQUE,
    date DATE NOT NULL,
    year INT NOT NULL,
    month INT NOT NULL,
    day_of_week INT NOT NULL,         -- 1=Minggu .. 7=Sabtu (PostgreSQL)
    hour_of_day INT NOT NULL,
    time_name VARCHAR(20) NOT NULL,   -- pagi/siang/sore/malam
    is_weekend BOOLEAN DEFAULT FALSE,
    is_exam_period BOOLEAN DEFAULT FALSE
);

-- Dimension: Camera/Location
CREATE TABLE IF NOT EXISTS dim_camera (
    camera_pk SERIAL PRIMARY KEY,
    camera_id VARCHAR(100) UNIQUE NOT NULL,   -- Business key
    camera_name VARCHAR(100) NOT NULL,
    location_description VARCHAR(210),
    gps_latitude NUMERIC(10,8),
    gps_longitude NUMERIC(11,8),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Dimension: Violation Type
CREATE TABLE IF NOT EXISTS dim_violation_type (
    type_pk SERIAL PRIMARY KEY,
    type_code VARCHAR(50) UNIQUE NOT NULL,
    type_name VARCHAR(100) NOT NULL,
    description VARCHAR(300),
    severity_level INT DEFAULT 3 CHECK (severity_level BETWEEN 1 AND 3),
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- Seed default data
-- ============================================

INSERT INTO dim_camera (camera_id, camera_name, location_description, gps_latitude, gps_longitude)
VALUES
    ('gate_utama_01', 'Kamera Gerbang Utama 1', 'Jalan Terusan Ryacudu, Gerbang Utama ITERA', -5.35783000, 105.31446000),
    ('gate_utama_02', 'Kamera Gerbang Utama 2', 'Jalan Terusan Ryacudu, Gerbang Utama ITERA', -5.35795000, 105.31465000),
    ('gate_belakang_01', 'Kamera Gerbang Belakang', 'Gerbang Belakang ITERA',                -5.36170000, 105.31800000)
ON CONFLICT (camera_id) DO NOTHING;

INSERT INTO dim_violation_type (type_code, type_name, description, severity_level)
VALUES
    ('no_helmet', 'Tidak Menggunakan Helm', 'Pengendara motor tanpa helm di area kampus', 3),
    ('no_helmet_passenger', 'Penumpang Tanpa Helm', 'Penumpang motor tanpa helm di area kampus', 3)
ON CONFLICT (type_code) DO NOTHING;

-- ============================================
-- Indexes for dimension lookups
-- ============================================
CREATE INDEX IF NOT EXISTS idx_dim_time_timestamp ON dim_time (full_timestamp);
CREATE INDEX IF NOT EXISTS idx_dim_time_date ON dim_time (date);
