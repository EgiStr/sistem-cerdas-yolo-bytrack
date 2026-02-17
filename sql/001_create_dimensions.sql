-- ============================================
-- ITERA Smart Sentinel - Dimension Tables
-- Star Schema for Violation Analytics
-- ============================================

-- Dimension: Camera/Location
CREATE TABLE IF NOT EXISTS dim_camera (
    camera_id VARCHAR(50) PRIMARY KEY,
    camera_name VARCHAR(100) NOT NULL,
    location VARCHAR(200),
    gate_name VARCHAR(100),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Dimension: Violation Type
CREATE TABLE IF NOT EXISTS dim_violation_type (
    violation_type_id SERIAL PRIMARY KEY,
    type_code VARCHAR(50) UNIQUE NOT NULL,
    type_name VARCHAR(100) NOT NULL,
    description TEXT,
    severity VARCHAR(20) DEFAULT 'high',
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- Seed default data
-- ============================================

INSERT INTO dim_camera (camera_id, camera_name, location, gate_name)
VALUES
    ('gate_utama_01', 'Kamera Gerbang Utama 1', 'Jalan Terusan Ryacudu, Gerbang Utama ITERA', 'Gerbang Utama'),
    ('gate_utama_02', 'Kamera Gerbang Utama 2', 'Jalan Terusan Ryacudu, Gerbang Utama ITERA', 'Gerbang Utama'),
    ('gate_belakang_01', 'Kamera Gerbang Belakang', 'Gerbang Belakang ITERA', 'Gerbang Belakang')
ON CONFLICT (camera_id) DO NOTHING;

INSERT INTO dim_violation_type (type_code, type_name, description, severity)
VALUES
    ('no_helmet', 'Tidak Menggunakan Helm', 'Pengendara motor tanpa helm di area kampus', 'high'),
    ('no_helmet_passenger', 'Penumpang Tanpa Helm', 'Penumpang motor tanpa helm di area kampus', 'high')
ON CONFLICT (type_code) DO NOTHING;
