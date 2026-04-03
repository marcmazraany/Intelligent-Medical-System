-- ============================================
-- RUN THESE COMMANDS IN PostgreSQL
-- ============================================

-- 1. Connect to PostgreSQL
-- Windows: Open Command Prompt as Administrator
-- Run: psql -U postgres

-- 2. Create the MAIN database (for YOUR system only)
CREATE DATABASE medmanager_db;

-- 3. Connect to it
\c medmanager_db

-- 4. Create all tables
CREATE TABLE medications (
                             id BIGSERIAL PRIMARY KEY,
                             name VARCHAR(255) NOT NULL,
                             generic_name VARCHAR(255),
                             dosage VARCHAR(100) NOT NULL,
                             manufacturer VARCHAR(255),
                             description TEXT,
                             barcode VARCHAR(100) UNIQUE,
                             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE pharmacy_nodes (
                                id BIGSERIAL PRIMARY KEY,
                                name VARCHAR(255) NOT NULL,
                                host VARCHAR(255) NOT NULL,
                                port INTEGER,
                                latitude DOUBLE PRECISION NOT NULL,
                                longitude DOUBLE PRECISION NOT NULL,
                                address TEXT,
                                phone_number VARCHAR(50),
                                data_source_type VARCHAR(50) NOT NULL,
                                api_endpoint TEXT,
                                database_url TEXT,
                                database_username VARCHAR(255),
                                database_password VARCHAR(255),
                                table_name VARCHAR(255),
                                file_path TEXT,
                                active BOOLEAN DEFAULT true
);

CREATE TABLE pharmacy_inventory_cache (
                                          id BIGSERIAL PRIMARY KEY,
                                          pharmacy_node_id BIGINT REFERENCES pharmacy_nodes(id),
                                          pharmacy_name VARCHAR(255) NOT NULL,
                                          medication_id BIGINT REFERENCES medications(id),
                                          medication_name VARCHAR(255) NOT NULL,
                                          stock_quantity INTEGER NOT NULL,
                                          price DOUBLE PRECISION NOT NULL,
                                          currency VARCHAR(10) DEFAULT 'LBP',
                                          pharmacy_latitude DOUBLE PRECISION NOT NULL,
                                          pharmacy_longitude DOUBLE PRECISION NOT NULL,
                                          last_updated TIMESTAMP NOT NULL,
                                          next_check_time TIMESTAMP NOT NULL,
                                          priority VARCHAR(20) NOT NULL,
                                          available BOOLEAN NOT NULL
);

CREATE TABLE stock_alerts (
                              id BIGSERIAL PRIMARY KEY,
                              user_email VARCHAR(255) NOT NULL,
                              user_phone VARCHAR(50),
                              medication_id BIGINT REFERENCES medications(id),
                              medication_name VARCHAR(255) NOT NULL,
                              max_price DOUBLE PRECISION,
                              max_distance DOUBLE PRECISION,
                              active BOOLEAN DEFAULT true,
                              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                              last_notified TIMESTAMP,
                              notify_by_email BOOLEAN DEFAULT true,
                              notify_by_sms BOOLEAN DEFAULT false
);

-- Create indexes
CREATE INDEX idx_cache_medication ON pharmacy_inventory_cache(medication_id, available);
CREATE INDEX idx_cache_next_check ON pharmacy_inventory_cache(next_check_time);
CREATE INDEX idx_alerts_medication ON stock_alerts(medication_id, active);

-- Insert sample medications
INSERT INTO medications (name, generic_name, dosage, manufacturer, description) VALUES
                                                                                    ('Panadol', 'Paracetamol', '500mg', 'GSK', 'Pain reliever'),
                                                                                    ('Augmentin', 'Amoxicillin', '1g', 'GSK', 'Antibiotic'),
                                                                                    ('Ventolin', 'Albuterol', '100mcg', 'GSK', 'Asthma inhaler'),
                                                                                    ('Aspirin', 'Acetylsalicylic Acid', '100mg', 'Bayer', 'Blood thinner'),
                                                                                    ('Atorvastatin', 'Atorvastatin', '20mg', 'Pfizer', 'Cholesterol medication');

-- Insert pharmacy nodes (they DON'T need databases - they just expose REST APIs!)
INSERT INTO pharmacy_nodes (name, host, port, latitude, longitude, address, phone_number, data_source_type, api_endpoint, active) VALUES
                                                                                                                                      ('Pharmacy Alpha', 'localhost', 8081, 33.8886, 35.4955, 'Street 1, Beirut', '+961-1-111111', 'REST_API', 'http://localhost:8081/api/pharmacy/inventory', true),
                                                                                                                                      ('Pharmacy Beta', 'localhost', 8082, 33.8938, 35.5018, 'Street 2, Beirut', '+961-1-222222', 'REST_API', 'http://localhost:8082/api/pharmacy/inventory', true),
                                                                                                                                      ('Pharmacy Gamma', 'localhost', 8083, 33.8750, 35.5100, 'Street 3, Beirut', '+961-1-333333', 'REST_API', 'http://localhost:8083/api/pharmacy/inventory', true);

-- Done! Exit
\q