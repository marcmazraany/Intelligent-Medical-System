-- ============================================================
-- V4: Extend alerts table for pharmacy-finder stock alerts
-- The auto system handles its own DB (pharmacy nodes, cache).
-- IntelliMeds only needs to know about user alerts.
-- ============================================================

ALTER TABLE alerts
    ADD COLUMN IF NOT EXISTS max_distance DOUBLE PRECISION;

-- Back-fill so existing rows are consistent
UPDATE alerts SET max_distance = NULL WHERE max_distance IS NULL;
