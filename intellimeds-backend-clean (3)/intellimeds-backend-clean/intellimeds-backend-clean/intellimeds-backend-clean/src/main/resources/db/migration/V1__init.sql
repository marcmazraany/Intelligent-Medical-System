-- Enable UUID generation if needed (Supabase/Postgres usually has pgcrypto)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- USERS
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    phone TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- PROFILES (FINAL SCHEMA)
CREATE TABLE IF NOT EXISTS profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,

    first_name TEXT NOT NULL,
    last_name  TEXT NOT NULL,
    dob        DATE,

    gender TEXT,
    height TEXT,
    weight TEXT,
    allergies TEXT,
    blood_type TEXT,
    chronic_conditions TEXT,
    notes TEXT
);

-- MEDICATIONS
CREATE TABLE IF NOT EXISTS medications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    dosage TEXT NOT NULL,
    expiry_date DATE NOT NULL,
    frequency TEXT NOT NULL,
    reminder_times JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_medications_user_id ON medications(user_id);

-- ALERTS
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    medication_name TEXT NOT NULL,
    max_price NUMERIC(12,2) NOT NULL,
    email_enabled BOOLEAN NOT NULL DEFAULT false,
    created_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_notified TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'active',
    active BOOLEAN NOT NULL DEFAULT true
);
CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON alerts(user_id);