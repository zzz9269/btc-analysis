-- ============================================================
-- Phase 0 / Phase 2 prep tables
-- Run this in Supabase → SQL Editor → New query
-- ============================================================

-- Per-signal contributions per tick (powers IC weighting, SHAP, rolling perf).
CREATE TABLE IF NOT EXISTS signal_detail (
    id           BIGSERIAL PRIMARY KEY,
    ts           TIMESTAMPTZ NOT NULL,
    signal_name  TEXT        NOT NULL,
    raw_value    NUMERIC,
    weight       NUMERIC,
    contribution NUMERIC
);
CREATE INDEX IF NOT EXISTS idx_signal_detail_ts   ON signal_detail(ts);
CREATE INDEX IF NOT EXISTS idx_signal_detail_name ON signal_detail(signal_name);

-- Per-strike Polymarket probabilities per tick (powers velocity / acceleration).
CREATE TABLE IF NOT EXISTS polymarket_log (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL,
    question    TEXT,
    strike_lbl  TEXT,
    probability NUMERIC,
    is_bull     BOOLEAN,
    mkt_score   NUMERIC,
    mkt_weight  NUMERIC
);
CREATE INDEX IF NOT EXISTS idx_polymarket_log_ts ON polymarket_log(ts);

-- ============================================================
-- RLS policy: leave RLS disabled (service_role key bypasses it anyway,
-- but disabling explicitly removes ambiguity).
-- ============================================================
ALTER TABLE signal_detail   DISABLE ROW LEVEL SECURITY;
ALTER TABLE polymarket_log  DISABLE ROW LEVEL SECURITY;
