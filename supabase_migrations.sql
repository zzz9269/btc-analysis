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
-- Phase B: enrich signal_log with the fields needed for IC weighting,
-- bull_prob calibration, regime-specific performance tracking, and
-- conviction-as-historical-accuracy (improvements.txt items 1, 2, 4, 6).
-- Safe to re-run — ADD COLUMN IF NOT EXISTS.
-- ============================================================
ALTER TABLE signal_log ADD COLUMN IF NOT EXISTS bull_prob   NUMERIC;
ALTER TABLE signal_log ADD COLUMN IF NOT EXISTS conviction  NUMERIC;
ALTER TABLE signal_log ADD COLUMN IF NOT EXISTS regime      TEXT;
ALTER TABLE signal_log ADD COLUMN IF NOT EXISTS score_24h   NUMERIC;

-- 2-signal audit baseline (EMA Structure + OI Funding, same ±100 scale).
-- Logged alongside the full score so audits can test whether the ~30-signal
-- blend beats a dumb baseline at episode level.
ALTER TABLE signal_log ADD COLUMN IF NOT EXISTS score_baseline NUMERIC;

-- ============================================================
-- Time × price liquidation heatmap history (Coinglass-style panel).
-- Cron logs top-25 synthetic-map bins per side every ~15 min, plus one
-- side='px' row carrying spot price for the overlay line.
-- ~50 rows / 15 min ≈ 4,800 rows/day. Prune occasionally:
--   DELETE FROM liq_heatmap_log WHERE ts < now() - interval '14 days';
-- ============================================================
CREATE TABLE IF NOT EXISTS liq_heatmap_log (
    id     BIGSERIAL PRIMARY KEY,
    ts     TIMESTAMPTZ NOT NULL,
    side   TEXT        NOT NULL,   -- 'long' | 'short' | 'px'
    price  NUMERIC     NOT NULL,
    usd    NUMERIC
);
CREATE INDEX IF NOT EXISTS idx_liq_heatmap_log_ts ON liq_heatmap_log(ts);

-- ============================================================
-- RLS policy: leave RLS disabled (service_role key bypasses it anyway,
-- but disabling explicitly removes ambiguity).
-- ============================================================
ALTER TABLE signal_detail   DISABLE ROW LEVEL SECURITY;
ALTER TABLE polymarket_log  DISABLE ROW LEVEL SECURITY;
ALTER TABLE liq_heatmap_log DISABLE ROW LEVEL SECURITY;
