"""
Diagnose / recalibrate the 72h-bias LONG/SHORT direction threshold.

Context (2026-06-23 audit): since the anti-churn cutoff
(_BACKTEST_CUTOFF_UTC_ISO = 2026-06-19T16:00 UTC) the engine has logged 100%
HOLD because its score range collapsed (~[-19, +1]) and never reaches the ±25
LONG/SHORT threshold. The win/loss scorecard therefore reads "0 resolved".

This script answers two questions the live dashboard can't:
  1. Is the threshold simply too high for the new (compressed) score scale?
  2. Or is the score MIS-CENTERED — a persistent negative offset that pins it
     in SHORT/HOLD regardless of threshold?

It grades the score's SIGN and RANK against the realized +72h move (pct_move,
already resolved on each row, incl. HOLD), compares against the drift null, and
proposes a re-centered threshold. It is READ-ONLY — it never writes the engine
or Supabase. Re-run as more post-cutoff data (and more market regimes) land;
today's sample is ~one independent episode, so treat the numbers as directional.

Usage:
    python calibrate_threshold.py
    python calibrate_threshold.py --target-fire 0.12   # design SHORT+LONG rate

Credentials: SUPABASE_URL / SUPABASE_KEY env vars, falling back to
.streamlit/secrets.toml (same as regrade_72h.py).
"""
import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

HERE   = Path(__file__).parent
TABLE  = "signal_log"
# Keep in sync with _BACKTEST_CUTOFF_UTC_ISO in btc_analysis_app.py.
CUTOFF = "2026-06-19T16:00:00+00:00"
OUTCOME_HRS = 72.0


def _load_creds():
    url = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
    key = os.environ.get("SUPABASE_KEY") or ""
    if url and key:
        return url, key
    secrets = HERE / ".streamlit" / "secrets.toml"
    if secrets.exists():
        try:
            import tomllib
            data = tomllib.loads(secrets.read_text(encoding="utf-8"))
            return (data.get("SUPABASE_URL") or "").rstrip("/"), data.get("SUPABASE_KEY") or ""
        except Exception as e:
            print(f"[warn] could not parse secrets.toml: {e}")
    return "", ""


SUPABASE_URL, SUPABASE_KEY = _load_creds()


def _get(query):
    url = f"{SUPABASE_URL}/rest/v1/{TABLE}?{query}"
    req = urllib.request.Request(
        url, headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _fetch_post_cutoff():
    rows, off = [], 0
    q_cut = urllib.parse.quote(CUTOFF)
    while True:
        batch = _get(f"select=ts,score,direction,pct_move,correct"
                     f"&ts=gte.{q_cut}&order=ts.asc&limit=1000&offset={off}")
        if not batch:
            break
        rows += batch
        off += 1000
        if len(batch) < 1000:
            break
    return rows


# ── tiny stats helpers (no numpy/pandas dependency) ──────────────────────────
def _spearman(xs, ys):
    n = len(xs)
    if n < 10:
        return None

    def _rank(a):
        order = sorted(range(n), key=lambda i: a[i])
        rk = [0.0] * n
        i = 0
        while i < n:                       # average ties
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    sd = (sum((v - mx) ** 2 for v in rx) * sum((v - my) ** 2 for v in ry)) ** 0.5
    return round(cov / sd, 3) if sd else None


def _sign_acc(scores, moves):
    hit = tot = 0
    for s, m in zip(scores, moves):
        if s == 0 or m == 0:
            continue
        tot += 1
        if (s > 0) == (m > 0):
            hit += 1
    return (round(hit / tot * 100, 1), tot) if tot else (None, 0)


def _causal_recenter(scores):
    """Subtract an expanding median of prior scores (no lookahead) — what a live
    re-centering would actually see tick-by-tick."""
    out, seen = [], []
    for s in scores:
        off = median(seen) if len(seen) >= 20 else 0.0
        out.append(s - off)
        seen.append(s)
    return out


def _n_independent_episodes(scores, moves):
    """Rough count of independent directional runs on the de-meaned sign — 5-min
    ticks graded on 72h overlap almost completely, so n_ticks hugely overstates
    evidence (see winrate_autocorrelation_finding)."""
    eps, prev = 0, None
    for s in scores:
        d = 1 if s > 0 else (-1 if s < 0 else 0)
        if d != 0 and d != prev:
            eps += 1
            prev = d
    return eps


def _fire_rates(scores, t):
    n = len(scores)
    lo = sum(1 for s in scores if s >= t) / n * 100
    sh = sum(1 for s in scores if s <= -t) / n * 100
    return lo + sh, lo, sh


def main():
    target_fire = 0.12
    if "--target-fire" in sys.argv:
        try:
            target_fire = float(sys.argv[sys.argv.index("--target-fire") + 1])
        except Exception:
            pass

    if not (SUPABASE_URL and SUPABASE_KEY):
        print("No Supabase credentials (env or .streamlit/secrets.toml). Abort.")
        return

    rows = _fetch_post_cutoff()
    if not rows:
        print("No post-cutoff rows returned.")
        return

    now = datetime.now(timezone.utc)
    scores_all = [float(r["score"]) for r in rows if r.get("score") is not None]
    matured = [r for r in rows
               if r.get("pct_move") not in (None, "")
               and (now - datetime.fromisoformat(r["ts"])).total_seconds() / 3600 >= OUTCOME_HRS]
    m_scores = [float(r["score"]) for r in matured]
    m_moves  = [float(r["pct_move"]) for r in matured]

    print("=" * 68)
    print(f"72h-BIAS THRESHOLD CALIBRATION   (cutoff {CUTOFF})")
    print("=" * 68)
    print(f"post-cutoff rows : {len(scores_all)}   matured (>=72h) : {len(matured)}")
    dirs = {}
    for r in rows:
        dirs[r.get("direction")] = dirs.get(r.get("direction"), 0) + 1
    print(f"direction logged : {dirs}")

    # ── 1. Distribution ─────────────────────────────────────────────────────
    s = sorted(scores_all)
    p = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    print("\n[1] SCORE DISTRIBUTION (current engine)")
    print(f"    min/med/max  : {s[0]:.1f} / {median(s):.1f} / {s[-1]:.1f}")
    print(f"    p5/p25/p75/p95: {p(.05):.1f} / {p(.25):.1f} / {p(.75):.1f} / {p(.95):.1f}")
    print(f"    persistent offset (median): {median(s):+.1f}  "
          f"-> {'NEGATIVE lean, pins SHORT/HOLD' if median(s) < -2 else 'roughly centered'}")

    if len(matured) < 10:
        print("\n[!] <10 matured rows — accuracy section needs more data. Re-run later.")
        return

    # ── 2. Skill: rank (offset-invariant) vs sign (offset-sensitive) ─────────
    ic = _spearman(m_scores, m_moves)
    raw_acc, n_raw = _sign_acc(m_scores, m_moves)
    full_dm = [x - median(m_scores) for x in m_scores]
    full_acc, _ = _sign_acc(full_dm, m_moves)
    caus_dm = _causal_recenter(m_scores)
    caus_acc, _ = _sign_acc(caus_dm, m_moves)
    up_rate = sum(1 for m in m_moves if m > 0) / len(m_moves) * 100
    drift_null = max(up_rate, 100 - up_rate)
    eps = _n_independent_episodes(caus_dm, m_moves)

    print("\n[2] SKILL ON MATURED ROWS")
    print(f"    Forward IC (Spearman score->+72h move): {ic}  <- offset-INVARIANT, the real signal")
    print(f"    Sign accuracy  raw score             : {raw_acc}%  (n={n_raw})")
    print(f"    Sign accuracy  de-meaned (full)      : {full_acc}%   [uses hindsight median]")
    print(f"    Sign accuracy  de-meaned (causal)    : {caus_acc}%   [no lookahead - realistic]")
    print(f"    DRIFT NULL (always-{'long' if up_rate >= 50 else 'short'})        : {drift_null:.1f}%   "
          f"<- beat THIS to claim sign skill")
    verdict = ("MIS-CENTERED: re-centering flips sign acc up sharply; the offset "
               "is the problem, not the information"
               if (full_acc or 0) - (raw_acc or 0) > 15 else
               "centering is not the dominant issue")
    print(f"    -> {verdict}")
    print(f"    independent episodes ~= {eps} (ticks={n_raw}; 72h overlap => "
          f"treat as directional, NOT a verdict)")

    # ── 3. Threshold options ─────────────────────────────────────────────────
    print(f"\n[3] FIRE-RATE @ thresholds  (target SHORT+LONG ~= {target_fire*100:.0f}%)")
    print("    RAW score (current, mis-centered):")
    for t in (25, 20, 15, 12, 10):
        f, lo, sh = _fire_rates(scores_all, t)
        print(f"      |s|>={t:>2}: fire {f:4.0f}%   LONG {lo:4.0f}% / SHORT {sh:4.0f}%"
              f"{'   <- ALL ONE SIDE = constant rule, untestable' if (lo == 0 or sh == 0) and f > 0 else ''}")
    rc = [x - median(scores_all) for x in scores_all]
    print(f"    RE-CENTERED score (minus median {median(scores_all):+.1f}):")
    for t in (15, 12, 10, 8, 6):
        f, lo, sh = _fire_rates(rc, t)
        print(f"      |s|>={t:>2}: fire {f:4.0f}%   LONG {lo:4.0f}% / SHORT {sh:4.0f}%")
    # pick threshold on re-centered score that hits target fire rate
    rc_abs = sorted((abs(x) for x in rc), reverse=True)
    thr = rc_abs[min(len(rc_abs) - 1, int(target_fire * len(rc_abs)))]
    f, lo, sh = _fire_rates(rc, thr)
    print(f"\n[4] PROPOSAL: re-center (subtract causal median) THEN threshold ~= +/-{thr:.0f}")
    print(f"    -> fire {f:.0f}%  (LONG {lo:.0f}% / SHORT {sh:.0f}%)  two-sided & gradeable")
    print("    NOTE: do NOT just lower +/-25 on the raw score -- that yields an all-SHORT")
    print("    constant rule. Re-centering is the lever; threshold is secondary.")
    print("=" * 68)


if __name__ == "__main__":
    main()
