"""
Headless 72h-bias signal logger.

Runs on GitHub Actions every 5 minutes (see .github/workflows/log_signal.yml).
Loads btc_analysis_app.py with a stubbed Streamlit, computes the full
16-signal 72h bias score, and writes one row to the Supabase `signal_log`
table. Also resolves any rows older than 72h that don't have an outcome yet.

Required env vars:
  SUPABASE_URL  — your project URL, e.g. https://xxxx.supabase.co
  SUPABASE_KEY  — the service_role key (preferred) or anon key
"""
import os
import sys
import json
import types
import importlib.util
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Windows consoles default to cp1252, which can't encode the ✓/… chars in our
# log prints — that raised UnicodeEncodeError mid-run (before the row insert),
# silently killing local logging. Force UTF-8 (replace on failure) so prints
# never crash the logger. No-op on GitHub Actions (already UTF-8).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or ""
TABLE        = "signal_log"
OUTCOME_HRS  = 72.0

# Local fallback: when run on this machine (not GitHub Actions) the env vars
# aren't set, so read them from .streamlit/secrets.toml next to this file.
# Keeps creds in ONE place — no duplicating the service key into a launcher.
if not (SUPABASE_URL and SUPABASE_KEY):
    _secrets = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if _secrets.exists():
        for _ln in _secrets.read_text(encoding="utf-8").splitlines():
            _ln = _ln.strip()
            if _ln.startswith("SUPABASE_URL") and not SUPABASE_URL:
                SUPABASE_URL = _ln.split("=", 1)[1].strip().strip('"').strip("'").rstrip("/")
            elif _ln.startswith("SUPABASE_KEY") and not SUPABASE_KEY:
                SUPABASE_KEY = _ln.split("=", 1)[1].strip().strip('"').strip("'")

if not (SUPABASE_URL and SUPABASE_KEY):
    print("ERROR: SUPABASE_URL or SUPABASE_KEY env var missing")
    sys.exit(1)

# Make the resolved creds visible to the app module's Supabase helpers. They
# read st.secrets -> os.environ; on GitHub Actions these are real env vars, but
# locally they came from secrets.toml into the vars above, so without this the
# app module's _supa_available() is False and _recenter_offset() /
# _slew_limit_score() silently no-op (offset 0.0, no slew limiting). setdefault
# so a real env var is never clobbered.
os.environ.setdefault("SUPABASE_URL", SUPABASE_URL)
os.environ.setdefault("SUPABASE_KEY", SUPABASE_KEY)


# ─────────────────────────────────────────────────────────────────
# Stub Streamlit so importing btc_analysis_app.py doesn't crash on
# UI calls and cache decorators. Cache decorators become passthroughs.
# ─────────────────────────────────────────────────────────────────
class _StubAny:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return self
    def __getattr__(self, _): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __bool__(self): return False
    def __iter__(self): return iter([])
    def __len__(self): return 0

class _StubCache:
    def __call__(self, *a, **kw):
        if a and callable(a[0]): return a[0]
        def deco(f): return f
        return deco

class _StubSecrets:
    def get(self, key, default=""):
        return os.environ.get(key, default) or default
    def __getitem__(self, k): return os.environ.get(k, "")
    def __contains__(self, k): return k in os.environ

class _StubSessionState(dict):
    def __getattr__(self, k): return self.get(k)
    def __setattr__(self, k, v): self[k] = v

st_mod = types.ModuleType("streamlit")
for name in [
    "set_page_config","title","header","subheader","write","markdown","metric",
    "columns","tabs","sidebar","container","expander","dataframe","plotly_chart",
    "warning","info","error","success","spinner","empty","button","checkbox",
    "selectbox","radio","slider","text_input","number_input","toggle","divider",
    "caption","code","json","progress","image","rerun","stop","pyplot","table",
    "altair_chart","line_chart","bar_chart","area_chart","number","form",
    "form_submit_button","download_button","file_uploader","date_input","time_input",
    "color_picker","multiselect","text_area","balloons","snow","status",
]:
    setattr(st_mod, name, _StubAny())
st_mod.cache_data     = _StubCache()
st_mod.cache_resource = _StubCache()
st_mod.secrets        = _StubSecrets()
st_mod.session_state  = _StubSessionState()
sys.modules["streamlit"] = st_mod

sar = types.ModuleType("streamlit_autorefresh")
sar.st_autorefresh = lambda **kw: None
sys.modules["streamlit_autorefresh"] = sar


# ─────────────────────────────────────────────────────────────────
# Load btc_analysis_app.py. The module-level UI code at the bottom
# will likely raise (TypeError on stubbed comparisons, etc.) — that's
# fine. By the time the error hits, all function definitions including
# run_analysis() are already bound in the module namespace.
# ─────────────────────────────────────────────────────────────────
APP_PATH = Path(__file__).parent / "btc_analysis_app.py"
spec = importlib.util.spec_from_file_location("btc_app", APP_PATH)
mod  = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
except Exception as e:
    print(f"[note] app module raised at UI layer (expected): {type(e).__name__}: {e}")

run_analysis = getattr(mod, "run_analysis", None)
if run_analysis is None:
    print("ERROR: run_analysis not found in btc_analysis_app.py")
    sys.exit(2)


# ─────────────────────────────────────────────────────────────────
# Supabase REST helpers
# ─────────────────────────────────────────────────────────────────
import time as _time

def _supa(method: str, path: str, body=None, retries: int = 3) -> "list | dict | None":
    """Supabase REST call with exponential-backoff retries on transient errors.
    Returns parsed JSON on success, or None after all attempts fail.
    4xx client errors (auth, validation) are NOT retried — they won't fix themselves."""
    url  = f"{SUPABASE_URL}/rest/v1{path}"
    data = json.dumps(body).encode() if body is not None else None
    last_err = ""
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers={
            "apikey":        SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type":  "application/json",
            "Prefer":        "return=representation",
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            body_snip = e.read().decode(errors='replace')[:200]
            last_err = f"HTTP {e.code}: {body_snip}"
            # 4xx is permanent — don't retry. 5xx and others may be transient.
            if 400 <= e.code < 500:
                print(f"[supabase {method} {path}] {last_err} (no retry)")
                return None
            print(f"[supabase {method} {path}] {last_err} (attempt {attempt}/{retries})")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"[supabase {method} {path}] {last_err} (attempt {attempt}/{retries})")
        if attempt < retries:
            _time.sleep(2 ** (attempt - 1))   # 1s, 2s, 4s
    return None


# ─────────────────────────────────────────────────────────────────
# 1. Resolve outcomes FIRST — before the expensive bias computation.
#    Root cause of the 3–8 Jun resolution outage: this section used to run
#    LAST, after run_analysis() + detail POSTs, and the workflow's
#    timeout-minutes kept killing the job before resolution was reached
#    (inserts kept landing, resolution made zero progress for days).
#    Resolution only needs a cheap spot price as fallback, so it runs first.
#    Graded against the BTC price at exactly entry+72h (core.price_history).
# ─────────────────────────────────────────────────────────────────
try:
    from core import price_history as _price_hist
except Exception:
    _price_hist = None

spot_price = _price_hist.btc_spot() if _price_hist else None
print(f"Spot (resolution fallback): {spot_price}")

cutoff_iso = (datetime.now(timezone.utc) - timedelta(hours=OUTCOME_HRS)).isoformat()
query = f"/{TABLE}?correct=is.null&ts=lt.{urllib.parse.quote(cutoff_iso)}" \
        "&select=id,ts,direction,entry_price&order=ts.asc&limit=200"

unresolved = _supa("GET", query) or []
print(f"Resolving {len(unresolved)} signals older than 72h…")

# Bulk-fetch exact-horizon exit prices: {row ts: price at ts+72h}
exact_exit = {}
if _price_hist is not None and unresolved:
    targets = {}
    for r in unresolved:
        ts = r.get("ts")
        if ts:
            try:
                targets[ts] = datetime.fromisoformat(ts) + timedelta(hours=OUTCOME_HRS)
            except Exception:
                pass
    if targets:
        try:
            bulk = _price_hist.btc_prices_at_bulk(list(targets.values()))
            exact_exit = {ts: bulk.get(tgt) for ts, tgt in targets.items()}
            n_hit = sum(1 for v in exact_exit.values() if v)
            print(f"  exact-horizon prices: {n_hit}/{len(targets)} found")
        except Exception as e:
            print(f"  ⚠ exact-horizon fetch failed ({type(e).__name__}: {e}) — using spot")

for r in unresolved:
    ep = r.get("entry_price")
    d  = r.get("direction")
    rid = r.get("id")
    if not ep or not d or not rid:
        continue
    xp = exact_exit.get(r.get("ts")) or spot_price
    if not xp:
        continue   # no exact candle AND no spot — leave unresolved, retry next run
    pct = (xp - float(ep)) / float(ep) * 100
    dir_right = (d == "LONG" and pct > 0) or (d == "SHORT" and pct < 0)
    if d == "HOLD":
        correct = "N/A"
    elif dir_right and abs(pct) >= 3.0:
        correct = "2"
    elif dir_right:
        correct = "1"
    else:
        correct = "0"
    patch = {
        "exit_price": round(xp, 2),
        "pct_move":   round(pct, 2),
        "correct":    correct,
    }
    res = _supa("PATCH", f"/{TABLE}?id=eq.{rid}", body=patch)
    if res is not None:
        print(f"  ✓ resolved id={rid} pct={pct:+.2f}% correct={correct}")
    else:
        print(f"  ✗ patch failed for id={rid}")


# ─────────────────────────────────────────────────────────────────
# 2. Compute current 72h bias + price
# ─────────────────────────────────────────────────────────────────
print("Computing 72h bias…")
result = run_analysis("BTC-USD")
if not result:
    print("ERROR: run_analysis returned empty dict")
    sys.exit(3)

price    = float(result.get("price") or 0)
bias_72h = result.get("bias_72h") or {}
score    = float(bias_72h.get("score") or 0)
label    = str(bias_72h.get("label") or "N/A")
# Anti-teleport on the charted series: clamp this tick's change vs the last
# durable Supabase row (shared limiter so cron + local app stay identical).
# This is the authoritative server-side writer, so it's where the slew limit
# matters most. Best-effort — returns score unchanged if the read fails.
_slew = getattr(mod, "_slew_limit_score", None)
if _slew is not None:
    _raw_score = score
    score = float(_slew(score))
    if abs(score - _raw_score) >= 0.05:
        print(f"  · slew-limited score {_raw_score:+.1f} → {score:+.1f}")
direction = "LONG" if score >= 25 else ("SHORT" if score <= -25 else "HOLD")

# Shadow re-centered score — de-biases the persistent ~-8 offset for auditing.
# Shared helper so cron + local app produce an identical series. Does NOT affect
# `direction` above (live call stays on the raw score). Best-effort: passthrough
# (offset 0) if the helper or its Supabase read fails.
_recenter = getattr(mod, "_recenter_offset", None)
_rc_off = 0.0
if _recenter is not None:
    try:
        _rc_off = float(_recenter())
    except Exception:
        _rc_off = 0.0
score_recentered = round(score - _rc_off, 1)

print(f"  price=${price:,.2f}  score={score:+.1f}  label={label}  direction={direction}"
      f"  score_rc={score_recentered:+.1f} (offset {_rc_off:+.1f})")


# ─────────────────────────────────────────────────────────────────
# 2. Log every tick (matches the local app — chart needs continuity;
#    calibration buckets filter to |score|>=25 themselves).
# ─────────────────────────────────────────────────────────────────
def _rnd(v, n):
    try:
        return round(float(v), n) if v is not None else None
    except Exception:
        return None

bias_24h = result.get("bias_24h") or {}
_regime  = bias_72h.get("regime")
row = {
    "ts":          datetime.now(timezone.utc).isoformat(),
    "score":       round(score, 1),
    "label":       label,
    "direction":   direction,
    "entry_price": round(price, 2),
    "exit_price":  None,
    "pct_move":    None,
    "correct":     None,
    # Phase-B fields — previously only the local app wrote these, so cron rows
    # were unusable for meta-model training / calibration. Now logged here too.
    "bull_prob":      _rnd(bias_72h.get("bull_prob"), 2),
    "conviction":     _rnd(bias_72h.get("conviction"), 4),
    "regime":         str(_regime) if _regime is not None else None,
    "score_24h":      _rnd(bias_24h.get("score"), 1),
    # 2-signal audit baseline (EMA Structure + OI Funding) on the same scale.
    "score_baseline": _rnd(bias_72h.get("score_baseline"), 1),
    # Shadow de-biased score (audit 2026-06-23) — see _recenter_offset.
    "score_recentered": score_recentered,
    "recenter_offset":  round(_rc_off, 1),
}
inserted = _supa("POST", f"/{TABLE}", body=row)
if inserted is None:
    # Schema lag → 4xx on the extended row. Drop ONLY the newest columns first
    # (score_recentered/recenter_offset migration not run yet) so we don't also
    # lose the Phase-B fields that DO exist. Core-fields is the last-resort net.
    _row2 = {k: v for k, v in row.items()
             if k not in ("score_recentered", "recenter_offset")}
    inserted = _supa("POST", f"/{TABLE}", body=_row2)
    if inserted:
        print("  ⚠ shadow-column insert failed — logged without "
              "score_recentered/recenter_offset (run supabase_migrations.sql)")
    else:
        core_fields = ["ts", "score", "label", "direction",
                       "entry_price", "exit_price", "pct_move", "correct"]
        inserted = _supa("POST", f"/{TABLE}", body={k: row[k] for k in core_fields})
        if inserted:
            print("  ⚠ extended insert failed — core-fields fallback succeeded "
                  "(run supabase_migrations.sql to add new columns)")
if inserted:
    print(f"  ✓ logged signal id={inserted[0].get('id') if isinstance(inserted, list) else '?'}")
else:
    # POST failed after all retries — accept the lost row and continue to
    # outcome resolution. Exiting non-zero here would skip resolution AND
    # email the user every time Supabase has a transient blip.
    print("  ⚠ Supabase insert failed after retries — continuing to resolution")


# ─────────────────────────────────────────────────────────────────
# 2b. Phase 0 — per-signal detail rows (signal_detail table)
#     Enables IC weighting, SHAP, and rolling performance later.
#     Best-effort: missing table / failures don't fail the run.
# ─────────────────────────────────────────────────────────────────
_ts = row["ts"]
try:
    sigs = bias_72h.get("signals") or {}
    wts  = bias_72h.get("weights") or {}
    detail_rows = []
    for name, tup in sigs.items():
        try:
            raw_v  = float(tup[0]) if tup else 0.0
            w_v    = float(wts.get(name, 0.0))
            contr  = raw_v * w_v * 100
            detail_rows.append({
                "ts":           _ts,
                "signal_name":  str(name),
                "raw_value":    round(raw_v, 4),
                "weight":       round(w_v, 4),
                "contribution": round(contr, 4),
            })
        except Exception:
            continue
    if detail_rows:
        det_res = _supa("POST", "/signal_detail", body=detail_rows)
        if det_res is not None:
            print(f"  ✓ logged {len(detail_rows)} signal_detail rows")
        else:
            print(f"  ⚠ signal_detail insert failed (table missing?) — continuing")
except Exception as e:
    print(f"  ⚠ signal_detail error: {type(e).__name__}: {e} — continuing")


# ─────────────────────────────────────────────────────────────────
# 2c. Phase 2 prep — Polymarket strike probabilities (polymarket_log)
#     Enables probability velocity / acceleration analysis later.
#     Best-effort.
# ─────────────────────────────────────────────────────────────────
try:
    poly_data = result.get("poly_sentiment") or {}
    poly_mkts = poly_data.get("markets") or []
    poly_rows = []
    for m in poly_mkts:
        try:
            q       = str(m.get("question") or m.get("event") or "")[:200]
            mkt_sc  = round(float(m.get("score", 0.0)), 2)
            mkt_w   = float(m.get("weight", 0))
            for b in (m.get("buckets") or []):
                if not b or len(b) < 2:
                    continue
                lbl       = str(b[0])[:80]
                prob_val  = float(b[1])
                is_bull   = None
                if len(b) >= 3 and b[2] is not None:
                    is_bull = bool(b[2])
                poly_rows.append({
                    "ts":          _ts,
                    "question":    q,
                    "strike_lbl":  lbl,
                    "probability": round(prob_val, 4),
                    "is_bull":     is_bull,
                    "mkt_score":   mkt_sc,
                    "mkt_weight":  mkt_w,
                })
        except Exception:
            continue
    if poly_rows:
        for i in range(0, len(poly_rows), 100):
            _supa("POST", "/polymarket_log", body=poly_rows[i:i+100])
        print(f"  ✓ logged {len(poly_rows)} polymarket_log rows")
except Exception as e:
    print(f"  ⚠ polymarket_log error: {type(e).__name__}: {e} — continuing")


# ─────────────────────────────────────────────────────────────────
# 2d. liq_heatmap_log — top synthetic-liq-map bins, every ~15 min.
#     Powers the time×price heatmap panel (Coinglass-style) so the time
#     axis is populated even when the local app was closed. Best-effort.
# ─────────────────────────────────────────────────────────────────
try:
    if datetime.now(timezone.utc).minute % 15 < 5:
        _lm = (result.get("btc_liq") or {}).get("liq_map") or {}
        hm_rows = []
        for _side in ("long", "short"):
            for _p, _u in (_lm.get(_side) or [])[:25]:
                hm_rows.append({"ts": _ts, "side": _side,
                                "price": round(float(_p), 2),
                                "usd":   round(float(_u), 2)})
        if hm_rows:
            # One 'px' row per snapshot carries spot for the price-line overlay.
            hm_rows.append({"ts": _ts, "side": "px",
                            "price": round(price, 2), "usd": 0})
            res_hm = _supa("POST", "/liq_heatmap_log", body=hm_rows)
            if res_hm is not None:
                print(f"  ✓ logged {len(hm_rows)} liq_heatmap_log rows")
            else:
                print("  ⚠ liq_heatmap_log insert failed (table missing? run migration) — continuing")
except Exception as e:
    print(f"  ⚠ liq_heatmap_log error: {type(e).__name__}: {e} — continuing")


# ─────────────────────────────────────────────────────────────────
# 3. Dead-man's switch — ping a healthcheck URL on successful completion.
#    Set HEALTHCHECK_URL (e.g. a healthchecks.io check) as a repo secret;
#    if the cron silently dies, the service alerts instead of nobody noticing
#    (resolution was dead 3–8 Jun and was only found by manual audit).
#    Best-effort: a ping failure never fails the run.
# ─────────────────────────────────────────────────────────────────
_hc_url = os.environ.get("HEALTHCHECK_URL", "").strip()
if _hc_url:
    try:
        with urllib.request.urlopen(_hc_url, timeout=10) as _r:
            print(f"  ✓ healthcheck ping ({_r.status})")
    except Exception as e:
        print(f"  ⚠ healthcheck ping failed: {type(e).__name__}: {e}")

print("Done.")
