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


SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or ""
TABLE        = "signal_log"
OUTCOME_HRS  = 72.0

if not (SUPABASE_URL and SUPABASE_KEY):
    print("ERROR: SUPABASE_URL or SUPABASE_KEY env var missing")
    sys.exit(1)


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
# 1. Compute current 72h bias + price
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
direction = "LONG" if score >= 25 else ("SHORT" if score <= -25 else "HOLD")

print(f"  price=${price:,.2f}  score={score:+.1f}  label={label}  direction={direction}")


# ─────────────────────────────────────────────────────────────────
# 2. Log every tick (matches the local app — chart needs continuity;
#    calibration buckets filter to |score|>=25 themselves).
# ─────────────────────────────────────────────────────────────────
row = {
    "ts":          datetime.now(timezone.utc).isoformat(),
    "score":       round(score, 1),
    "label":       label,
    "direction":   direction,
    "entry_price": round(price, 2),
    "exit_price":  None,
    "pct_move":    None,
    "correct":     None,
}
inserted = _supa("POST", f"/{TABLE}", body=row)
if inserted:
    print(f"  ✓ logged signal id={inserted[0].get('id') if isinstance(inserted, list) else '?'}")
else:
    # POST failed after all retries — accept the lost row and continue to
    # outcome resolution. Exiting non-zero here would skip resolution AND
    # email the user every time Supabase has a transient blip.
    print("  ⚠ Supabase insert failed after retries — continuing to resolution")


# ─────────────────────────────────────────────────────────────────
# 3. Resolve outcomes for any rows older than 72h with correct=null
# ─────────────────────────────────────────────────────────────────
cutoff_iso = (datetime.now(timezone.utc) - timedelta(hours=OUTCOME_HRS)).isoformat()
query = f"/{TABLE}?correct=is.null&ts=lt.{urllib.parse.quote(cutoff_iso)}" \
        "&select=id,ts,direction,entry_price&limit=200"

unresolved = _supa("GET", query) or []
print(f"Resolving {len(unresolved)} signals older than 72h…")

for r in unresolved:
    ep = r.get("entry_price")
    d  = r.get("direction")
    rid = r.get("id")
    if not ep or not d or not rid:
        continue
    pct = (price - float(ep)) / float(ep) * 100
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
        "exit_price": round(price, 2),
        "pct_move":   round(pct, 2),
        "correct":    correct,
    }
    res = _supa("PATCH", f"/{TABLE}?id=eq.{rid}", body=patch)
    if res is not None:
        print(f"  ✓ resolved id={rid} pct={pct:+.2f}% correct={correct}")
    else:
        print(f"  ✗ patch failed for id={rid}")

print("Done.")
