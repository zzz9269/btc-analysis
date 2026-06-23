"""
Self-check for the 72h-bias signal logger. Run this ANYTIME to confirm logging
is alive — no need to squint at the chart or trust anyone.

    python check_logging.py     (or double-click check_logging.bat)

It answers three questions, independently:
  1. How long since the last signal_log row?           (is the series fresh?)
  2. Any gaps > 12 min in the last 24h?                 (did it ever stall?)
  3. Did the SCHEDULED TASK (not the app) write recently? via liq_heatmap_log,
     a table the dashboard cannot write — so a fresh row here proves the
     background task ran on its own.
Plus it reads the Windows task's last run result.
"""
import json, subprocess, sys, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime, timezone, timedelta

SGT = timezone(timedelta(hours=8))
OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"

# ── creds from .streamlit/secrets.toml (same place the logger reads) ──
u = k = ""
secrets = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
for ln in secrets.read_text(encoding="utf-8").splitlines():
    ln = ln.strip()
    if ln.startswith("SUPABASE_URL"):
        u = ln.split("=", 1)[1].strip().strip('"').strip("'").rstrip("/")
    if ln.startswith("SUPABASE_KEY"):
        k = ln.split("=", 1)[1].strip().strip('"').strip("'")
if not (u and k):
    print(f"{BAD} Could not read Supabase creds from {secrets}")
    sys.exit(1)


def get(path):
    req = urllib.request.Request(u + "/rest/v1" + path,
        headers={"apikey": k, "Authorization": "Bearer " + k})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def fmt(ts):
    return datetime.fromisoformat(ts).astimezone(SGT).strftime("%m-%d %H:%M:%S SGT")


print("=" * 60)
print("  72h-BIAS LOGGER HEALTH CHECK")
print("=" * 60)

# 1) freshness of signal_log
try:
    last = get("/signal_log?select=ts,score&order=ts.desc&limit=1")
    ts = datetime.fromisoformat(last[0]["ts"])
    age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
    tag = OK if age <= 12 else (WARN if age <= 30 else BAD)
    print(f"\n{tag} Last signal_log row: {fmt(last[0]['ts'])}  "
          f"({age:.1f} min ago, score={last[0]['score']})")
    if age > 12:
        print("       Expected a row every ~5 min. If this keeps growing, check:")
        print("       PC on? plugged in? logged in (not signed out)? internet up?")
except Exception as e:
    print(f"{BAD} Could not reach Supabase signal_log: {type(e).__name__}: {e}")

# 2) gaps in the last 24h
try:
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    rows = get("/signal_log?select=ts&ts=gte." + urllib.parse.quote(since)
               + "&order=ts.asc&limit=2000")
    gaps, prev = [], None
    for r in rows:
        t = datetime.fromisoformat(r["ts"])
        if prev and (t - prev).total_seconds() / 60 > 12:
            gaps.append((prev, t, (t - prev).total_seconds() / 60))
        prev = t
    if not gaps:
        print(f"\n{OK} No gaps > 12 min in the last 24h ({len(rows)} rows).")
    else:
        print(f"\n{WARN} {len(gaps)} gap(s) > 12 min in the last 24h:")
        for a, b, d in gaps:
            print(f"       {d/60:.1f}h : {fmt(a.isoformat())} -> {fmt(b.isoformat())}")
except Exception as e:
    print(f"{BAD} Gap check failed: {type(e).__name__}: {e}")

# 3) task-exclusive proof: liq_heatmap_log (app cannot write this)
try:
    hm = get("/liq_heatmap_log?select=ts&order=ts.desc&limit=1")
    if hm:
        ts = datetime.fromisoformat(hm[0]["ts"])
        age = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        # written every ~15 min, so allow more slack
        tag = OK if age <= 25 else (WARN if age <= 60 else BAD)
        print(f"\n{tag} Background TASK proof (liq_heatmap_log): {fmt(hm[0]['ts'])}  "
              f"({age:.1f} min ago)")
        print("       (the dashboard cannot write this table — a fresh row here")
        print("        means the scheduled task is running on its own.)")
    else:
        print(f"\n{WARN} liq_heatmap_log empty — task may not have run a 15-min mark yet.")
except Exception as e:
    print(f"{WARN} liq_heatmap_log check skipped: {type(e).__name__}: {e}")

# 4) Windows scheduled task last result
try:
    out = subprocess.run(
        ["schtasks", "/Query", "/TN", "BTC Signal Logger", "/V", "/FO", "LIST"],
        capture_output=True, text=True, timeout=15).stdout
    lr = next((l.split(":", 1)[1].strip() for l in out.splitlines()
               if l.strip().startswith("Last Run Time")), "?")
    res = next((l.split(":", 1)[1].strip() for l in out.splitlines()
                if l.strip().startswith("Last Result")), "?")
    nxt = next((l.split(":", 1)[1].strip() for l in out.splitlines()
                if l.strip().startswith("Next Run Time")), "?")
    print(f"\n{OK} Scheduled task: last run {lr} (result {res}), next {nxt}")
    print("       (result 0 = the launcher fired; the Supabase checks above")
    print("        are what confirm a row was actually written.)")
except Exception as e:
    print(f"{WARN} Could not query scheduled task: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
