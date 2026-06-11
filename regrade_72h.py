"""
Retroactive re-grading of signal_log outcomes at the exact 72h horizon.

Historical rows were graded against the price at whatever moment the resolver
happened to run (≥72h after entry — sometimes 75h, 90h+). This script re-grades
every resolved row against the BTC price at exactly entry + 72h, using
core.price_history (Binance 5m klines, Bybit fallback).

Usage:
    python regrade_72h.py            # DRY RUN — prints what would change
    python regrade_72h.py --apply    # PATCH Supabase + rewrite signal_log.csv

Credentials: SUPABASE_URL / SUPABASE_KEY env vars, falling back to
.streamlit/secrets.toml. Local CSV is re-graded even without credentials.
"""
import csv
import json
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.price_history import btc_prices_at_bulk

HERE         = Path(__file__).parent
CSV_PATH     = HERE / "signal_log.csv"
OUTCOME_HRS  = 72.0
TABLE        = "signal_log"
APPLY        = "--apply" in sys.argv


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


def _supa(method, path, body=None):
    url = f"{SUPABASE_URL}/rest/v1{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def _grade(direction, pct):
    if direction == "HOLD":
        return "N/A"
    right = (direction == "LONG" and pct > 0) or (direction == "SHORT" and pct < 0)
    if right and abs(pct) >= 3.0:
        return "2"
    return "1" if right else "0"


def _regrade(rows, source):
    """rows: list of dicts with ts/direction/entry_price/exit_price/pct_move/correct.
    Returns list of (row, new_exit, new_pct, new_correct) for rows that change."""
    graded = [r for r in rows if r.get("correct") not in (None, "") and r.get("entry_price")]
    print(f"\n{source}: {len(graded)} resolved rows to re-grade")
    if not graded:
        return []
    targets = {r["ts"]: datetime.fromisoformat(r["ts"]) + timedelta(hours=OUTCOME_HRS)
               for r in graded}
    prices = btc_prices_at_bulk(list(targets.values()))
    exact  = {ts: prices.get(tgt) for ts, tgt in targets.items()}
    n_miss = sum(1 for v in exact.values() if not v)
    if n_miss:
        print(f"  [warn] no exact-horizon candle for {n_miss} rows — left unchanged")

    changes, flips = [], 0
    for r in graded:
        xp = exact.get(r["ts"])
        if not xp:
            continue
        ep      = float(r["entry_price"])
        new_pct = (xp - ep) / ep * 100
        new_cor = _grade(r.get("direction"), new_pct)
        old_cor = str(r.get("correct"))
        old_pct = r.get("pct_move")
        pct_changed = old_pct in (None, "") or abs(float(old_pct) - new_pct) >= 0.005
        if new_cor != old_cor or pct_changed:
            changes.append((r, xp, new_pct, new_cor))
            if new_cor != old_cor:
                flips += 1
                print(f"  GRADE FLIP  {r['ts']}  {r.get('direction'):5s} "
                      f"{old_cor} -> {new_cor}   pct {old_pct} -> {new_pct:+.2f}")
    print(f"  {len(changes)} rows change ({flips} grade flips, "
          f"{len(changes) - flips} pct-only adjustments)")
    return changes


def _resolve_backlog():
    """Resolve rows ≥72h old that were never graded at all (the cron's
    resolution stalled 3–8 Jun because the workflow timeout killed the job
    before the resolution section ran). Same exact-horizon grading."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=OUTCOME_HRS)).isoformat()
    rows, offset = [], 0
    while True:
        batch = _supa("GET", f"/{TABLE}?correct=is.null&ts=lt.{urllib.parse.quote(cutoff)}"
                             "&select=id,ts,direction,entry_price"
                             f"&order=ts.asc&limit=1000&offset={offset}") or []
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    print(f"\nSupabase unresolved backlog: {len(rows)} rows >= 72h old")
    if not rows:
        return
    targets = {r["ts"]: datetime.fromisoformat(r["ts"]) + timedelta(hours=OUTCOME_HRS)
               for r in rows if r.get("ts")}
    prices = btc_prices_at_bulk(list(targets.values()))
    exact  = {ts: prices.get(tgt) for ts, tgt in targets.items()}
    resolvable = [r for r in rows if exact.get(r.get("ts")) and r.get("entry_price")]
    print(f"  {len(resolvable)} resolvable with exact-horizon prices")
    if not APPLY:
        return
    done = 0
    for r in resolvable:
        xp  = exact[r["ts"]]
        ep  = float(r["entry_price"])
        pct = (xp - ep) / ep * 100
        _supa("PATCH", f"/{TABLE}?id=eq.{r['id']}",
              body={"exit_price": round(xp, 2), "pct_move": round(pct, 2),
                    "correct": _grade(r.get("direction"), pct)})
        done += 1
        if done % 200 == 0:
            print(f"  ... {done}/{len(resolvable)}")
    print(f"  OK: resolved {done} backlog rows")


def main():
    print(f"Mode: {'APPLY' if APPLY else 'DRY RUN (pass --apply to write)'}")

    # ── Supabase: clear never-resolved backlog first ─────────────
    if SUPABASE_URL and SUPABASE_KEY:
        _resolve_backlog()

    # ── Supabase: re-grade already-resolved rows ─────────────────
    supa_rows = []
    if SUPABASE_URL and SUPABASE_KEY:
        offset = 0
        while True:
            batch = _supa("GET", f"/{TABLE}?correct=not.is.null"
                                 "&select=id,ts,direction,entry_price,exit_price,pct_move,correct"
                                 f"&order=ts.asc&limit=1000&offset={offset}") or []
            supa_rows.extend(batch)
            if len(batch) < 1000:
                break
            offset += 1000
    else:
        print("[warn] no Supabase credentials — skipping remote re-grade")

    supa_changes = _regrade(supa_rows, "Supabase")
    if APPLY and supa_changes:
        for r, xp, pct, cor in supa_changes:
            _supa("PATCH", f"/{TABLE}?id=eq.{r['id']}",
                  body={"exit_price": round(xp, 2), "pct_move": round(pct, 2), "correct": cor})
        print(f"  OK: patched {len(supa_changes)} Supabase rows")

    # ── Local CSV ────────────────────────────────────────────────
    csv_changes = []
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader     = csv.DictReader(f)
            fieldnames = reader.fieldnames
            csv_rows   = list(reader)
        supa_ts     = {r.get("ts") for r in supa_rows}
        local_only  = [r for r in csv_rows if r.get("ts") not in supa_ts]
        csv_changes = _regrade(local_only, f"Local CSV (rows not in Supabase: {len(local_only)})")
        if APPLY and csv_changes:
            for r, xp, pct, cor in csv_changes:
                r["exit_price"] = str(round(xp, 2))
                r["pct_move"]   = f"{pct:+.2f}"
                r["correct"]    = cor
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, restval="", extrasaction="ignore")
                w.writeheader()
                w.writerows(csv_rows)
            print(f"  OK: rewrote {CSV_PATH.name} ({len(csv_changes)} rows updated)")

    total = len(supa_changes) + len(csv_changes)
    if not APPLY and total:
        print(f"\nDry run complete — {total} rows would change. Re-run with --apply to write.")
    elif not total:
        print("\nNothing to change — all grades already match the exact 72h horizon.")


if __name__ == "__main__":
    main()
