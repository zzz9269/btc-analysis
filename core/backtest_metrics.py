"""
Industry-grade evaluation metrics for the 72h bias engine.

Design principles (why each choice is defensible under scrutiny):

1. EPISODE is the unit of observation, not the 5-min tick. Ticks on a 72h
   horizon are ~99.9% autocorrelated; treating them as independent inflates
   n by ~100x and makes any CI a lie. An episode = one continuous
   same-direction signal run.

2. The NULL is the drift/persistence baseline, NOT 50%. In a market with a
   trend, "always short" or "always long" already beats a coin flip, so skill
   = winrate MINUS the relevant constant-rule baseline. We surface those
   baselines explicitly. (If the engine only ever fires one direction, it is
   mathematically identical to that constant rule and has zero measurable
   directional edge — the code reports this honestly.)

3. EXPECTANCY (mean signed return), not hit-rate, is the headline. Binary
   win/loss discards magnitude; a strategy can be 80% "right" and lose money
   if the 20% are large. We report mean signed return per episode with a
   bootstrap CI, and the EXCESS over the always-short baseline.

4. UNCERTAINTY via episode-level bootstrap, so CIs reflect the true (small)
   number of independent observations.

5. PROBABILISTIC skill via the Brier skill score on bull_prob — strictly
   proper, threshold-free, and benchmarked against the base rate.

6. MIN-N GATING everywhere: a metric returns None rather than a noisy number
   when its sample is too small.

All functions are pure (numpy/scipy only) and unit-tested.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

MIN_EPISODES_CI   = 4    # below this, no bootstrap CI (too few to mean anything)
MIN_N_IC          = 30   # below this, no information coefficient
MIN_N_BRIER       = 25   # below this, no Brier skill score
MIN_EPISODES_BASE = 4    # below this, no engine-vs-baseline delta


def _parse(rows: list) -> list:
    """Normalize raw signal_log rows → sorted list of resolved LONG/SHORT dicts."""
    out = []
    for r in rows:
        if r.get("correct") not in ("0", "1", "2"):
            continue
        if r.get("direction") not in ("LONG", "SHORT"):
            continue
        try:
            ts = datetime.fromisoformat(r["ts"])
            pm = float(r["pct_move"])
            sc = float(r["score"])
        except Exception:
            continue
        def _f(v):
            try:
                return float(v) if v not in (None, "", "None") else None
            except Exception:
                return None
        out.append({"ts": ts, "dir": r["direction"], "pm": pm, "score": sc,
                    "bull_prob": _f(r.get("bull_prob")),
                    "score_baseline": _f(r.get("score_baseline"))})
    out.sort(key=lambda x: x["ts"])
    return out


def _episodes(parsed: list, gap_hours: float = 6.0) -> list:
    """Collapse consecutive same-direction signals (<gap_hours apart) into
    episodes, the independent unit of observation."""
    eps, cur = [], None
    for s in parsed:
        sign   = 1 if s["dir"] == "LONG" else -1
        signed = sign * s["pm"]
        if cur and s["dir"] == cur["dir"] and \
           (s["ts"] - cur["end"]).total_seconds() < gap_hours * 3600:
            cur["end"] = s["ts"]
            cur["pm"].append(s["pm"]); cur["signed"].append(signed)
            cur["score"].append(s["score"]); cur["wins"] += int(signed > 0)
        else:
            if cur:
                eps.append(cur)
            cur = {"dir": s["dir"], "start": s["ts"], "end": s["ts"],
                   "pm": [s["pm"]], "signed": [signed], "score": [s["score"]],
                   "wins": int(signed > 0)}
    if cur:
        eps.append(cur)
    for e in eps:
        e["n"]           = len(e["pm"])
        e["mean_signed"] = float(np.mean(e["signed"]))
        e["mean_pm"]     = float(np.mean(e["pm"]))
        e["win"]         = e["wins"] / e["n"] > 0.5
    return eps


def _boot_ci(values, stat=np.mean, B: int = 10000, alpha: float = 0.05,
             seed: int = 0) -> "tuple":
    """Percentile bootstrap CI over independent observations (episodes)."""
    v = np.asarray(values, dtype=float)
    if len(v) < MIN_EPISODES_CI:
        return (None, None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    bs  = stat(v[idx], axis=1)
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(lo), 3), round(float(hi), 3))


def evaluate(rows: list, gap_hours: float = 6.0) -> dict:
    """Full evaluation. Returns a flat dict of metrics + honest verdict.
    Every value is None-gated when its sample is too small."""
    p = _parse(rows)
    out = {"n_signals": len(p)}
    if not p:
        return out
    eps = _episodes(p, gap_hours)
    pm  = np.array([s["pm"] for s in p])
    sc  = np.array([s["score"] for s in p])
    n_long  = sum(1 for s in p if s["dir"] == "LONG")
    n_short = len(p) - n_long

    # ── Winrates (tick + episode) ───────────────────────────────────────
    tick_signed = np.array([(1 if s["dir"] == "LONG" else -1) * s["pm"] for s in p])
    ep_outcomes = np.array([1.0 if e["win"] else 0.0 for e in eps])

    # ── Drift / persistence nulls (the correct benchmark, not 50%) ──────
    always_short_wr = float(np.mean(pm < 0))
    always_long_wr  = float(np.mean(pm > 0))
    # The relevant null = the constant rule matching the engine's actual bias.
    drift_null_wr   = always_short_wr if n_short >= n_long else always_long_wr

    # ── Expectancy (episode-level) + bootstrap CI ───────────────────────
    ep_signed = np.array([e["mean_signed"] for e in eps])
    expectancy    = float(np.mean(ep_signed))
    expectancy_ci = _boot_ci(ep_signed)

    # ── Excess expectancy vs always-short (per episode) ─────────────────
    # always-short signed return = −pm; engine − baseline:
    #   short episode → 0 (engine IS always-short there)
    #   long  episode → 2·mean_pm
    ep_excess = np.array([(2 * e["mean_pm"] if e["dir"] == "LONG" else 0.0) for e in eps])
    excess    = float(np.mean(ep_excess))
    excess_ci = _boot_ci(ep_excess)

    # ── Information coefficient (resolved subset — flagged as confounded) ─
    ic = None
    if len(p) >= MIN_N_IC and np.ptp(sc) > 0:
        from scipy.stats import spearmanr
        _ic = spearmanr(sc, pm).statistic
        ic = round(float(_ic), 3) if _ic == _ic else None  # NaN guard

    # ── Brier skill score on bull_prob (strictly proper, vs base rate) ──
    bp_pairs = [(s["bull_prob"], 1.0 if s["pm"] > 0 else 0.0)
                for s in p if s["bull_prob"] is not None]
    brier = brier_skill = None
    n_bp = len(bp_pairs)
    if n_bp >= MIN_N_BRIER:
        raw   = np.array([b for b, _ in bp_pairs], dtype=float)
        # Detect scale ARRAY-WIDE, not per value: a single value of 1.0 is
        # ambiguous (1% on a 0-100 scale vs 100% on a 0-1 scale), but if any
        # value exceeds 1 the whole series is 0-100.
        probs = raw / 100.0 if raw.max() > 1.0 else raw
        outc  = np.array([o for _, o in bp_pairs])
        brier = float(np.mean((probs - outc) ** 2))
        base  = float(np.mean(outc))
        brier_base = float(np.mean((base - outc) ** 2))
        # Brier skill score: 1 = perfect, 0 = no better than always-predicting
        # the base rate, <0 = worse than the base rate.
        brier_skill = round(1 - brier / brier_base, 3) if brier_base > 1e-9 else None

    # ── Honest verdict ──────────────────────────────────────────────────
    if n_long == 0 or n_short == 0:
        verdict = ("UNMEASURABLE (one-directional): all signals are "
                   f"{'SHORT' if n_long == 0 else 'LONG'}, so the engine is "
                   "mathematically identical to that constant rule — no "
                   "directional edge can be measured until it fires both ways.")
        vcode = "unmeasurable"
    elif excess_ci[0] is None:
        verdict = f"INCONCLUSIVE: only {len(eps)} episodes — too few for a CI."
        vcode = "inconclusive"
    elif excess_ci[0] > 0:
        verdict = (f"EDGE DEMONSTRATED: excess return over always-short is "
                   f"{excess:+.2f}% (95% CI {excess_ci[0]:+.2f}..{excess_ci[1]:+.2f}, "
                   f"excludes 0).")
        vcode = "edge"
    elif excess_ci[1] < 0:
        verdict = (f"NEGATIVE EDGE: engine is worse than always-short "
                   f"({excess:+.2f}%, CI {excess_ci[0]:+.2f}..{excess_ci[1]:+.2f}).")
        vcode = "negative"
    else:
        verdict = (f"NO EDGE DEMONSTRATED: excess over always-short {excess:+.2f}% "
                   f"(95% CI {excess_ci[0]:+.2f}..{excess_ci[1]:+.2f}, spans 0).")
        vcode = "no_edge"

    out.update({
        "n_long": n_long, "n_short": n_short,
        "n_episodes": len(eps), "n_ep_wins": int(ep_outcomes.sum()),
        "tick_winrate":    round(float(np.mean(tick_signed > 0)) * 100, 0),
        "episode_winrate": round(float(np.mean(ep_outcomes)) * 100, 0) if len(eps) else None,
        "always_short_wr": round(always_short_wr * 100, 0),
        "always_long_wr":  round(always_long_wr * 100, 0),
        "drift_null_wr":   round(drift_null_wr * 100, 0),
        "edge_vs_drift_pp": round((float(np.mean(tick_signed > 0)) - drift_null_wr) * 100, 1),
        "expectancy": round(expectancy, 2), "expectancy_ci": expectancy_ci,
        "excess_vs_short": round(excess, 2), "excess_ci": excess_ci,
        "avg_win":  round(float(tick_signed[tick_signed > 0].mean()), 2) if (tick_signed > 0).any() else None,
        "avg_loss": round(float(tick_signed[tick_signed < 0].mean()), 2) if (tick_signed < 0).any() else None,
        "ic_resolved": ic, "ic_score_range": (round(float(sc.min()), 0), round(float(sc.max()), 0)),
        "brier": round(brier, 4) if brier is not None else None,
        "brier_skill": brier_skill, "n_bull_prob": n_bp,
        "verdict": verdict, "verdict_code": vcode,
    })
    return out


def baseline_comparison(rows: list, gap_hours: float = 6.0) -> dict:
    """Engine vs 2-signal EMA+OI baseline (score_baseline), episode-level.
    Gated: returns n_episodes only until MIN_EPISODES_BASE baseline episodes
    exist (the +50pp-on-1-episode trap)."""
    bl = []
    for r in rows:
        sb = r.get("score_baseline"); pm = r.get("pct_move")
        if sb in (None, "", "None") or pm in (None, "", "None"):
            continue
        try:
            sb = float(sb); pm = float(pm); ts = datetime.fromisoformat(r["ts"])
        except Exception:
            continue
        d = "LONG" if sb >= 25 else ("SHORT" if sb <= -25 else None)
        if d is None:
            continue
        bl.append({"ts": ts, "dir": d, "pm": pm, "correct": "1"})
    bl.sort(key=lambda x: x["ts"])
    # reuse episode collapsing via a tiny adapter
    rows_adapter = [{"ts": s["ts"].isoformat(), "direction": s["dir"],
                     "pct_move": s["pm"],
                     "correct": "1" if ((s["dir"] == "LONG") == (s["pm"] > 0)) else "0",
                     "score": 0} for s in bl]
    eps = _episodes(_parse(rows_adapter), gap_hours)
    n = len(eps)
    ready = n >= MIN_EPISODES_BASE
    return {
        "n_episodes": n,
        "ready": ready,
        "ep_winrate": round(sum(1 for e in eps if e["win"]) / n * 100, 0) if ready else None,
    }
