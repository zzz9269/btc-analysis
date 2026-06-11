"""
Phase 8 — Meta-model on top of the hand-crafted 72h bias engine.

Per improvements.txt item 8: the engine's outputs become FEATURES for a
second-layer ML model that learns when to trust the engine and when not to.

Design (locked in 2026-06-07 with user):
  - Model:    LogisticRegression (Phase v1). Switch to LightGBM once N >= ~5000.
  - Target:   binary up/down (sign of pct_move at horizon).
  - Features: top-level Phase B fields only — score, score_24h, bull_prob,
              conviction, regime (one-hot). Per-signal contributions deferred to v2.
  - CV:       TimeSeriesSplit(5) — random k-fold leaks future data on time series.
  - Cadence:  on-demand train via Streamlit button. Weekly cron is the long-term plan.
  - Output:   P(up) per tick. Displayed as a "second-opinion" chip beside the engine
              gauge — never replaces engine score.

The module is intentionally dependency-light (sklearn + joblib + numpy + pandas).
All functions return plain dicts so the main app can call them without import-time
side effects.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import math


_MODEL_DIR  = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "meta_logreg_v1.joblib"
_META_PATH  = _MODEL_DIR / "meta_logreg_v1.meta.json"

# Minimum resolved-outcome rows before training is allowed. With Phase-B
# logging cadence (~3 resolved trades/day) this is ~5 weeks. Below this,
# CV results are too noisy to mean anything.
MIN_TRAIN_N = 100

# Feature order — frozen for v1. Adding a feature requires a model retrain
# AND a name bump (meta_logreg_v2) so old saved models don't silently break.
FEATURE_NAMES_V1 = [
    "score",
    "score_24h",
    "bull_prob",
    "conviction",
    "regime_trend",       # one-hot
    "regime_range",       # one-hot
    "regime_transition",  # one-hot
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def _safe_float(v, default=None):
    try:
        if v in (None, ""):
            return default
        return float(v)
    except Exception:
        return default


def _row_to_features(row: dict) -> "list[float] | None":
    """Convert one signal_log row into the v1 feature vector.
    Returns None if any required field is missing — caller drops the row."""
    score      = _safe_float(row.get("score"))
    score_24h  = _safe_float(row.get("score_24h"))
    bull_prob  = _safe_float(row.get("bull_prob"))
    conviction = _safe_float(row.get("conviction"))
    regime     = (row.get("regime") or "").strip().lower()
    if None in (score, score_24h, bull_prob, conviction) or regime not in ("trend", "range", "transition"):
        return None
    return [
        score,
        score_24h,
        bull_prob,
        conviction,
        1.0 if regime == "trend"      else 0.0,
        1.0 if regime == "range"      else 0.0,
        1.0 if regime == "transition" else 0.0,
    ]


def extract_training_set(rows: list) -> "tuple[list[list[float]], list[int], list[str]] | tuple[None, None, None]":
    """Build (X, y, timestamps) from resolved signal_log rows.

    Target: 1 if pct_move > 0 else 0  (market closed up at horizon).
    Excludes HOLD rows and any row missing a v1 feature.
    Timestamps returned so caller can verify ordering before TimeSeriesSplit.
    """
    if not rows:
        return None, None, None
    X, y, ts = [], [], []
    for r in rows:
        if r.get("correct") not in ("0", "1", "2"):
            continue
        if r.get("direction") not in ("LONG", "SHORT"):
            continue
        pm = _safe_float(r.get("pct_move"))
        if pm is None:
            continue
        feats = _row_to_features(r)
        if feats is None:
            continue
        X.append(feats)
        y.append(1 if pm > 0 else 0)
        ts.append(r.get("ts") or "")
    if not X:
        return None, None, None
    # Sort by ts so TimeSeriesSplit gets monotonically increasing time.
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    X  = [X[i] for i in order]
    y  = [y[i] for i in order]
    ts = [ts[i] for i in order]
    return X, y, ts


# ============================================================
# TRAINING
# ============================================================

def train(X: list, y: list) -> dict:
    """Train LogisticRegression with TimeSeriesSplit CV.

    Returns dict with: model, cv_auc_mean, cv_auc_std, n_train, feature_names, trained_at.
    On insufficient data, returns dict with error and no model.
    """
    if X is None or len(X) < MIN_TRAIN_N:
        return {
            "error":    f"need >= {MIN_TRAIN_N} resolved rows (have {0 if X is None else len(X)})",
            "model":    None,
            "n_train":  0 if X is None else len(X),
        }

    # Local imports keep module-load cheap when sklearn isn't installed yet.
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        return {"error": f"sklearn not installed: {e}", "model": None, "n_train": len(X)}

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)

    if len(set(y_arr.tolist())) < 2:
        return {"error": "only one class present (all up or all down)", "model": None, "n_train": len(X)}

    # CV: 5 expanding-window splits, last fold's score reported as the headline.
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    for tr_idx, va_idx in tscv.split(X_arr):
        if len(set(y_arr[tr_idx].tolist())) < 2 or len(set(y_arr[va_idx].tolist())) < 2:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(max_iter=1000, C=1.0)),
        ])
        pipe.fit(X_arr[tr_idx], y_arr[tr_idx])
        prob = pipe.predict_proba(X_arr[va_idx])[:, 1]
        aucs.append(roc_auc_score(y_arr[va_idx], prob))

    # Final model on ALL data — this is what gets saved + used live.
    final = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, C=1.0)),
    ])
    final.fit(X_arr, y_arr)

    return {
        "model":         final,
        "cv_auc_mean":   float(sum(aucs) / len(aucs)) if aucs else None,
        "cv_auc_std":    float((sum((a - sum(aucs)/len(aucs))**2 for a in aucs) / len(aucs))**0.5) if len(aucs) >= 2 else None,
        "cv_fold_aucs":  [float(a) for a in aucs],
        "n_train":       int(len(X)),
        "feature_names": list(FEATURE_NAMES_V1),
        "trained_at":    datetime.utcnow().isoformat() + "Z",
    }


# ============================================================
# PERSISTENCE
# ============================================================

def save_model(result: dict) -> bool:
    """Save model + metadata to disk. Returns True on success."""
    if result.get("model") is None:
        return False
    try:
        import joblib
    except ImportError:
        return False
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(result["model"], _MODEL_PATH)
        meta = {k: v for k, v in result.items() if k != "model"}
        _META_PATH.write_text(json.dumps(meta, indent=2))
        return True
    except Exception:
        return False


def load_model() -> "dict | None":
    """Load saved model + metadata. Returns None if not found."""
    if not _MODEL_PATH.exists():
        return None
    try:
        import joblib
        model = joblib.load(_MODEL_PATH)
        meta = {}
        if _META_PATH.exists():
            try:
                meta = json.loads(_META_PATH.read_text())
            except Exception:
                meta = {}
        meta["model"] = model
        return meta
    except Exception:
        return None


# ============================================================
# LIVE PREDICTION
# ============================================================

def predict(model, live_features: dict) -> "float | None":
    """Return P(up) for the current tick. live_features must match FEATURE_NAMES_V1.

    Returns None on missing features or model error so caller can fall back to engine score.
    """
    if model is None:
        return None
    score      = live_features.get("score")
    score_24h  = live_features.get("score_24h")
    bull_prob  = live_features.get("bull_prob")
    conviction = live_features.get("conviction")
    regime     = (live_features.get("regime") or "").strip().lower()
    if None in (score, score_24h, bull_prob, conviction) or regime not in ("trend", "range", "transition"):
        return None
    feats = [[
        float(score),
        float(score_24h),
        float(bull_prob),
        float(conviction),
        1.0 if regime == "trend"      else 0.0,
        1.0 if regime == "range"      else 0.0,
        1.0 if regime == "transition" else 0.0,
    ]]
    try:
        return float(model.predict_proba(feats)[0][1])
    except Exception:
        return None
