"""Recovery model and next-best-action selection.

The model estimates P(an attempt succeeds | payment, action, timing) from *logged attempts*
produced by the simulator in `simulation.py` (the only data available is simulated). It is a
gradient-boosted classifier trained with a time-based split and evaluated against a base-rate
model and against the simulator's own true probabilities (the best any model could do).

The agent picks the action with the best expected value across candidate actions and timings,
then `policy.py` decides whether it may run automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import simulation as sim
from .db import get_conn
from .policy import DEFAULT_POLICY

MODEL: HistGradientBoostingClassifier | None = None
MODEL_INFO: dict = {}

TRAIN_SEED = 11
N_TRAIN_PAYMENTS = 60000
TRAIN_DAYS = 120
SPLIT_DAY = 90          # first 90 days train, the last 30 days are held out

NUMERIC = ["amount_log", "customer_previous_payments", "customer_success_rate",
           "attempt_no", "hour", "dow", "dom", "delay_h"]
CATEGORICAL = ["reason", "method", "action"]
FEATURES = NUMERIC + CATEGORICAL
_CAT_MASK = np.array([False] * len(NUMERIC) + [True] * len(CATEGORICAL))

# Assumed costs of acting (rupee-equivalent friction): a retry costs processing fees and a
# small annoyance cost, a customer request costs more. They only break ties between actions
# with similar success odds; payment amounts dominate the expected value.
COST = {"retry": 3.0, "card_update": 20.0, "reauth": 20.0}

# Candidate (action, delay in hours) pairs the agent can choose from.
CANDIDATES = (
    [("retry", d) for d in sim.RETRY_DELAYS_H]
    + [("card_update", d) for d in sim.CONTACT_DELAYS_H]
    + [("reauth", d) for d in sim.CONTACT_DELAYS_H]
)

LABEL_KIND = {
    "Smart retry": "retry", "Retry later": "retry",
    "Update payment method": "card_update", "Re-authenticate": "reauth",
}


def action_label(kind: str, delay_h: float) -> str:
    if kind == "card_update":
        return "Update payment method"
    if kind == "reauth":
        return "Re-authenticate"
    return "Smart retry" if delay_h <= 3 else "Retry later"


# ---------------------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------------------
def _codes(series: pd.Series, vocab: list[str]) -> np.ndarray:
    lookup = {v: i for i, v in enumerate(vocab)}
    return series.map(lambda v: lookup.get(v, len(vocab))).to_numpy(int)


def encode(frame: pd.DataFrame) -> np.ndarray:
    """Turn an attempt frame (see simulation.attempt_frame) into the model's feature matrix."""
    return np.column_stack([
        np.log10(np.maximum(frame["amount"].to_numpy(float), 1.0)),
        frame["customer_previous_payments"].to_numpy(float),
        frame["customer_success_rate"].to_numpy(float),
        frame["attempt_no"].to_numpy(float),
        frame["hour"].to_numpy(float),
        frame["dow"].to_numpy(float),
        frame["dom"].to_numpy(float),
        frame["delay_h"].to_numpy(float),
        _codes(frame["failure_reason"], sim.REASONS),
        _codes(frame["payment_method"], sim.METHODS),
        _codes(frame["action"], sim.ACTIONS),
    ])


# ---------------------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------------------
def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi if hi == edges[-1] else p < hi)
        if mask.any():
            out.append({"predicted": round(float(p[mask].mean()), 3),
                        "observed": round(float(y[mask].mean()), 3), "n": int(mask.sum())})
    return out


def train_model(extra: pd.DataFrame | None = None) -> None:
    """Train on the simulated logs; `extra` optionally adds attempts logged by the app itself
    (same columns as simulation.attempt_frame plus `success`)."""
    global MODEL, MODEL_INFO
    payments = sim.generate_failures(N_TRAIN_PAYMENTS, seed=TRAIN_SEED, days=TRAIN_DAYS)
    logs = sim.simulate_logged_attempts(payments, seed=TRAIN_SEED + 1)

    cutoff = payments["timestamp"].min().normalize() + pd.Timedelta(days=SPLIT_DAY)
    is_train = (logs["payment_ts"] < cutoff).to_numpy()
    X, y = encode(logs), logs["success"].to_numpy()

    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
        categorical_features=_CAT_MASK, random_state=0,
    )
    X_fit, y_fit = X[is_train], y[is_train]
    n_extra = 0
    if extra is not None and len(extra):
        X_fit = np.vstack([X_fit, encode(extra)])
        y_fit = np.concatenate([y_fit, extra["success"].to_numpy()])
        n_extra = len(extra)
    model.fit(X_fit, y_fit)
    MODEL = model

    p = model.predict_proba(X[~is_train])[:, 1]
    y_test = y[~is_train]
    base = float(y[is_train].mean())
    p_true = logs.loc[~is_train, "p_true"].to_numpy()
    MODEL_INFO = {
        "model": "HistGradientBoostingClassifier",
        "data": "Simulated attempt logs (backend/simulation.py). No real payment data.",
        "split": f"time-based: payments from the first {SPLIT_DAY} days train, the last "
                 f"{TRAIN_DAYS - SPLIT_DAY} days are held out",
        "training_attempts": int(is_train.sum()) + n_extra,
        "logged_attempts_used": n_extra,
        "validation_attempts": int((~is_train).sum()),
        "features": FEATURES,
        "roc_auc": round(float(roc_auc_score(y_test, p)), 3),
        "roc_auc_ceiling": round(float(roc_auc_score(y_test, p_true)), 3),
        "pr_auc": round(float(average_precision_score(y_test, p)), 3),
        "pr_auc_base_rate": round(float(y_test.mean()), 3),
        "brier_score": round(float(brier_score_loss(y_test, p)), 4),
        "brier_score_base_rate": round(float(brier_score_loss(y_test, np.full_like(p, base))), 4),
        "mean_predicted": round(float(p.mean()), 3),
        "mean_observed": round(float(y_test.mean()), 3),
        "calibration": _calibration_table(y_test, p),
        "label_note": "Labels are outcomes of simulated attempts; results show the method on a "
                      "simulated world, not production performance.",
    }


def ensure_model() -> None:
    if MODEL is None:
        train_model()


def predict_attempts(frame: pd.DataFrame) -> np.ndarray:
    ensure_model()
    return MODEL.predict_proba(encode(frame))[:, 1]


# ---------------------------------------------------------------------------------------
# Choosing the next-best action
# ---------------------------------------------------------------------------------------
def score_candidates(payments: pd.DataFrame, attempt_no, base_time) -> tuple[np.ndarray, np.ndarray]:
    """Success probability and expected value for every candidate action, shape (n, n_candidates)."""
    n = len(payments)
    probs = np.empty((n, len(CANDIDATES)))
    for j, (kind, delay) in enumerate(CANDIDATES):
        frame = sim.attempt_frame(payments, kind, delay, attempt_no, base_time)
        probs[:, j] = predict_attempts(frame)
    cost = np.array([COST[k] for k, _ in CANDIDATES])
    ev = probs * payments["amount"].to_numpy(float)[:, None] - cost[None, :]
    return probs, ev


def choose_actions(payments: pd.DataFrame, attempt_no, base_time, threshold: float | None = None) -> pd.DataFrame:
    """Best candidate per payment by expected value. `act` is True when the best option clears the
    confidence threshold and has positive expected value."""
    threshold = DEFAULT_POLICY["recovery_threshold"] if threshold is None else threshold
    probs, ev = score_candidates(payments, attempt_no, base_time)
    best = ev.argmax(axis=1)
    rows = np.arange(len(payments))
    p_best = probs[rows, best]
    return pd.DataFrame({
        "action": [CANDIDATES[j][0] for j in best],
        "delay_h": [float(CANDIDATES[j][1]) for j in best],
        "prob": p_best,
        "ev": ev[rows, best],
        "act": (p_best >= threshold) & (ev[rows, best] > 0),
    }, index=payments.index)


# ---------------------------------------------------------------------------------------
# App integration
# ---------------------------------------------------------------------------------------
def _records_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    return df[["failure_reason", "payment_method", "amount",
               "customer_previous_payments", "customer_success_rate"]].copy()


def recommend(record: dict) -> dict:
    """Recommendation for one payment: action label, delay, probability and an explanation."""
    ensure_model()
    pay = _records_frame([record])
    attempt_no = min(int(record.get("retry_count", 0)) + 1, sim.MAX_ATTEMPTS)
    base = np.array([sim.now_ist()])
    probs, ev = score_candidates(pay, attempt_no, base)
    order = np.argsort(-ev[0])
    best = int(order[0])
    kind, delay = CANDIDATES[best]
    prob = float(probs[0, best])
    amount = float(record["amount"])
    retries = int(record.get("retry_count", 0))

    if amount >= DEFAULT_POLICY["high_value_threshold"]:
        label = "Escalate"
    elif retries >= DEFAULT_POLICY["max_retries"] or prob < DEFAULT_POLICY["recovery_threshold"] \
            or ev[0, best] <= 0:
        label = "No action"
    else:
        label = action_label(kind, delay)

    options = [
        {"name": f"{action_label(CANDIDATES[j][0], CANDIDATES[j][1])}"
                 + (f" in {int(CANDIDATES[j][1])}h" if CANDIDATES[j][1] else " now"),
         "value": round(float(probs[0, j]) * 100, 1),
         "direction": "positive" if j == best else "neutral"}
        for j in order[:3]
    ]
    signals = options + [
        {"name": "Customer success rate", "value": round(float(record["customer_success_rate"]) * 100, 1),
         "direction": "positive"},
        {"name": "Failure reason", "value": record["failure_reason"], "direction": "context"},
        {"name": "Attempt number", "value": attempt_no, "direction": "negative" if attempt_no >= 2 else "neutral"},
    ]
    return {"prob": prob, "action": label, "delay_h": float(delay) if label not in ("Escalate", "No action") else 0.0,
            "kind": kind, "ev": float(ev[0, best]), "signals": signals}


def predict(record: dict) -> tuple[float, str, list[dict]]:
    r = recommend(record)
    return r["prob"], r["action"], r["signals"]


def rescore_payments() -> int:
    """Score every open failed payment with the model and store probability, expected value,
    next-best action and its timing. Safe to run repeatedly. Returns how many were rescored."""
    ensure_model()
    conn = get_conn()
    rows = [dict(r) for r in conn.execute("SELECT * FROM payments WHERE status='failed'")]
    if not rows:
        conn.close()
        return 0
    df = pd.DataFrame(rows)
    pay = _records_frame(rows)
    attempt_no = np.minimum(df["retry_count"].to_numpy(int) + 1, sim.MAX_ATTEMPTS)
    chosen = choose_actions(pay, attempt_no, np.full(len(pay), sim.now_ist()), threshold=0.0)

    threshold = DEFAULT_POLICY["recovery_threshold"]
    updates = []
    for r, (_, c) in zip(rows, chosen.iterrows()):
        amount = float(r["amount"])
        if amount >= DEFAULT_POLICY["high_value_threshold"]:
            label, delay = "Escalate", 0.0
        elif int(r["retry_count"]) >= DEFAULT_POLICY["max_retries"] or c["prob"] < threshold or c["ev"] <= 0:
            label, delay = "No action", 0.0
        else:
            label, delay = action_label(c["action"], c["delay_h"]), float(c["delay_h"])
        updates.append((round(float(c["prob"]), 4), round(amount * float(c["prob"]), 2), label, delay, r["payment_id"]))
    conn.executemany(
        "UPDATE payments SET recovery_probability=?, recoverable_value=?, recommended_action=?, "
        "recommended_delay_h=? WHERE payment_id=?", updates)
    conn.commit()
    conn.close()
    return len(updates)


def retrain_from_logged() -> dict:
    """Retrain with the attempts stored in the `attempts` table added to the simulated logs, then
    rescore open payments. Returns metrics before and after on the same held-out simulated test set."""
    ensure_model()
    before = {k: MODEL_INFO[k] for k in ("roc_auc", "pr_auc", "brier_score")}
    conn = get_conn()
    rows = pd.read_sql_query("SELECT * FROM attempts", conn)
    conn.close()
    if rows.empty:
        return {"retrained": False, "reason": "No logged attempts yet. Run a recovery cycle first.",
                "logged_attempts": 0, "before": before}
    extra = rows.rename(columns={"action": "action"})[[
        "failure_reason", "payment_method", "amount", "customer_previous_payments",
        "customer_success_rate", "attempt_no", "action", "delay_h", "hour", "dow", "dom", "success"]]
    train_model(extra)
    rescored = rescore_payments()
    after = {k: MODEL_INFO[k] for k in ("roc_auc", "pr_auc", "brier_score")}
    return {"retrained": True, "logged_attempts": int(len(extra)), "rescored": rescored,
            "before": before, "after": after}
