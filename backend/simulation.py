"""A simulated payments world with hidden recovery rules.

There is no real payment data, so this module defines "the world": how likely a recovery
attempt is to succeed given the failure, the customer, the action and the time. The rules
are explicit assumptions (see TRUE RULES below) written down *before* any model sees the data.
The model never reads these rules; it only sees logged attempts and their outcomes, and has
to learn the patterns. Everything produced here is simulated, not real payment data.

All timestamps are naive India Standard Time (IST) so that hour-of-day and day-of-month
effects line up with Indian payment behaviour (salary credits, banking hours).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REASONS = [
    "Network timeout", "Gateway timeout", "Bank declined",
    "Insufficient funds", "Authentication failed", "Card expired",
]
METHODS = ["UPI", "Card", "Netbanking", "Wallet"]

# Action kinds. 'retry' re-attempts the charge, the other two need the customer to act.
ACTIONS = ["retry", "card_update", "reauth"]
RETRY_DELAYS_H = [0, 3, 12, 24, 48, 96, 168]   # when a retry may be scheduled
CONTACT_DELAYS_H = [0, 24]                      # when a customer request may be sent

MAX_ATTEMPTS = 3

# ----------------------------------------------------------------------------------------
# TRUE RULES (hidden from the model)
# ----------------------------------------------------------------------------------------
# Base success probability by failure reason and action, before modifiers.
_BASE = {
    "retry": {
        "Network timeout": 0.72, "Gateway timeout": 0.66, "Bank declined": 0.38,
        "Insufficient funds": 0.22, "Authentication failed": 0.08, "Card expired": 0.01,
    },
    "card_update": {
        "Network timeout": 0.20, "Gateway timeout": 0.20, "Bank declined": 0.30,
        "Insufficient funds": 0.15, "Authentication failed": 0.35, "Card expired": 0.68,
    },
    "reauth": {
        "Network timeout": 0.15, "Gateway timeout": 0.15, "Bank declined": 0.15,
        "Insufficient funds": 0.10, "Authentication failed": 0.62, "Card expired": 0.05,
    },
}
_BANK_SIDE = {"Bank declined", "Insufficient funds", "Authentication failed"}


def true_success_prob(df: pd.DataFrame) -> np.ndarray:
    """Probability that an attempt succeeds. `df` needs the columns:
    failure_reason, amount, customer_success_rate, attempt_no, action, delay_h, hour, dom.
    (`hour` and `dom` describe the moment the attempt happens.)"""
    base = np.empty(len(df))
    for action in ACTIONS:
        mask = (df["action"] == action).to_numpy()
        if mask.any():
            base[mask] = df.loc[mask, "failure_reason"].map(_BASE[action]).to_numpy()

    is_retry = (df["action"] == "retry").to_numpy()
    attempt = df["attempt_no"].to_numpy()
    p = base
    # Reliable customers recover more often.
    p = p * (0.7 + 0.6 * df["customer_success_rate"].to_numpy())
    # Repeating the same fix is less effective each time.
    p = p * np.where(is_retry, 0.70, 0.85) ** (attempt - 1)
    # Bigger payments are slightly harder to recover.
    p = p * (1 - 0.06 * np.log10(np.maximum(df["amount"].to_numpy(), 100) / 100))
    # Bank-side retries: banks are flaky overnight and responsive in business hours.
    hour = df["hour"].to_numpy()
    bank = df["failure_reason"].isin(_BANK_SIDE).to_numpy() & is_retry
    p = p * np.where(bank & (hour < 6), 0.6, 1.0)
    p = p * np.where(bank & (hour >= 10) & (hour < 20), 1.1, 1.0)
    # Insufficient funds: much more likely to clear just after salary credits.
    dom = df["dom"].to_numpy()
    insuff = (df["failure_reason"] == "Insufficient funds").to_numpy() & is_retry
    p = p * np.where(insuff & (dom <= 5), 2.0, 1.0)
    p = p * np.where(insuff & (dom >= 20), 0.7, 1.0)
    # The longer a customer waits, the more likely they have abandoned the purchase.
    p = p * np.exp(-df["delay_h"].to_numpy() / (30 * 24))
    return np.clip(p, 0.01, 0.97)


# ----------------------------------------------------------------------------------------
# Generating failed payments
# ----------------------------------------------------------------------------------------
def generate_failures(n: int, seed: int, start: str = "2026-01-01", days: int = 120) -> pd.DataFrame:
    """Create `n` first-time payment failures spread over `days` days."""
    rng = np.random.default_rng(seed)
    method = rng.choice(METHODS, n, p=[0.45, 0.30, 0.15, 0.10])
    reason = rng.choice(REASONS, n, p=[0.18, 0.14, 0.24, 0.22, 0.12, 0.10])
    reason = np.where((reason == "Card expired") & (method != "Card"), "Bank declined", reason)

    day = rng.integers(0, days, n)
    hour = rng.choice(24, n, p=_hour_weights())
    minute = rng.integers(0, 60, n)
    ts = pd.Timestamp(start) + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour * 60 + minute, unit="min")

    amount = np.clip(np.round(rng.lognormal(np.log(6000), 1.2, n), -2), 200, 500000)
    return pd.DataFrame({
        "payment_id": [f"SIM{i:06d}" for i in range(n)],
        "customer_id": [f"CUST{i:06d}" for i in rng.integers(0, max(n // 3, 1), n)],
        "amount": amount,
        "payment_method": method,
        "failure_reason": reason,
        "customer_previous_payments": rng.poisson(15, n) + 1,
        "customer_success_rate": np.round(rng.beta(8, 2, n), 4),
        "timestamp": ts,
    }).sort_values("timestamp").reset_index(drop=True)


def _hour_weights() -> np.ndarray:
    w = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 5, 3, 2], float)
    return w / w.sum()


# ----------------------------------------------------------------------------------------
# Attempt features (shared by the world, the model and the strategies)
# ----------------------------------------------------------------------------------------
def attempt_frame(payments: pd.DataFrame, action, delay_h, attempt_no, base_time) -> pd.DataFrame:
    """Describe an attempt: which payment, which action, scheduled `delay_h` hours after
    `base_time` (an IST datetime Series/array). `action`, `delay_h`, `attempt_no` may be scalars
    or arrays aligned with `payments`."""
    n = len(payments)
    when = pd.to_datetime(pd.Series(np.asarray(base_time), index=payments.index)) \
        + pd.to_timedelta(np.broadcast_to(np.asarray(delay_h, float), n), unit="h")
    return pd.DataFrame({
        "failure_reason": payments["failure_reason"].to_numpy(),
        "payment_method": payments["payment_method"].to_numpy(),
        "amount": payments["amount"].to_numpy(float),
        "customer_previous_payments": payments["customer_previous_payments"].to_numpy(float),
        "customer_success_rate": payments["customer_success_rate"].to_numpy(float),
        "attempt_no": np.broadcast_to(np.asarray(attempt_no, int), n),
        "action": np.broadcast_to(np.asarray(action, dtype=object), n),
        "delay_h": np.broadcast_to(np.asarray(delay_h, float), n),
        "hour": when.dt.hour.to_numpy(),
        "dow": when.dt.dayofweek.to_numpy(),
        "dom": when.dt.day.to_numpy(),
    }, index=payments.index)


# ----------------------------------------------------------------------------------------
# Historical logs: attempts made by an exploring (randomised) policy
# ----------------------------------------------------------------------------------------
def simulate_logged_attempts(payments: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Run every payment through up to MAX_ATTEMPTS attempts chosen at random (exploration), so
    the logs cover every action and timing. Returns one row per attempt with the outcome."""
    rng = np.random.default_rng(seed)
    n = len(payments)
    active = np.ones(n, bool)
    elapsed = np.zeros(n)
    base = payments["timestamp"].to_numpy()
    rows = []
    for k in range(1, MAX_ATTEMPTS + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        sub = payments.iloc[idx]
        action = rng.choice(ACTIONS, idx.size, p=[0.6, 0.2, 0.2])
        delay = np.where(
            action == "retry",
            rng.choice(RETRY_DELAYS_H, idx.size),
            rng.choice(CONTACT_DELAYS_H, idx.size),
        ).astype(float)
        when_base = pd.to_datetime(base[idx]) + pd.to_timedelta(elapsed[idx], unit="h")
        frame = attempt_frame(sub, action, delay, k, when_base)
        p_true = true_success_prob(frame)
        success = rng.random(idx.size) < p_true
        frame = frame.assign(success=success.astype(int), p_true=p_true,
                             payment_ts=sub["timestamp"].to_numpy(), payment_id=sub["payment_id"].to_numpy())
        rows.append(frame)
        elapsed[idx] += delay
        active[idx[success]] = False
    return pd.concat(rows, ignore_index=True)


def now_ist() -> pd.Timestamp:
    """Current time as a naive IST timestamp."""
    return pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
