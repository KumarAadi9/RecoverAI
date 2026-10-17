import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import simulation as sim  # noqa: E402


def one(reason="Bank declined", action="retry", delay=0, attempt=1, hour=12, dom=15,
        amount=5000, rate=0.8):
    frame = pd.DataFrame([{
        "failure_reason": reason, "payment_method": "UPI", "amount": amount,
        "customer_previous_payments": 10, "customer_success_rate": rate, "attempt_no": attempt,
        "action": action, "delay_h": delay, "hour": hour, "dow": 2, "dom": dom,
    }])
    return float(sim.true_success_prob(frame)[0])


def test_world_rules_behave_as_documented():
    # Retrying an expired card is hopeless, asking the customer to update it works.
    assert one("Card expired", "retry") < 0.05
    assert one("Card expired", "card_update") > 0.4
    # Authentication failures need re-authentication, not retries.
    assert one("Authentication failed", "reauth") > 5 * one("Authentication failed", "retry")
    # Insufficient funds clears more often just after salary day than late in the month.
    assert one("Insufficient funds", dom=3) > 2 * one("Insufficient funds", dom=25)
    # Each repeated retry is less effective and waiting longer loses customers.
    assert one(attempt=3) < one(attempt=1)
    assert one(delay=168) < one(delay=0)
    # Bank-side retries are worse in the small hours.
    assert one("Bank declined", hour=3) < one("Bank declined", hour=14)


def test_probabilities_stay_in_bounds():
    pay = sim.generate_failures(2000, seed=3)
    rng = np.random.default_rng(0)
    frame = sim.attempt_frame(pay, rng.choice(sim.ACTIONS, len(pay)), rng.choice(sim.RETRY_DELAYS_H, len(pay)),
                              rng.integers(1, 4, len(pay)), pay["timestamp"])
    p = sim.true_success_prob(frame)
    assert p.min() >= 0.01 and p.max() <= 0.97


def test_generation_is_deterministic_and_realistic():
    a, b = sim.generate_failures(500, seed=7), sim.generate_failures(500, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["failure_reason"]) <= set(sim.REASONS)
    assert not ((a["payment_method"] != "Card") & (a["failure_reason"] == "Card expired")).any()
    assert (a["amount"] >= 200).all()


def test_logged_attempts_respect_the_attempt_limit():
    pay = sim.generate_failures(1500, seed=4)
    logs = sim.simulate_logged_attempts(pay, seed=5)
    per_payment = logs.groupby("payment_id").size()
    assert per_payment.max() <= sim.MAX_ATTEMPTS
    # a payment never has an attempt after a successful one
    won_at = logs[logs["success"] == 1].groupby("payment_id")["attempt_no"].min()
    last = logs.groupby("payment_id")["attempt_no"].max()
    assert (last.loc[won_at.index] == won_at).all()
