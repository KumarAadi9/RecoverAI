import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import backtest, ml_engine  # noqa: E402
from backend import simulation as sim  # noqa: E402


def test_model_beats_the_base_rate_and_is_calibrated():
    ml_engine.ensure_model()
    i = ml_engine.MODEL_INFO
    assert i["pr_auc"] > i["pr_auc_base_rate"] + 0.1          # learned real signal
    assert i["brier_score"] < i["brier_score_base_rate"]
    assert abs(i["mean_predicted"] - i["mean_observed"]) < 0.03
    assert i["roc_auc"] <= i["roc_auc_ceiling"] + 0.01          # cannot beat the true probabilities


def test_model_learns_the_expired_card_pattern():
    ml_engine.ensure_model()
    pay = pd.DataFrame([{"failure_reason": "Card expired", "payment_method": "Card", "amount": 5000,
                         "customer_previous_payments": 10, "customer_success_rate": 0.8}])
    probs, _ = ml_engine.score_candidates(pay, 1, np.array([pd.Timestamp("2026-06-10 11:00")]))
    kinds = [k for k, _ in ml_engine.CANDIDATES]
    best_retry = max(probs[0, j] for j, k in enumerate(kinds) if k == "retry")
    best_update = max(probs[0, j] for j, k in enumerate(kinds) if k == "card_update")
    assert best_update > 5 * best_retry


def test_strategies_follow_the_rules_of_the_experiment():
    ml_engine.ensure_model()
    pay = sim.generate_failures(1500, seed=21, start="2026-05-15", days=30)
    pay = pay[pay["amount"] < 100000].reset_index(drop=True)
    U = np.random.default_rng(1).random((len(pay), sim.MAX_ATTEMPTS))
    for strat in (backtest.naive_strategy, backtest.rules_strategy, backtest.make_recoverai_strategy(0.0)):
        r = backtest.run_strategy(pay, strat, U)
        assert r["attempts"].max() <= sim.MAX_ATTEMPTS
        assert (r["failed_retries"] + r["contacts"] <= r["attempts"]).all()
        assert (r["recovered"] <= (r["attempts"] > 0)).all()      # nothing recovered without an attempt
    again = backtest.run_strategy(pay, backtest.make_recoverai_strategy(0.0), U)
    first = backtest.run_strategy(pay, backtest.make_recoverai_strategy(0.0), U)
    assert (again["recovered"] == first["recovered"]).all()        # same draws, same result


def test_stricter_gate_means_fewer_attempts():
    ml_engine.ensure_model()
    pay = sim.generate_failures(1500, seed=22, start="2026-05-15", days=30)
    pay = pay[pay["amount"] < 100000].reset_index(drop=True)
    U = np.random.default_rng(2).random((len(pay), sim.MAX_ATTEMPTS))
    loose = backtest.run_strategy(pay, backtest.make_recoverai_strategy(0.0), U)["attempts"].sum()
    strict = backtest.run_strategy(pay, backtest.make_recoverai_strategy(0.6), U)["attempts"].sum()
    assert strict < loose
