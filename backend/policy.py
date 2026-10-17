from __future__ import annotations

DEFAULT_POLICY = {
    'auto_recovery': True,
    'high_value_approval': True,
    'high_value_threshold': 100000,
    # Confidence floor for automatic action. 0.0 means "rely on the expected-value check".
    # Chosen on a validation set by backend/backtest.py (select_threshold); a strict 0.60 floor
    # recovers far less revenue because a single retry of a bank decline succeeds well under 60%
    # of the time even though it is still worth attempting. Raise it to act more conservatively.
    'recovery_threshold': 0.0,
    'max_retries': 3,
}

# The policy the running app uses. Only the automatic-recovery switch can be changed at runtime
# (a kill switch); the other guardrails stay fixed so they cannot be switched off from the UI.
ACTIVE_POLICY = dict(DEFAULT_POLICY)
EDITABLE = ("auto_recovery",)


def update_policy(changes: dict) -> dict:
    for key, value in changes.items():
        if key not in EDITABLE:
            raise ValueError(f"{key} cannot be changed at runtime")
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        ACTIVE_POLICY[key] = value
    return dict(ACTIVE_POLICY)


def decide(payment: dict, probability: float, policy: dict | None = None) -> dict:
    p = {**DEFAULT_POLICY, **(policy or {})}
    amount = float(payment['amount'])
    retries = int(payment.get('retry_count', 0))

    if amount >= p['high_value_threshold'] and p['high_value_approval']:
        return {'allowed': False, 'decision': 'manual_review', 'reason': 'High-value transaction requires merchant approval.'}

    if not p['auto_recovery']:
        return {'allowed': False, 'decision': 'manual_review', 'reason': 'Automatic recovery is disabled.'}

    if probability < p['recovery_threshold']:
        return {'allowed': False, 'decision': 'no_action', 'reason': 'Recovery probability is below the automation threshold.'}

    if retries >= p['max_retries']:
        return {'allowed': False, 'decision': 'no_action', 'reason': 'Maximum automatic retries reached.'}

    return {'allowed': True, 'decision': 'execute', 'reason': 'Payment meets recovery confidence and merchant policy.'}
