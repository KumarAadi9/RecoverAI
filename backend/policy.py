from __future__ import annotations

DEFAULT_POLICY = {
    'auto_recovery': True,
    'high_value_approval': True,
    'high_value_threshold': 100000,
    'recovery_threshold': 0.60,
    'max_retries': 3,
}


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
