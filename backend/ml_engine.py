from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'payments_50000.json'

MODEL: Pipeline | None = None
MODEL_INFO: dict = {}

NUMERIC = [
    'amount', 'customer_previous_payments', 'customer_success_rate',
    'retry_count', 'hour', 'day_of_week'
]
CATEGORICAL = ['payment_method', 'failure_reason']


def deterministic_uniform(key: str) -> float:
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return int(digest, 16) / float(16**12 - 1)


def make_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    ts = pd.to_datetime(df['timestamp'], utc=True)
    df['hour'] = ts.dt.hour
    df['day_of_week'] = ts.dt.dayofweek
    return df


def train_model() -> None:
    global MODEL, MODEL_INFO
    raw = json.loads(DATA_PATH.read_text(encoding='utf-8'))
    df = make_frame(raw)

    # Synthetic outcome for the prototype. The supplied dataset has a
    # recovery_probability but not a real recovered/not-recovered label.
    df['recovered_label'] = [
        1 if deterministic_uniform(pid) < float(prob) else 0
        for pid, prob in zip(df['payment_id'], df['recovery_probability'])
    ]

    X = df[NUMERIC + CATEGORICAL].copy()
    y = df['recovered_label']

    pre = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])
    MODEL = Pipeline([
        ('preprocess', pre),
        ('model', LogisticRegression(max_iter=600, class_weight='balanced')),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    MODEL.fit(X_train, y_train)
    probs = MODEL.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.50).astype(int)

    MODEL_INFO = {
        'model': 'Logistic Regression',
        'training_records': int(len(X_train)),
        'validation_records': int(len(X_test)),
        'features': NUMERIC + CATEGORICAL,
        'roc_auc': round(float(roc_auc_score(y_test, probs)), 3),
        'accuracy': round(float(accuracy_score(y_test, preds)), 3),
        'label_note': 'Prototype labels are synthetically sampled from supplied recovery_probability values.',
    }


def ensure_model() -> None:
    if MODEL is None:
        train_model()


def predict(record: dict) -> tuple[float, str, list[dict]]:
    ensure_model()
    frame = make_frame([record])
    X = frame[NUMERIC + CATEGORICAL]
    prob = float(MODEL.predict_proba(X)[0, 1])

    # Keep the recommendation deterministic and transparent.
    reason = record['failure_reason']
    amount = float(record['amount'])
    retries = int(record.get('retry_count', 0))
    if amount >= 100000:
        action = 'Escalate'
    elif reason == 'Card expired':
        action = 'Update payment method'
    elif reason == 'Authentication failed':
        action = 'Re-authenticate'
    elif prob >= 0.80 and retries < 3:
        action = 'Smart retry'
    elif prob >= 0.60 and retries < 3:
        action = 'Retry later'
    else:
        action = 'No action'

    success = float(record['customer_success_rate'])
    history = int(record['customer_previous_payments'])
    signals = [
        {'name': 'Customer success rate', 'value': round(success * 100, 1), 'direction': 'positive'},
        {'name': 'Prior payments', 'value': history, 'direction': 'positive'},
        {'name': 'Failure reason', 'value': reason, 'direction': 'context'},
        {'name': 'Retry count', 'value': retries, 'direction': 'negative' if retries >= 2 else 'neutral'},
        {'name': 'Transaction value', 'value': amount, 'direction': 'negative' if amount >= 100000 else 'neutral'},
    ]
    return prob, action, signals
