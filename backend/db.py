from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "recoverai.db"
DATA_PATH = ROOT / "payments_50000.json"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS payments (
            payment_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            amount REAL NOT NULL,
            payment_method TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            failure_reason TEXT NOT NULL,
            customer_previous_payments INTEGER NOT NULL,
            customer_success_rate REAL NOT NULL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            recovery_probability REAL NOT NULL,
            recoverable_value REAL NOT NULL,
            recovered_value REAL NOT NULL DEFAULT 0,
            recommended_action TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL,
            payment_id TEXT,
            title TEXT NOT NULL,
            detail TEXT NOT NULL,
            amount REAL NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )

    count = conn.execute(
        "SELECT COUNT(*) AS n FROM payments"
    ).fetchone()["n"]

    if count == 0:
        seed_from_dataset(conn)

    conn.commit()
    conn.close()


def seed_from_dataset(conn: sqlite3.Connection) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATA_PATH}"
        )

    raw = json.loads(
        DATA_PATH.read_text(encoding="utf-8")
    )

    # Keep the local demo responsive while still
    # giving the app thousands of records.
    rows = raw[:5000]

    for p in rows:
        prob = float(
            p.get("recovery_probability", 0.5)
        )

        amount = float(p["amount"])

        # Seed a small amount of historical recovered
        # traffic so the dashboard has a meaningful
        # baseline on first launch.
        digest = hashlib.sha256(
            p["payment_id"].encode()
        ).hexdigest()[:8]

        historical_recovered = (
            int(digest, 16) % 100
        ) < int(prob * 12)

        seeded_status = (
            "recovered"
            if historical_recovered
            else "failed"
        )

        # IMPORTANT:
        # Historical recovered payments must also
        # store the amount actually recovered.
        recovered_value = (
            amount
            if historical_recovered
            else 0
        )

        # Select the recommended recovery action.
        if prob >= 0.80:
            action = "Smart retry"

        elif p["failure_reason"] == "Card expired":
            action = "Update payment method"

        elif p["failure_reason"] == "Authentication failed":
            action = "Re-authenticate"

        elif prob >= 0.60:
            action = "Retry later"

        else:
            action = "No action"

        conn.execute(
            """
            INSERT INTO payments(
                payment_id,
                customer_id,
                amount,
                payment_method,
                timestamp,
                status,
                failure_reason,
                customer_previous_payments,
                customer_success_rate,
                retry_count,
                recovery_probability,
                recoverable_value,
                recovered_value,
                recommended_action
            )
            VALUES(
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            (
                p["payment_id"],
                p["customer_id"],
                amount,
                p["payment_method"],
                p["timestamp"],
                seeded_status,
                p["failure_reason"],
                int(p["customer_previous_payments"]),
                float(p["customer_success_rate"]),
                int(p.get("retry_count", 0))
                + (1 if historical_recovered else 0),
                prob,
                float(
                    p.get(
                        "recoverable_value",
                        amount * prob
                    )
                ),
                recovered_value,
                action,
            ),
        )


def insert_event(
    event_type: str,
    payment_id: str | None,
    title: str,
    detail: str,
    amount: float = 0,
    metadata: dict[str, Any] | None = None,
) -> None:

    conn = get_conn()

    conn.execute(
        """
        INSERT INTO events(
            event_type,
            payment_id,
            title,
            detail,
            amount,
            metadata_json
        )
        VALUES(?, ?, ?, ?, ?, ?)
        """,
        (
            event_type,
            payment_id,
            title,
            detail,
            amount,
            json.dumps(metadata or {}),
        ),
    )

    conn.commit()
    conn.close()


def clear_demo_data() -> None:
    conn = get_conn()

    conn.execute("DELETE FROM events")
    conn.execute("DELETE FROM payments")

    seed_from_dataset(conn)

    conn.commit()
    conn.close()