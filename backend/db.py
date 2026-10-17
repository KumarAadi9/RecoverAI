from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "recoverai.db"


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
            recommended_delay_h REAL NOT NULL DEFAULT 0,
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

    # Databases created by older versions lack the recommended_delay_h column.
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(payments)")]
    if "recommended_delay_h" not in cols:
        conn.execute("ALTER TABLE payments ADD COLUMN recommended_delay_h REAL NOT NULL DEFAULT 0")

    count = conn.execute(
        "SELECT COUNT(*) AS n FROM payments"
    ).fetchone()["n"]

    if count == 0:
        seed_from_dataset(conn)

    conn.commit()
    conn.close()


SEED_PAYMENTS = 5000


def seed_from_dataset(conn: sqlite3.Connection) -> None:
    """Fill the demo database with simulated failed payments from the same world the model and
    the backtest use (see simulation.py). Scores and actions are filled in by
    ml_engine.rescore_payments() right after."""
    from . import simulation as sim

    start = sim.now_ist().normalize() - __import__("pandas").Timedelta(days=30)
    df = sim.generate_failures(SEED_PAYMENTS, seed=2026, start=str(start.date()), days=30)

    for p in df.itertuples():
        # A small share of payments start out already recovered so the dashboard has a baseline.
        digest = hashlib.sha256(p.payment_id.encode()).hexdigest()[:8]
        recovered = (int(digest, 16) % 100) < 8
        amount = float(p.amount)
        conn.execute(
            """
            INSERT INTO payments(
                payment_id, customer_id, amount, payment_method, timestamp, status,
                failure_reason, customer_previous_payments, customer_success_rate,
                retry_count, recovery_probability, recoverable_value, recovered_value,
                recommended_action, recommended_delay_h
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p.payment_id, p.customer_id, amount, p.payment_method,
                p.timestamp.tz_localize("Asia/Kolkata").isoformat(),
                "recovered" if recovered else "failed",
                p.failure_reason, int(p.customer_previous_payments),
                float(p.customer_success_rate),
                1 if recovered else 0,
                0.5, 0.0 if recovered else amount * 0.5, amount if recovered else 0,
                "No action", 0.0,
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