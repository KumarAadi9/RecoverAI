from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import get_conn, init_db, insert_event
from . import ml_engine
from .policy import decide, DEFAULT_POLICY


# =========================================================
# APP CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(
    title="RecoverAI API",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODELS
# =========================================================

class SimulatePayment(BaseModel):
    amount: float | None = Field(
        default=None,
        gt=100,
    )
    payment_method: str | None = None
    failure_reason: str | None = None


class ActionRequest(BaseModel):
    action: str


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup() -> None:
    init_db()
    ml_engine.ensure_model()


# =========================================================
# PAYMENT HELPERS
# =========================================================

def normalize_payment(row) -> dict:
    """
    Convert a SQLite payment row into the structure
    expected by the frontend.
    """

    d = dict(row)

    return {
        "id": d["payment_id"],
        "merchant": "Acme Commerce",
        "customer": d["customer_id"],
        "amount": d["amount"],
        "reason": d["failure_reason"],
        "method": d["payment_method"],
        "timestamp": d["timestamp"],
        "previous": d["customer_previous_payments"],
        "successes": round(
            d["customer_previous_payments"]
            * d["customer_success_rate"]
        ),
        "successRate": d["customer_success_rate"],
        "prob": round(
            d["recovery_probability"] * 100
        ),
        "recommended": d["recommended_action"],
        "status": (
            "Recovered"
            if d["status"] == "recovered"
            else "Action needed"
        ),
        "retries": d["retry_count"],
        "risk": round(
            d["recoverable_value"]
        ),
        "device": "Web",
    }


def payment_record(payment_id: str) -> dict:
    """
    Fetch a payment directly from SQLite.
    """

    conn = get_conn()

    row = conn.execute(
        """
        SELECT *
        FROM payments
        WHERE payment_id=?
        """,
        (payment_id,),
    ).fetchone()

    conn.close()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Payment not found",
        )

    return dict(row)


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "RecoverAI",
    }


# =========================================================
# MODEL INFO
# =========================================================

@app.get("/api/model/info")
def model_info():
    ml_engine.ensure_model()

    return ml_engine.MODEL_INFO


# =========================================================
# PAYMENTS
# =========================================================

@app.get("/api/payments")
def payments(
    limit: int = Query(
        1200,
        ge=1,
        le=5000,
    )
):
    conn = get_conn()

    rows = conn.execute(
        """
        SELECT *
        FROM payments
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    conn.close()

    return {
        "payments": [
            normalize_payment(row)
            for row in rows
        ]
    }


@app.get("/api/payments/{payment_id}")
def payment(payment_id: str):
    return normalize_payment(
        payment_record(payment_id)
    )


# =========================================================
# EVENTS
# =========================================================

@app.get("/api/events")
def events(
    limit: int = Query(
        25,
        ge=1,
        le=100,
    )
):
    conn = get_conn()

    rows = conn.execute(
        """
        SELECT *
        FROM events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    conn.close()

    output = []

    for row in rows:

        output.append(
            {
                "time": datetime.fromisoformat(
                    row["created_at"].replace(
                        "Z",
                        "+00:00",
                    )
                )
                .astimezone()
                .strftime("%H:%M:%S"),

                "type": row["event_type"],

                "title": row["title"],

                "detail": row["detail"],

                "amount": row["amount"],
            }
        )

    return {
        "events": output
    }


# =========================================================
# METRICS
# =========================================================

@app.get("/api/metrics")
def metrics():

    conn = get_conn()

    row = conn.execute(
        """
        SELECT

            COALESCE(
                SUM(amount),
                0
            ) AS gross,

            COALESCE(
                SUM(recoverable_value),
                0
            ) AS recoverable,

            COALESCE(
                SUM(recovered_value),
                0
            ) AS recovered,

            SUM(
                CASE
                    WHEN status='failed'
                    THEN 1
                    ELSE 0
                END
            ) AS failed,

            SUM(
                CASE
                    WHEN status='recovered'
                    THEN 1
                    ELSE 0
                END
            ) AS successful,

            SUM(
                CASE
                    WHEN recovery_probability >= 0.80
                    AND status='failed'
                    THEN 1
                    ELSE 0
                END
            ) AS high_conf

        FROM payments
        """
    ).fetchone()

    conn.close()

    gross = float(
        row["gross"] or 0
    )

    recoverable_total = float(
        row["recoverable"] or 0
    )

    recovered = float(
        row["recovered"] or 0
    )

    # Demo assumption:
    # 16% of total payment volume is considered
    # revenue at risk.

    at_risk = gross * 0.16

    recoverable = (
        recoverable_total * 0.16
    )

    recovery_rate = (
        round(
            recovered
            / at_risk
            * 100
        )
        if at_risk
        else 0
    )

    return {

        "atRisk": round(
            at_risk
        ),

        "recoverable": round(
            recoverable
        ),

        "recovered": round(
            recovered
        ),

        "openFailures": int(
            row["failed"] or 0
        ),

        "successfulRecoveries": int(
            row["successful"] or 0
        ),

        "highConfidence": int(
            row["high_conf"] or 0
        ),

        "rate": recovery_rate,
    }


# =========================================================
# RECOVERY OUTCOME
# =========================================================

def outcome_probability(
    action: str,
    model_prob: float,
) -> float:
    """
    Calculate the simulated success probability
    based on the selected recovery action.
    """

    offsets = {

        "Smart retry": 0.03,

        "Retry later": 0.00,

        "Update payment method": 0.04,

        "Re-authenticate": -0.02,
    }

    return max(
        0.05,
        min(
            0.98,
            model_prob
            + offsets.get(
                action,
                0.0,
            ),
        ),
    )


# =========================================================
# SIMULATE PAYMENT FAILURE
# =========================================================

@app.post("/api/payments/simulate")
def simulate_payment(
    payload: SimulatePayment,
):

    reasons = [

        "Gateway timeout",

        "Bank declined",

        "Network timeout",

        "Card expired",

        "Authentication failed",

        "Insufficient funds",
    ]

    methods = [

        "UPI",

        "Card",

        "Netbanking",

        "Wallet",
    ]

    # Keep demo transactions below the
    # high-value approval threshold.

    amount = (

        payload.amount

        if payload.amount is not None

        else round(
            random.uniform(
                5000,
                90000,
            )
            / 100
        )
        * 100
    )

    reason = (
        payload.failure_reason
        or random.choice(reasons)
    )

    method = (
        payload.payment_method
        or random.choice(methods)
    )

    now = datetime.now(
        timezone.utc
    ).isoformat()

    payment_id = (
        f"RP{random.randint(300000, 399999)}"
    )

    customer_id = (
        f"CUST{random.randint(100000, 109999)}"
    )

    previous = random.randint(
        3,
        35,
    )

    success_rate = round(
        random.uniform(
            0.55,
            0.98,
        ),
        4,
    )

    record = {

        "amount": amount,

        "payment_method": method,

        "timestamp": now,

        "failure_reason": reason,

        "customer_previous_payments": previous,

        "customer_success_rate": success_rate,

        "retry_count": 0,
    }

    # Run ML prediction.

    prob, action, signals = (
        ml_engine.predict(
            record
        )
    )

    conn = get_conn()

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

            payment_id,

            customer_id,

            amount,

            method,

            now,

            "failed",

            reason,

            previous,

            success_rate,

            0,

            prob,

            amount * prob,

            0,

            action,
        ),
    )

    conn.commit()

    conn.close()

    # Log payment failure.

    insert_event(

        "failure",

        payment_id,

        "Payment failed",

        f"{payment_id} · {reason}",

        amount,

        {
            "probability": prob,

            "signals": signals,
        },
    )

    # Log AI analysis.

    insert_event(

        "analysis",

        payment_id,

        "AI analysis completed",

        (
            f"{payment_id} · "
            f"{prob * 100:.0f}% "
            "recovery probability"
        ),

        amount,

        {
            "action": action
        },
    )

    return normalize_payment(
        payment_record(
            payment_id
        )
    )


# =========================================================
# EXECUTE RECOVERY ACTION
# =========================================================

@app.post(
    "/api/payments/{payment_id}/action"
)
def payment_action(
    payment_id: str,
    payload: ActionRequest,
):

    # -----------------------------------------------------
    # GET PAYMENT
    # -----------------------------------------------------

    row = payment_record(
        payment_id
    )

    action = payload.action

    prob = float(
        row["recovery_probability"]
    )

    # Calculate policy for auditability.

    policy = decide(
        row,
        prob,
        DEFAULT_POLICY,
    )

    # -----------------------------------------------------
    # EXPLICIT ESCALATION
    # -----------------------------------------------------

    if action == "Escalate":

        insert_event(

            "escalation",

            payment_id,

            "Merchant escalation",

            (
                f"{payment_id} · "
                "manual review required"
            ),

            row["amount"],

            {
                "reason": policy["reason"]
            },
        )

        return {

            "ok": True,

            "result": "escalated",

            "payment": normalize_payment(
                row
            ),

            "policy": policy,
        }

    # -----------------------------------------------------
    # EXPLICIT NO ACTION
    # -----------------------------------------------------

    if action == "No action":

        insert_event(

            "no_action",

            payment_id,

            "No recovery action",

            (
                f"{payment_id} · "
                f"{policy['reason']}"
            ),

            row["amount"],
        )

        return {

            "ok": True,

            "result": "no_action",

            "payment": normalize_payment(
                row
            ),

            "policy": policy,
        }

    # -----------------------------------------------------
    # PREVENT DUPLICATE RECOVERY
    # -----------------------------------------------------

    if row["status"] == "recovered":

        updated_payment = payment_record(
            payment_id
        )

        return {

            "ok": True,

            "result": "already_recovered",

            "payment": normalize_payment(
                updated_payment
            ),

            "policy": policy,

            "recovered_value": row[
                "recovered_value"
            ],
        }

    # -----------------------------------------------------
    # CALCULATE SUCCESS PROBABILITY
    # -----------------------------------------------------

    chance = outcome_probability(

        action,

        prob,
    )

    # -----------------------------------------------------
    # DEMO EXECUTION LOGIC
    # -----------------------------------------------------
    #
    # When the merchant explicitly clicks
    # "Execute action", this demo records the
    # transaction as successfully recovered.
    #
    # This makes the revenue recovery flow
    # deterministic for the presentation/demo.
    #
    # The model probability and calculated
    # success probability remain visible for
    # analytics and explainability.
    # -----------------------------------------------------

    success = True

    conn = get_conn()

    # Increment retry/action counter.

    conn.execute(

        """
        UPDATE payments

        SET retry_count =
            retry_count + 1

        WHERE payment_id=?
        """,

        (payment_id,),
    )

    # -----------------------------------------------------
    # RECORD RECOVERED VALUE
    # -----------------------------------------------------
    #
    # THIS IS THE KEY DATABASE UPDATE.
    #
    # The dashboard's recovered metric is calculated
    # from recovered_value.
    # -----------------------------------------------------

    if success:

        conn.execute(
        """
        UPDATE payments

        SET

            status='recovered',

            recovered_value=amount,

            recoverable_value=0

        WHERE payment_id=?
        """,

        (payment_id,),
    )

    conn.commit()

    conn.close()

    # -----------------------------------------------------
    # LOG RECOVERY ACTION
    # -----------------------------------------------------

    insert_event(

        "recovery",

        payment_id,

        "Recovery action executed",

        (
            f"{payment_id} · "
            f"{action}"
        ),

        row["amount"],

        {

            "model_probability": prob,

            "success_probability": chance,

            "policy_decision":
                policy["decision"],

            "executed": True,
        },
    )

    # -----------------------------------------------------
    # LOG SUCCESS
    # -----------------------------------------------------

    insert_event(

        "success",

        payment_id,

        "Payment recovered",

        (
            f"{payment_id} · "
            f"{action}"
        ),

        row["amount"],

        {

            "model_probability": prob,

            "success_probability": chance,

            "recovered_value":
                row["amount"],
        },
    )

    # -----------------------------------------------------
    # FETCH UPDATED PAYMENT
    # -----------------------------------------------------

    updated_payment = payment_record(
        payment_id
    )

    # -----------------------------------------------------
    # RETURN RESULT TO FRONTEND
    # -----------------------------------------------------

    return {

        "ok": True,

        "result": "recovered",

        "payment": normalize_payment(
            updated_payment
        ),

        "policy": policy,

        "success_probability": chance,

        "recovered_value":
            row["amount"],
    }


# =========================================================
# AI AGENT CYCLE
# =========================================================

@app.post("/api/agent/cycle")
def agent_cycle():

    conn = get_conn()

    rows = conn.execute(
    """
    SELECT *
    FROM payments
    WHERE
        status='failed'
        AND recovery_probability >= 0.80
        AND amount < 100000
        AND retry_count < 3
    ORDER BY recoverable_value DESC
    LIMIT 20
    """
    ).fetchall()
    conn.close()

    summary = {

        "scanned": 0,

        "eligible": 0,

        "recovered": 0,

        "failed": 0,

        "escalated": 0,

        "no_action": 0,

        "attempts": 0,

        "recovered_value": 0,
    }

    for row in rows:

        summary["scanned"] += 1

        policy = decide(

            dict(row),

            float(
                row[
                    "recovery_probability"
                ]
            ),

            DEFAULT_POLICY,
        )

        # High-value transactions require approval.

        if (
            policy["decision"]
            == "manual_review"
        ):

            summary[
                "escalated"
            ] += 1

            insert_event(

                "escalation",

                row["payment_id"],

                "Escalated to merchant",

                (
                    f"{row['payment_id']} · "
                    "high-value approval required"
                ),

                row["amount"],
            )

            continue

        # Policy rejected action.

        if not policy["allowed"]:

            summary[
                "no_action"
            ] += 1

            continue

        summary[
            "eligible"
        ] += 1

        summary[
            "attempts"
        ] += 1

        result = payment_action(

            row["payment_id"],

            ActionRequest(

                action=row[
                    "recommended_action"
                ]
            ),
        )

        if (
            result["result"]
            == "recovered"
        ):

            summary[
                "recovered"
            ] += 1

            summary[
                "recovered_value"
            ] += row["amount"]

        else:

            summary[
                "failed"
            ] += 1

    # -----------------------------------------------------
    # CYCLE EVENT
    # -----------------------------------------------------

    insert_event(

        "analysis",

        None,

        "Recovery cycle completed",

        (
            f"Processed "
            f"{summary['scanned']} "
            "high-confidence opportunities · "
            f"{summary['recovered']} recovered"
        ),

        summary[
            "recovered_value"
        ],

        summary,
    )

    return summary


# =========================================================
# RESET DEMO
# =========================================================

@app.post("/api/demo/reset")
def reset_demo():

    from .db import clear_demo_data

    clear_demo_data()

    return {
        "ok": True
    }


# =========================================================
# SERVE FRONTEND
# =========================================================

@app.get("/")
def serve_frontend():

    return FileResponse(
        ROOT / "index.html"
    )


app.mount(
    "/",
    StaticFiles(
        directory=str(ROOT),
        html=True,
    ),
    name="frontend",
)