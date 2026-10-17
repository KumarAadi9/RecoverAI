from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import get_conn, init_db, insert_event
import numpy as np
import pandas as pd

from . import ml_engine
from . import simulation as sim
from .policy import decide, DEFAULT_POLICY, ACTIVE_POLICY, update_policy


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


# Recovery attempts succeed with the modelled probability. Set RECOVERAI_SEED to make
# runs reproducible (used by the tests).
_seed = os.environ.get("RECOVERAI_SEED")
OUTCOME_RNG = random.Random(int(_seed)) if _seed else random.Random()

# How many payments one agent cycle may act on.
AGENT_BATCH_SIZE = 20


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup() -> None:
    init_db()
    ml_engine.ensure_model()
    # Replace the dataset's generated probability with the trained model's score.
    ml_engine.rescore_payments()


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

        "delayHours": d.get("recommended_delay_h") or 0,
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

                "id": row["id"],

                "payment_id": row["payment_id"],

                "metadata": json.loads(row["metadata_json"] or "{}"),

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
    """Dashboard numbers computed from the payments table (no assumed percentages).

    atRisk: value of payments still failed. recoverable: expected value (probability x amount)
    of those open payments. recovered: value recovered so far. rate: recovered value as a share
    of all failed value (recovered + still open)."""

    limit = DEFAULT_POLICY["high_value_threshold"]

    conn = get_conn()

    row = conn.execute(
        """
        SELECT
            COALESCE(SUM(CASE WHEN status='failed' THEN amount END), 0) AS open_value,
            COALESCE(SUM(CASE WHEN status='failed' THEN recoverable_value END), 0) AS expected,
            COALESCE(SUM(recovered_value), 0) AS recovered,
            SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
            SUM(CASE WHEN status='recovered' THEN 1 ELSE 0 END) AS successful,
            SUM(CASE WHEN status='failed' AND recovery_probability >= 0.80 THEN 1 ELSE 0 END) AS high_conf,
            SUM(CASE WHEN status='failed'
                      AND recommended_action NOT IN ('No action', 'Escalate') THEN 1 ELSE 0 END) AS actionable,
            SUM(CASE WHEN status='failed' AND amount >= ? THEN 1 ELSE 0 END) AS awaiting_n,
            COALESCE(SUM(CASE WHEN status='failed' AND amount >= ? THEN amount END), 0) AS awaiting_value
        FROM payments
        """,
        (limit, limit),
    ).fetchone()

    conn.close()

    open_value = float(row["open_value"] or 0)
    recovered = float(row["recovered"] or 0)
    total = open_value + recovered

    return {
        "atRisk": round(open_value),
        "recoverable": round(float(row["expected"] or 0)),
        "recovered": round(recovered),
        "openFailures": int(row["failed"] or 0),
        "successfulRecoveries": int(row["successful"] or 0),
        "highConfidence": int(row["high_conf"] or 0),
        "actionable": int(row["actionable"] or 0),
        "awaitingApproval": int(row["awaiting_n"] or 0),
        "awaitingApprovalValue": round(float(row["awaiting_value"] or 0)),
        "rate": round(100 * recovered / total, 1) if total else 0,
    }


# =========================================================
# POLICY
# =========================================================

class PolicyUpdate(BaseModel):
    auto_recovery: bool | None = None


@app.get("/api/policy")
def get_policy():
    return {**ACTIVE_POLICY, "agent_batch_size": AGENT_BATCH_SIZE}


@app.post("/api/policy")
def set_policy(payload: PolicyUpdate):
    changes = payload.model_dump(exclude_none=True)
    try:
        policy = update_policy(changes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    insert_event(
        "analysis",
        None,
        "Policy changed",
        "Automatic recovery " + ("enabled" if policy["auto_recovery"] else "disabled"),
        0,
        {"policy": policy},
    )
    return {**policy, "agent_batch_size": AGENT_BATCH_SIZE}


# =========================================================
# RECOVERY OUTCOME
# =========================================================

def world_success_probability(row: dict, action: str) -> float:
    """True chance that this action succeeds right now, from the simulated world (simulation.py).

    The model only *estimates* this; execution in the demo is drawn from the world itself, so a
    recommendation can turn out wrong, as it would in real life."""
    kind = ml_engine.LABEL_KIND.get(action, "retry")
    pay = pd.DataFrame([row])[["failure_reason", "payment_method", "amount",
                               "customer_previous_payments", "customer_success_rate"]]
    attempt_no = min(int(row["retry_count"]) + 1, sim.MAX_ATTEMPTS)
    delay = float(row.get("recommended_delay_h") or 0)
    frame = sim.attempt_frame(pay, kind, delay, attempt_no, np.array([sim.now_ist()]))
    return float(sim.true_success_prob(frame)[0])


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

    rec = ml_engine.recommend(record)

    prob, action, signals = rec["prob"], rec["action"], rec["signals"]

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

            recommended_action,

            recommended_delay_h

        )
        VALUES(
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?
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

            rec["delay_h"],
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
        ACTIVE_POLICY,
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

    chance = world_success_probability(
        row,
        action,
    )

    # -----------------------------------------------------
    # EXECUTE THE ATTEMPT
    # -----------------------------------------------------
    #
    # The outcome is sampled from the success probability
    # (model score adjusted for the action), so an attempt
    # can fail. Every attempt uses up one retry.
    # -----------------------------------------------------

    success = OUTCOME_RNG.random() < chance

    conn = get_conn()

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
    # FAILED ATTEMPT
    # -----------------------------------------------------

    if not success:

        failed_payment = payment_record(
            payment_id
        )

        insert_event(

            "retry_failed",

            payment_id,

            "Recovery attempt failed",

            (
                f"{payment_id} · "
                f"{action} · "
                f"attempt {failed_payment['retry_count']} did not recover the payment"
            ),

            row["amount"],

            {

                "model_probability": prob,

                "success_probability": chance,

                "policy_decision":
                    policy["decision"],

                "executed": True,

                "retry_count":
                    failed_payment["retry_count"],
            },
        )

        return {

            "ok": True,

            "result": "retry_failed",

            "payment": normalize_payment(
                failed_payment
            ),

            "policy": policy,

            "success_probability": chance,
        }

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
    """One autonomous pass over the open failures.

    Every open failed payment is evaluated by policy.decide() in order of expected
    recoverable value. High-value payments are routed to merchant approval, payments
    below the confidence threshold or at the retry limit are skipped, and the agent
    acts on up to AGENT_BATCH_SIZE eligible payments.
    """

    if not ACTIVE_POLICY["auto_recovery"]:

        insert_event(
            "no_action",
            None,
            "Agent paused",
            "Automatic recovery is disabled, so no actions were taken",
            0,
        )

        return {
            "paused": True, "scanned": 0, "eligible": 0, "recovered": 0, "failed": 0,
            "escalated": 0, "no_action": 0, "attempts": 0, "recovered_value": 0,
            "awaiting_approval_value": 0,
        }

    conn = get_conn()

    rows = conn.execute(
        """
        SELECT *
        FROM payments
        WHERE status='failed'
        ORDER BY recoverable_value DESC
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

        "awaiting_approval_value": 0,
    }

    for row in rows:

        if summary["attempts"] >= AGENT_BATCH_SIZE:
            break

        summary["scanned"] += 1

        policy = decide(
            dict(row),
            float(row["recovery_probability"]),
            ACTIVE_POLICY,
        )

        # High-value payments wait for merchant approval.

        if policy["decision"] == "manual_review":

            summary["escalated"] += 1

            summary["awaiting_approval_value"] += row["amount"]

            continue

        # The model found no action worth taking (expected value not positive).

        if row["recommended_action"] == "No action":

            summary["no_action"] += 1

            continue

        # Below the confidence threshold or out of retries.

        if not policy["allowed"]:

            summary["no_action"] += 1

            continue

        summary["eligible"] += 1

        summary["attempts"] += 1

        result = payment_action(
            row["payment_id"],
            ActionRequest(action=row["recommended_action"]),
        )

        if result["result"] == "recovered":

            summary["recovered"] += 1

            summary["recovered_value"] += row["amount"]

        else:

            summary["failed"] += 1

    # -----------------------------------------------------
    # CYCLE EVENTS
    # -----------------------------------------------------

    if summary["escalated"]:

        insert_event(

            "escalation",

            None,

            "Awaiting merchant approval",

            (
                f"{summary['escalated']} high-value payments "
                "need approval before any automatic action"
            ),

            summary["awaiting_approval_value"],

            {"count": summary["escalated"]},
        )

    insert_event(

        "analysis",

        None,

        "Recovery cycle completed",

        (
            f"Acted on {summary['attempts']} payments · "
            f"{summary['recovered']} recovered, "
            f"{summary['failed']} failed"
        ),

        summary["recovered_value"],

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

    ml_engine.rescore_payments()

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