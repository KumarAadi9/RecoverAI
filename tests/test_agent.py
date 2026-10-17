import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db, main, ml_engine  # noqa: E402
from backend.policy import DEFAULT_POLICY, decide  # noqa: E402


class FixedRng:
    """Stand-in for the outcome RNG: random() always returns `value`."""

    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    with TestClient(main.app) as c:  # runs startup: seed, train, rescore
        yield c


def open_payments(where=""):
    conn = db.get_conn()
    rows = [dict(r) for r in conn.execute(f"SELECT * FROM payments WHERE status='failed' {where}")]
    conn.close()
    return rows


def events(event_type):
    conn = db.get_conn()
    rows = [dict(r) for r in conn.execute("SELECT * FROM events WHERE event_type=?", (event_type,))]
    conn.close()
    return rows


# ---------- policy ----------
def test_policy_gates():
    base = {"amount": 5000, "retry_count": 0}
    assert decide(base, 0.9)["decision"] == "execute"
    assert decide({**base, "amount": DEFAULT_POLICY["high_value_threshold"]}, 0.99)["decision"] == "manual_review"
    strict = {"recovery_threshold": 0.6}
    assert decide(base, 0.59, strict)["decision"] == "no_action"    # below a 60% confidence floor
    assert decide(base, 0.61, strict)["decision"] == "execute"
    assert decide({**base, "retry_count": 3}, 0.9)["decision"] == "no_action"   # retry limit
    assert decide(base, 0.9, {"auto_recovery": False})["decision"] == "manual_review"


# ---------- model ----------
def test_model_probabilities_are_calibrated(client):
    info = ml_engine.MODEL_INFO
    assert abs(info["mean_predicted"] - info["mean_observed"]) < 0.03


# ---------- outcomes ----------
def test_failed_attempt_stays_open_and_uses_a_retry(client, monkeypatch):
    monkeypatch.setattr(main, "OUTCOME_RNG", FixedRng(0.999))  # always above the success chance
    p = open_payments("AND amount < 100000 AND retry_count < 3 AND recommended_action NOT IN ('No action', 'Escalate') LIMIT 1")[0]
    r = client.post(f"/api/payments/{p['payment_id']}/action", json={"action": p["recommended_action"]}).json()
    assert r["result"] == "retry_failed"
    after = db.get_conn().execute("SELECT status, retry_count, recovered_value FROM payments WHERE payment_id=?",
                                  (p["payment_id"],)).fetchone()
    assert after["status"] == "failed"
    assert after["retry_count"] == p["retry_count"] + 1
    assert after["recovered_value"] == 0
    assert any(e["payment_id"] == p["payment_id"] for e in events("retry_failed"))


def test_successful_attempt_records_recovery(client, monkeypatch):
    monkeypatch.setattr(main, "OUTCOME_RNG", FixedRng(0.0))  # always succeeds
    p = open_payments("AND amount < 100000 AND retry_count < 3 AND recommended_action NOT IN ('No action', 'Escalate') LIMIT 1")[0]
    r = client.post(f"/api/payments/{p['payment_id']}/action", json={"action": p["recommended_action"]}).json()
    assert r["result"] == "recovered"
    after = db.get_conn().execute("SELECT status, recovered_value FROM payments WHERE payment_id=?",
                                  (p["payment_id"],)).fetchone()
    assert after["status"] == "recovered" and after["recovered_value"] == p["amount"]


# ---------- agent cycle ----------
def test_cycle_obeys_policy_and_can_fail(client, monkeypatch):
    monkeypatch.setattr(main, "OUTCOME_RNG", FixedRng(0.999))
    before = {p["payment_id"]: p for p in open_payments()}
    s = client.post("/api/agent/cycle").json()

    assert s["attempts"] == main.AGENT_BATCH_SIZE
    assert s["recovered"] == 0 and s["failed"] == s["attempts"]          # no free recoveries
    assert s["eligible"] == s["recovered"] + s["failed"]
    assert s["escalated"] > 0                                            # high-value queue is real

    acted = [e for e in events("retry_failed")]
    assert len(acted) == s["attempts"]
    for e in acted:
        pay = before[e["payment_id"]]
        assert pay["amount"] < DEFAULT_POLICY["high_value_threshold"]    # never auto-act on high value
        assert pay["retry_count"] < DEFAULT_POLICY["max_retries"]
        assert pay["recovery_probability"] >= DEFAULT_POLICY["recovery_threshold"]
        assert pay["recommended_action"] not in ("No action", "Escalate")   # only acts on real recommendations


def test_cycle_stops_retrying_after_the_limit(client, monkeypatch):
    monkeypatch.setattr(main, "OUTCOME_RNG", FixedRng(0.999))
    for _ in range(6):
        client.post("/api/agent/cycle")
    conn = db.get_conn()
    # the agent must never push a payment past the retry limit
    over = conn.execute(
        "SELECT COUNT(*) n FROM events e JOIN payments p ON p.payment_id=e.payment_id "
        "WHERE e.event_type='retry_failed' AND p.retry_count > ?", (DEFAULT_POLICY["max_retries"],)).fetchone()["n"]
    conn.close()
    assert over == 0
