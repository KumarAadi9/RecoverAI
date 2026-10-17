import hashlib
import hmac
import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db, main, policy  # noqa: E402

CSV = (
    "payment_id,customer_id,amount,payment_method,failure_reason,timestamp,"
    "customer_previous_payments,customer_success_rate\n"
    "REAL001,C1,2500,UPI,Network timeout,2026-10-10 11:30:00,12,0.9\n"
    "REAL002,C2,18000,Card,Card expired,2026-10-11 09:00:00,4,0.7\n"
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.delenv("RECOVERAI_API_KEY", raising=False)
    with TestClient(main.app) as c:
        yield c
    policy.update_policy({"auto_recovery": True})


def test_import_scores_new_payments_and_skips_duplicates(client):
    r = client.post("/api/payments/import", content=CSV, headers={"content-type": "text/csv"})
    assert r.json() == {"imported": 2, "skipped_duplicates": 0}
    p = client.get("/api/payments/REAL002").json()
    p = p.get("payment", p)
    assert p["status"] != "Recovered" and 0 < p["prob"] <= 100
    again = client.post("/api/payments/import", content=CSV, headers={"content-type": "text/csv"})
    assert again.json()["skipped_duplicates"] == 2


def test_import_rejects_bad_rows(client):
    bad = CSV.replace("0.9", "7")
    r = client.post("/api/payments/import", content=bad, headers={"content-type": "text/csv"})
    assert r.status_code == 400
    r = client.post("/api/payments/import", content="a,b\n1,2\n", headers={"content-type": "text/csv"})
    assert r.status_code == 400 and "Missing columns" in r.json()["detail"]


def sign(secret, body):
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_webhook_requires_valid_signature_and_applies_outcome(client, monkeypatch):
    client.post("/api/payments/import", content=CSV, headers={"content-type": "text/csv"})
    body = json.dumps({"payment_id": "REAL001", "success": True, "action": "retry"}).encode()
    assert client.post("/api/webhooks/outcome", content=body).status_code == 503
    monkeypatch.setenv("RECOVERAI_WEBHOOK_SECRET", "s3cret")
    assert client.post("/api/webhooks/outcome", content=body,
                       headers={"x-signature": "bad"}).status_code == 401
    ok = client.post("/api/webhooks/outcome", content=body, headers={"x-signature": sign("s3cret", body)})
    assert ok.json()["result"] == "recovered"
    conn = db.get_conn()
    assert conn.execute("SELECT status FROM payments WHERE payment_id='REAL001'").fetchone()[0] == "recovered"
    assert conn.execute("SELECT COUNT(*) FROM attempts WHERE source='webhook'").fetchone()[0] == 1
    conn.close()


def test_api_key_guards_writes_only(client, monkeypatch):
    monkeypatch.setenv("RECOVERAI_API_KEY", "k")
    assert client.get("/api/metrics").status_code == 200
    assert client.post("/api/agent/cycle").status_code == 401
    assert client.post("/api/agent/cycle", headers={"x-api-key": "k"}).status_code == 200


def test_attempts_are_logged_and_retrain_uses_them(client):
    client.post("/api/agent/cycle")
    n = client.get("/api/analytics").json()["attempts"]["logged"]
    assert n > 0
    r = client.post("/api/model/retrain").json()
    assert r["retrained"] and r["logged_attempts"] == n
    assert client.get("/api/model/info").json()["logged_attempts_used"] == n


def test_analytics_are_computed_from_payments(client):
    a = client.get("/api/analytics").json()
    assert sum(x["n"] for x in a["failureMix"]) == 5000
    assert sum(d["n"] for d in a["daily"]) == 5000
