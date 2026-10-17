import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import db, main, policy  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    with TestClient(main.app) as c:
        yield c
    policy.update_policy({"auto_recovery": True})


def test_metrics_match_the_payments_table(client):
    m = client.get("/api/metrics").json()
    conn = db.get_conn()
    open_value = conn.execute("SELECT SUM(amount) FROM payments WHERE status='failed'").fetchone()[0]
    recovered = conn.execute("SELECT SUM(recovered_value) FROM payments").fetchone()[0] or 0
    conn.close()
    assert m["atRisk"] == round(open_value)
    assert m["recovered"] == round(recovered)
    assert m["rate"] == round(100 * recovered / (recovered + open_value), 1)
    assert m["recoverable"] <= m["atRisk"]


def test_policy_only_allows_the_kill_switch(client):
    assert client.get("/api/policy").json()["auto_recovery"] is True
    assert client.post("/api/policy", json={"auto_recovery": False}).json()["auto_recovery"] is False
    assert client.get("/api/policy").json()["high_value_approval"] is True
    with pytest.raises(ValueError):
        policy.update_policy({"high_value_approval": False})


def test_paused_agent_takes_no_actions(client):
    client.post("/api/policy", json={"auto_recovery": False})
    before = client.get("/api/metrics").json()
    result = client.post("/api/agent/cycle").json()
    assert result["paused"] is True
    assert client.get("/api/metrics").json()["recovered"] == before["recovered"]


def test_events_expose_ids_and_payment_ids(client):
    client.post("/api/agent/cycle")
    ev = client.get("/api/events?limit=10").json()["events"]
    assert ev and all("id" in e and "metadata" in e for e in ev)
