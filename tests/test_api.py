import re
import sqlite3
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from app.adapters.mock_notifier import MockNotifier
from app.api.main import create_app
from app.core.security import MockVault, UnitSecrets
from app.graph.workflow import build_workflow
from app.services.vision_service import MockProvider, VisionService

CODE = "9137#"
PNG = b"\x89PNG\r\n\x1a\n fake image bytes"

DRAIN = {
    "severity": "minor", "confidence": 0.92, "trade": "plumber",
    "is_safe_for_diy": True, "summary": "Slow kitchen drain",
}
BURST = {
    "severity": "emergency", "confidence": 0.97, "trade": "plumber",
    "is_safe_for_diy": False, "summary": "Burst pipe flooding the floor",
}


@pytest.fixture
def make_client(tmp_path):
    opened = []

    def factory(vision_response):
        conn = sqlite3.connect(str(tmp_path / "api.sqlite"), check_same_thread=False)
        opened.append(conn)
        notifier = MockNotifier(echo=False)
        vault = MockVault({"vault/12B": UnitSecrets("12B", CODE)})
        vision = VisionService([MockProvider([vision_response])])
        workflow = build_workflow(vision, notifier, vault, SqliteSaver(conn))
        client = TestClient(create_app(workflow=workflow, vault=vault))
        client.notifier = notifier
        return client

    yield factory
    for c in opened:
        c.close()


def intake(client, unit="12B", content_type="image/png", data=PNG):
    return client.post(
        "/api/tickets/intake",
        files={"image": ("leak.png", data, content_type)},
        data={"tenant_id": "tenant-1", "unit_id": unit},
    )


def approval_path(client, decision):
    body = client.notifier.to("manager")[-1].body
    url = re.search(rf"(http\S+decision={decision}\S+)", body).group(1)
    p = urlparse(url)
    return f"{p.path}?{p.query}"


def test_minor_intake_waits_for_tenant_feedback(make_client):
    with make_client(DRAIN) as client:
        r = intake(client)
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "awaiting_feedback" and body["awaiting"] == "wait_for_feedback"
        assert "image_b64" not in body and "vault_key" not in body


def test_full_escalation_flow_over_http(make_client):
    with make_client(BURST) as client:
        body = intake(client).json()
        tid = body["ticket_id"]
        assert body["awaiting"] == "wait_for_quote"

        r = client.post(
            "/api/webhooks/technician-quote",
            json={"ticket_id": tid, "quote": 175.0, "eta": "2 PM"},
        )
        assert r.status_code == 200 and r.json()["awaiting"] == "wait_for_approval"

        r = client.get(approval_path(client, "approved"))
        assert r.status_code == 200
        assert r.json()["status"] == "dispatched"
        assert CODE in client.notifier.to("technician")[-1].body
        assert CODE not in str(client.get(f"/api/tickets/{tid}").json())


def test_rejection_over_http(make_client):
    with make_client(BURST) as client:
        tid = intake(client).json()["ticket_id"]
        client.post(
            "/api/webhooks/technician-quote",
            json={"ticket_id": tid, "quote": 900, "eta": "next week"},
        )
        r = client.get(approval_path(client, "rejected"))
        assert r.json()["status"] == "rejected_rerouted"


def test_approval_link_is_single_use(make_client):
    with make_client(BURST) as client:
        tid = intake(client).json()["ticket_id"]
        client.post(
            "/api/webhooks/technician-quote",
            json={"ticket_id": tid, "quote": 175, "eta": "2 PM"},
        )
        path = approval_path(client, "approved")
        assert client.get(path).status_code == 200
        assert client.get(path).status_code == 409


def test_description_is_accepted_and_returned(make_client):
    with make_client(BURST) as client:
        r = client.post(
            "/api/tickets/intake",
            files={"image": ("leak.png", PNG, "image/png")},
            data={"tenant_id": "t", "unit_id": "12B", "description": "  Leak under the sink  "},
        )
        assert r.status_code == 201 and r.json()["description"] == "Leak under the sink"
        too_long = client.post(
            "/api/tickets/intake",
            files={"image": ("leak.png", PNG, "image/png")},
            data={"tenant_id": "t", "unit_id": "12B", "description": "x" * 501},
        )
        assert too_long.status_code == 422


def test_feedback_endpoint_escalates_and_validates(make_client):
    with make_client(DRAIN) as client:
        tid = intake(client).json()["ticket_id"]
        assert client.post("/api/tickets/T-nope/feedback", json={"resolved": True}).status_code == 404
        assert client.post(f"/api/tickets/{tid}/feedback", json={"resolved": "maybe"}).status_code == 422
        assert (
            client.post(f"/api/tickets/{tid}/feedback", json={"resolved": False, "note": "x" * 501}).status_code
            == 422
        )
        r = client.post(f"/api/tickets/{tid}/feedback", json={"resolved": False, "note": "Still leaking"})
        assert r.status_code == 200 and r.json()["awaiting"] == "wait_for_quote"
        # the feedback gate is single-use
        again = client.post(f"/api/tickets/{tid}/feedback", json={"resolved": True})
        assert again.status_code == 409


def test_tenant_can_skip_diy_and_ask_for_a_technician(make_client):
    with make_client(DRAIN) as client:
        tid = intake(client).json()["ticket_id"]
        # "fixed" and "send a technician" together make no sense
        both = client.post(f"/api/tickets/{tid}/feedback", json={"resolved": True, "want_technician": True})
        assert both.status_code == 422
        assert client.get(f"/api/tickets/{tid}").json()["awaiting"] == "wait_for_feedback"

        r = client.post(f"/api/tickets/{tid}/feedback", json={"resolved": False, "want_technician": True})
        assert r.status_code == 200 and r.json()["awaiting"] == "wait_for_quote"
        assert r.json()["feedback"]["want_technician"] is True


def test_feedback_when_ticket_is_not_at_the_gate_is_409(make_client):
    with make_client(BURST) as client:
        tid = intake(client).json()["ticket_id"]  # major: waiting on a quote, not feedback
        r = client.post(f"/api/tickets/{tid}/feedback", json={"resolved": True})
        assert r.status_code == 409


def test_forged_and_swapped_tokens_are_rejected(make_client):
    with make_client(BURST) as client:
        tid = intake(client).json()["ticket_id"]
        client.post(
            "/api/webhooks/technician-quote",
            json={"ticket_id": tid, "quote": 175, "eta": "2 PM"},
        )
        bad = client.get(
            "/api/webhooks/manager-approval",
            params={"ticket_id": tid, "decision": "approved", "token": "forged"},
        )
        assert bad.status_code == 403
        # an approve token must not authorise a reject, and vice versa
        approve_q = approval_path(client, "approved").split("token=")[1]
        swapped = client.get(
            "/api/webhooks/manager-approval",
            params={"ticket_id": tid, "decision": "rejected", "token": approve_q},
        )
        assert swapped.status_code == 403
        assert client.get(f"/api/tickets/{tid}").json()["awaiting"] == "wait_for_approval"


def test_quote_for_unknown_ticket_is_404(make_client):
    with make_client(BURST) as client:
        r = client.post(
            "/api/webhooks/technician-quote",
            json={"ticket_id": "T-nope", "quote": 10, "eta": "now"},
        )
        assert r.status_code == 404


def test_quote_when_not_at_gate_is_409(make_client):
    with make_client(DRAIN) as client:
        tid = intake(client).json()["ticket_id"]
        r = client.post(
            "/api/webhooks/technician-quote",
            json={"ticket_id": tid, "quote": 10, "eta": "now"},
        )
        assert r.status_code == 409


@pytest.mark.parametrize("quote", [0, -5])
def test_non_positive_quote_is_rejected(make_client, quote):
    with make_client(BURST) as client:
        tid = intake(client).json()["ticket_id"]
        r = client.post(
            "/api/webhooks/technician-quote",
            json={"ticket_id": tid, "quote": quote, "eta": "2 PM"},
        )
        assert r.status_code == 422
        assert client.get(f"/api/tickets/{tid}").json()["awaiting"] == "wait_for_quote"


def test_intake_validation(make_client):
    with make_client(BURST) as client:
        assert intake(client, content_type="text/plain").status_code == 415
        assert intake(client, data=b"").status_code == 422
        assert intake(client, unit="99Z").status_code == 422
        assert client.get("/api/tickets/T-nope").status_code == 404
