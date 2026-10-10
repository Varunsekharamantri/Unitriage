"""Photos are accepted at once and assessed in the background, retrying while the AI is busy."""
import sqlite3

from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from app.adapters.mock_notifier import MockNotifier
from app.api.main import create_app
from app.core.security import MockVault, UnitSecrets
from app.graph.workflow import build_workflow
from app.services.vision_service import MockProvider, VisionService

PNG = b"\x89PNG\r\n\x1a\n fake image bytes"
BURST = {"severity": "emergency", "confidence": 0.97, "trade": "plumber", "is_safe_for_diy": False,
         "summary": "Burst pipe"}


def stack(tmp_path, responses, retry_window=0.0, background=True):
    conn = sqlite3.connect(str(tmp_path / "b.sqlite"), check_same_thread=False)
    notifier = MockNotifier(echo=False)
    vault = MockVault({"vault/12B": UnitSecrets("12B", "4821#")})
    provider = MockProvider(responses)
    vision = VisionService([provider], retries_per_provider=1, retry_delay=0)
    wf = build_workflow(vision, notifier, vault, SqliteSaver(conn), retry_window=retry_window)
    wf_nodes = None
    client = TestClient(create_app(workflow=wf, vault=vault, notifier=notifier, background_intake=background)).__enter__()
    return client, notifier, provider


def intake(client):
    return client.post("/api/tickets/intake", files={"image": ("p.png", PNG, "image/png")},
                       data={"tenant_id": "tenant-12B", "unit_id": "12B", "description": "leak"})


def test_intake_answers_at_once_and_the_assessment_follows(tmp_path):
    client, notifier, _ = stack(tmp_path, [BURST])
    r = intake(client)
    assert r.status_code == 201 and r.json()["status"] == "received"
    tid = r.json()["ticket_id"]
    # the tenant is told straight away, and the assessment lands (TestClient runs background tasks)
    subjects = [m.subject for m in notifier.sent]
    assert any("received" in s for s in subjects)
    assert client.get(f"/api/tickets/{tid}").json()["status"] == "rfq_sent"
    client.__exit__(None, None, None)


def test_without_background_intake_the_response_carries_the_state(tmp_path):
    client, _, _ = stack(tmp_path, [BURST], background=False)
    assert intake(client).json()["status"] == "rfq_sent"
    client.__exit__(None, None, None)


def test_a_busy_provider_is_retried_instead_of_sending_the_ticket_to_review(tmp_path, monkeypatch):
    import app.graph.nodes as nodes

    monkeypatch.setattr(nodes.time, "sleep", lambda s: None)  # don't really wait
    client, _, provider = stack(tmp_path, [RuntimeError("429 busy"), RuntimeError("429 busy"), BURST], retry_window=60)
    r = intake(client)
    tid = r.json()["ticket_id"]
    t = client.get(f"/api/tickets/{tid}").json()
    assert t["status"] == "rfq_sent"           # it did get assessed, on the third try
    assert provider.calls == 3
    client.__exit__(None, None, None)


def test_it_still_ends_in_manual_review_when_the_providers_never_recover(tmp_path, monkeypatch):
    import app.graph.nodes as nodes

    clock = {"t": 0.0}
    monkeypatch.setattr(nodes.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(nodes.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    client, _, provider = stack(tmp_path, [RuntimeError("429 busy")], retry_window=30)
    tid = intake(client).json()["ticket_id"]
    assert client.get(f"/api/tickets/{tid}").json()["status"] == "triage_failed" or \
        client.get(f"/api/tickets/{tid}").json()["status"] == "manual_review"
    assert provider.calls >= 2
    client.__exit__(None, None, None)
