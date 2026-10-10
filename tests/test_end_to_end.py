"""Full ticket lifecycles through the real HTTP API, graph, SQLite and failover layers.

Only the model providers, the notifier and the vault are fakes.
"""
import re
import sqlite3
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from app.adapters.mock_notifier import MockNotifier
from app.api.main import create_app
from app.core.security import MockVault, UnitSecrets
from app.core.usage import Usage, UsageTracker
from app.graph.workflow import build_workflow
from app.services.vision_service import MockProvider, VisionService

CODE_12B, CODE_3A = "4821#", "7305#"
PNG = b"\x89PNG\r\n\x1a\n fake image bytes"
PRICES = {"gemini-model": (0.10, 0.40), "groq-model": (0.05, 0.08)}  # USD per 1M tokens

DRAIN = {
    "severity": "minor", "confidence": 0.92, "trade": "plumber",
    "is_safe_for_diy": True, "summary": "Slow kitchen drain",
}
BURST = {
    "severity": "emergency", "confidence": 0.97, "trade": "plumber",
    "is_safe_for_diy": False, "summary": "Burst pipe flooding the floor",
}


class Stack:
    """One deployment: sqlite file + graph + API. restart() keeps only the file."""

    def __init__(self, tmp_path, providers):
        self.db = str(tmp_path / "e2e.sqlite")
        self.providers = providers
        self.notifier = MockNotifier(echo=False)
        self.vault = MockVault(
            {"vault/12B": UnitSecrets("12B", CODE_12B), "vault/3A": UnitSecrets("3A", CODE_3A)}
        )
        self.tracker = UsageTracker(PRICES)
        self.conn = None
        self.client = None
        self.start()

    def start(self):
        self.conn = sqlite3.connect(self.db, check_same_thread=False)
        vision = VisionService(self.providers, tracker=self.tracker)
        workflow = build_workflow(vision, self.notifier, self.vault, SqliteSaver(self.conn))
        # entering the client runs the app's startup, which attaches workflow/vault/usage
        self.client = TestClient(
            create_app(
                workflow=workflow, vault=self.vault, usage=self.tracker, notifier=self.notifier
            )
        ).__enter__()

    def close(self):
        self.client.__exit__(None, None, None)
        self.conn.close()

    def restart(self):
        self.close()
        self.start()

    def intake(self, unit="12B"):
        r = self.client.post(
            "/api/tickets/intake",
            files={"image": ("photo.png", PNG, "image/png")},
            data={"tenant_id": "tenant-1", "unit_id": unit},
        )
        assert r.status_code == 201, r.text
        return r.json()

    def quote(self, ticket_id, amount=175.0, eta="2 PM"):
        r = self.client.post(
            "/api/webhooks/technician-quote",
            json={"ticket_id": ticket_id, "quote": amount, "eta": eta},
        )
        assert r.status_code == 200, r.text
        return r.json()

    def decide(self, ticket_id, decision):
        """Follow the one-click link from the manager card addressed to this ticket."""
        card = next(
            m for m in reversed(self.notifier.to("manager")) if f"ticket {ticket_id}" in m.subject
        )
        url = re.search(rf"(http\S+decision={decision}\S+)", card.body).group(1)
        p = urlparse(url)
        r = self.client.get(f"{p.path}?{p.query}")
        assert r.status_code == 200, r.text
        return r.json()

    def status(self, ticket_id):
        return self.client.get(f"/api/tickets/{ticket_id}").json()


@pytest.fixture
def make_stack(tmp_path):
    stacks = []

    def factory(*providers):
        s = Stack(tmp_path, list(providers))
        stacks.append(s)
        return s

    yield factory
    for s in stacks:
        s.close()


def test_happy_path_minor_diy(make_stack):
    stack = make_stack(MockProvider([DRAIN], name="gemini"))
    ticket = stack.intake()

    assert ticket["status"] == "awaiting_feedback"
    assert ticket["awaiting"] == "wait_for_feedback"
    assert stack.notifier.to("tenant")[0].subject.endswith("you can fix this yourself")

    # the tenant confirms the DIY fix worked: the ticket closes
    r = stack.client.post(
        f"/api/tickets/{ticket['ticket_id']}/feedback", json={"resolved": True}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "resolved" and r.json()["awaiting"] is None
    assert stack.notifier.to("tenant")[-1].subject.endswith("resolved")
    # the technician is never bothered, and the owner only sees the tenant's own messages
    assert stack.notifier.to("technician") == []
    assert {m.sender for m in stack.notifier.to("manager")} == {"tenant"}
    assert all(CODE_12B not in m.body for m in stack.notifier.sent)


def test_escalation_path_major_quote_approval_dispatch(make_stack):
    stack = make_stack(MockProvider([BURST], name="gemini"))

    ticket = stack.intake()
    tid = ticket["ticket_id"]
    assert ticket["awaiting"] == "wait_for_quote"
    assert ticket["assessment"]["severity"] == "emergency"

    assert stack.quote(tid)["awaiting"] == "wait_for_approval"
    final = stack.decide(tid, "approved")

    assert final["status"] == "dispatched" and final["awaiting"] is None
    assert final["dispatch_packet"]["quote"] == 175.0
    # exact order of the owner/technician conversation: enquiry -> the technician's quote ->
    # the owner-only approval card -> approved -> dispatch
    work = [m for m in stack.notifier.sent if m.conversation == "technician"]
    assert [(m.sender, m.subject.split(":")[0].split(" for ")[0]) for m in work] == [
        ("manager", "Quote request"),
        ("technician", "Quote"),
        ("system", "Approval needed"),
        ("manager", "Quote approved"),
        ("manager", "DISPATCH"),
    ]
    # the tenant is kept informed at both ends, without ever seeing the access code
    tenant = stack.notifier.to("tenant")
    assert [m.subject.split(": ")[1] for m in tenant] == ["technician requested", "technician on the way"]
    dispatch = work[-1]
    assert "Unit 12B" in dispatch.body and CODE_12B in dispatch.body
    # the secret reached the technician and nowhere else, including the API and the DB
    assert CODE_12B not in str(final)
    assert all(CODE_12B not in m.body for m in stack.notifier.sent if m is not dispatch)
    stack.conn.commit()
    assert CODE_12B.encode() not in open(stack.db, "rb").read()


def test_escalation_survives_server_restart_between_every_step(make_stack):
    stack = make_stack(MockProvider([BURST], name="gemini"))
    tid = stack.intake()["ticket_id"]

    stack.restart()
    assert stack.status(tid)["awaiting"] == "wait_for_quote"
    stack.quote(tid)

    stack.restart()
    assert stack.status(tid)["awaiting"] == "wait_for_approval"
    assert stack.decide(tid, "approved")["status"] == "dispatched"


def test_concurrent_tickets_do_not_mix_state(make_stack):
    stack = make_stack(MockProvider([BURST], name="gemini"))
    a = stack.intake("12B")["ticket_id"]
    b = stack.intake("3A")["ticket_id"]

    stack.quote(a, 100.0, "1 PM")
    stack.quote(b, 250.0, "5 PM")
    stack.decide(b, "approved")
    stack.decide(a, "rejected")

    assert stack.status(a)["status"] == "rejected_rerouted"
    assert stack.status(b)["status"] == "dispatched"
    assert stack.status(b)["dispatch_packet"]["quote"] == 250.0
    dispatches = [m for m in stack.notifier.to("technician") if m.subject.startswith("DISPATCH")]
    assert len(dispatches) == 1
    assert "Unit 3A" in dispatches[0].body and CODE_3A in dispatches[0].body
    assert CODE_12B not in dispatches[0].body


def test_failover_still_completes_ticket_and_costs_are_tracked(make_stack):
    primary = MockProvider([RuntimeError("429 rate limited")], name="gemini")
    backup = MockProvider([BURST], name="groq", usage=Usage(2000, 200))
    stack = make_stack(primary, backup)

    tid = stack.intake()["ticket_id"]
    stack.quote(tid)
    assert stack.decide(tid, "approved")["status"] == "dispatched"

    usage = stack.client.get("/api/usage").json()
    # the failed gemini call produced no usage; only groq was billed
    assert usage["calls"] == 1
    assert usage["input_tokens"] == 2000 and usage["output_tokens"] == 200
    assert usage["by_model"][0]["provider"] == "groq"
    assert usage["cost_usd"] == pytest.approx((2000 * 0.05 + 200 * 0.08) / 1_000_000)


def test_triage_outage_routes_to_manual_review(make_stack):
    stack = make_stack(MockProvider([RuntimeError("down")], name="gemini"))
    ticket = stack.intake()
    assert ticket["status"] == "manual_review"
    # the owner is told in the tenant's chat, as an owner-only notice, besides the tenant's message
    notices = [m for m in stack.notifier.to("manager") if m.sender == "system"]
    assert len(notices) == 1 and notices[0].conversation == "tenant:tenant-1"
    assert stack.client.get("/api/usage").json()["calls"] == 0
