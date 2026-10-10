"""The chat simulator's backend: conversations, per-phone feeds, action buttons, DEMO_UI switch."""
import json
import sqlite3
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from app import config
from app.adapters.mock_notifier import MockNotifier
from app.api.main import create_app
from app.core.security import MockVault, UnitSecrets, sign_token
from app.graph.workflow import build_workflow
from app.services.vision_service import MockProvider, VisionService

PNG = b"\x89PNG\r\n\x1a\n fake image bytes"
CODE = "4821#"
BURST = {
    "severity": "emergency", "confidence": 0.97, "trade": "plumber",
    "is_safe_for_diy": False, "summary": "Burst pipe flooding the floor",
}
DRAIN = {
    "severity": "minor", "confidence": 0.92, "trade": "plumber",
    "is_safe_for_diy": True, "summary": "Slow kitchen drain",
}


@pytest.fixture
def make_client(tmp_path):
    opened = []

    def factory(vision_response=BURST):
        conn = sqlite3.connect(str(tmp_path / f"demo{len(opened)}.sqlite"), check_same_thread=False)
        notifier = MockNotifier(echo=False)
        vault = MockVault({"vault/12B": UnitSecrets("12B", CODE)})
        workflow = build_workflow(
            VisionService([MockProvider([vision_response])]), notifier, vault, SqliteSaver(conn)
        )
        c = TestClient(create_app(workflow=workflow, vault=vault, notifier=notifier)).__enter__()
        opened.append((c, conn))
        return c

    yield factory
    for c, conn in opened:
        c.__exit__(None, None, None)
        conn.close()


@pytest.fixture
def client(make_client):
    return make_client(BURST)


def start_ticket(client, tenant_id="tenant-12B", name="Priya Sharma", description=""):
    r = client.post(
        "/api/tickets/intake",
        files={"image": ("p.png", PNG, "image/png")},
        data={"tenant_id": tenant_id, "unit_id": "12B", "tenant_name": name, "description": description},
    )
    assert r.status_code == 201, r.text
    return r.json()["ticket_id"]


def quote(client, tid, amount=175, eta="2 PM"):
    r = client.post("/api/webhooks/technician-quote", json={"ticket_id": tid, "quote": amount, "eta": eta})
    assert r.status_code == 200, r.text


def feed(client, **params):
    return client.get("/api/demo/messages", params=params).json()


def path_of(url):
    p = urlparse(url)
    return f"{p.path}?{p.query}"


def approval_card(client):
    return next(m for m in feed(client) if m["subject"].startswith("Approval needed"))


def approve(client):
    r = client.get(path_of(approval_card(client)["actions"][0]["url"]))
    assert r.status_code == 200
    return r.json()


# --- pages and config -------------------------------------------------------


def test_root_and_demo_serve_the_three_phone_page_and_phone_serves_the_app(client):
    for path in ("/", "/demo"):
        r = client.get(path)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        for role in ("tenant", "technician", "owner"):
            assert f"/phone?role={role}" in r.text
    assert client.get("/phone").status_code == 200


def test_demo_config_exposes_the_contact_names(client, monkeypatch):
    monkeypatch.setattr(config, "OWNER_NAME", "Priya Nair")
    monkeypatch.setattr(config, "TECHNICIAN_NAME", "Lee Park")
    cfg = client.get("/api/demo/config").json()
    assert cfg["owner_name"] == "Priya Nair" and cfg["technician_name"] == "Lee Park"
    assert cfg["tenants"] == config.DEMO_TENANTS and "12B" in cfg["tenants"]


# --- conversations ------------------------------------------------------------


def test_every_message_has_a_sender_a_recipient_and_a_conversation(client):
    tid = start_ticket(client, description="Water everywhere")
    quote(client, tid)
    approve(client)
    got = [(m["sender"], m["recipient"], m["conversation"]) for m in feed(client)]
    assert got == [
        ("tenant", "manager", "tenant:tenant-12B"),       # the tenant's photo and words
        ("manager", "tenant", "tenant:tenant-12B"),       # emergency advice, as the owner
        ("manager", "technician", "technician"),          # quote request with the photo
        ("technician", "manager", "technician"),          # the quote
        ("system", "manager", "technician"),              # approval card, owner only
        ("manager", "technician", "technician"),          # "approved"
        ("manager", "technician", "technician"),          # dispatch
        ("manager", "tenant", "tenant:tenant-12B"),       # confirmation
    ]


def test_the_owner_is_notified_when_the_tenant_sends_something(client):
    tid = start_ticket(client, description="Pipe burst under the sink")
    first = feed(client, viewer="manager", conversation="tenant:tenant-12B")[0]
    assert (first["sender"], first["recipient"]) == ("tenant", "manager")
    assert first["body"] == "Pipe burst under the sink"
    assert first["attachments"][0]["kind"] == "image" and first["ticket_id"] == tid


def test_tenant_and_technician_feeds_hold_only_their_own_conversation(client):
    tid = start_ticket(client)
    quote(client, tid)
    approve(client)

    tenant = feed(client, viewer="tenant")
    tech = feed(client, viewer="technician")
    assert {m["conversation"] for m in tenant} == {"tenant:tenant-12B"}
    assert {m["conversation"] for m in tech} == {"technician"}
    # nothing crosses over: the tenant never sees the quote, the price or the access code
    assert CODE not in json.dumps(tenant) and "175" not in json.dumps(tenant)
    assert all(m["subject"] == "" or "Quote" not in m["subject"] for m in tenant)
    # and the owner, party to both, sees everything
    assert len(feed(client, viewer="manager")) == len(feed(client))


def test_a_different_tenant_cannot_see_another_tenants_chat(client):
    start_ticket(client, tenant_id="tenant-12B", name="Priya Sharma")
    assert feed(client, viewer="tenant", conversation="tenant:tenant-3A") == []
    assert len(feed(client, viewer="tenant", conversation="tenant:tenant-12B")) >= 2


def test_viewer_must_be_a_known_role(client):
    assert client.get("/api/demo/messages", params={"viewer": "stranger"}).status_code == 422


def test_feed_is_incremental(client):
    tid = start_ticket(client)
    n = len(feed(client))
    assert feed(client, after=n) == []
    quote(client, tid)
    assert [m["sender"] for m in feed(client, after=n)] == ["technician", "system"]


# --- the technician must never be able to approve their own quote -----------------


def test_the_technician_never_receives_approval_links_or_buttons(client):
    tid = start_ticket(client)
    quote(client, tid)
    tech = feed(client, viewer="technician")
    blob = json.dumps(tech)
    assert "decision=" not in blob and "manager-approval" not in blob
    assert all(a["kind"] not in ("approve", "reject") for m in tech for a in m["actions"])
    # the technician does see their own quote
    assert any(m["sender"] == "technician" and "Quote: $175.00" in m["body"] for m in tech)
    # the owner-only card exists, for the owner
    card = approval_card(client)
    assert card["sender"] == "system" and card["recipient"] == "manager"
    assert [a["kind"] for a in card["actions"]] == ["approve", "reject"]
    assert card not in tech


# --- buttons ------------------------------------------------------------------


def test_rfq_has_a_quote_action_for_the_technician(client):
    start_ticket(client)
    rfq = next(m for m in feed(client) if m["subject"].startswith("Quote request"))
    assert rfq["recipient"] == "technician" and [a["kind"] for a in rfq["actions"]] == ["quote"]


def test_approval_card_buttons_work_by_path_regardless_of_host(client):
    tid = start_ticket(client)
    quote(client, tid)
    assert approve(client)["status"] == "dispatched"
    assert client.get(f"/api/tickets/{tid}").json()["awaiting"] is None


# --- photos -------------------------------------------------------------------


def test_photo_is_visible_in_the_tenants_message_and_the_technicians_enquiry(client):
    start_ticket(client)
    msgs = feed(client)
    with_photo = [m for m in msgs if m["attachments"]]
    assert {(m["sender"], m["recipient"]) for m in with_photo} == {
        ("tenant", "manager"),
        ("manager", "technician"),
    }
    for m in with_photo:
        att = m["attachments"][0]
        assert att["kind"] == "image"
        r = client.get(path_of(att["url"]))
        assert r.status_code == 200 and r.content == PNG
        assert r.headers["content-type"] == "image/png" and r.headers["x-content-type-options"] == "nosniff"


def test_photo_is_available_straight_away_before_triage_has_finished(client):
    tid = start_ticket(client)
    # served from the photo cache, not only from the ticket checkpoint
    assert client.app.state.photos.get(tid) == (PNG, "image/png")


def test_photo_endpoint_rejects_bad_tokens_before_revealing_anything(client):
    tid = start_ticket(client)
    assert client.get(f"/api/tickets/{tid}/photo?token=forged").status_code == 403
    assert client.get(f"/api/tickets/{tid}/photo").status_code == 422  # token is required
    # an unknown ticket and a bad token look the same, so ids can't be probed
    assert client.get("/api/tickets/T-nope/photo?token=forged").status_code == 403
    # a token for another action can't be reused to fetch the photo
    assert client.get(f"/api/tickets/{tid}/photo?token={sign_token(tid, 'approved')}").status_code == 403
    # a valid token for an unknown ticket is a plain 404
    assert client.get(f"/api/tickets/T-nope/photo?token={sign_token('T-nope', 'photo')}").status_code == 404


# --- the content of the messages -----------------------------------------------


def test_owner_card_names_the_tenant_and_unit_but_the_technician_is_not_told_before_approval(client):
    tid = start_ticket(client, name="Priya Sharma")
    assert client.get(f"/api/tickets/{tid}").json()["tenant_name"] == "Priya Sharma"
    quote(client, tid)
    assert "Tenant: Priya Sharma · Unit 12B" in approval_card(client)["body"]
    tech = json.dumps(feed(client, viewer="technician"))
    assert "Priya" not in tech and "12B" not in tech
    approve(client)
    dispatch = next(m for m in feed(client, viewer="technician") if m["subject"].startswith("DISPATCH"))
    assert "tenant: Priya Sharma" in dispatch["body"] and CODE in dispatch["body"]


def test_tenant_name_falls_back_to_the_tenant_id(client):
    tid = start_ticket(client, name="")
    assert client.get(f"/api/tickets/{tid}").json()["tenant_name"] == "tenant-12B"


def test_emergency_tenant_message_has_trade_specific_safety_advice(client):
    start_ticket(client)
    tenant = next(m for m in feed(client, viewer="tenant") if m["sender"] == "manager")
    for text in ("emergency", "main water valve", "emergency number", "right now"):
        assert text in tenant["body"].lower() or text in tenant["body"]


def test_approval_confirmation_to_tenant_says_the_owner_approved(client):
    tid = start_ticket(client)
    quote(client, tid, 175, "2 PM")
    approve(client)
    confirm = [m for m in feed(client, viewer="tenant") if m["sender"] == "manager"][-1]
    assert "property owner approved" in confirm["body"] and "2 PM" in confirm["body"]
    assert "175" not in confirm["body"] and "4821" not in confirm["body"]


def test_technician_sees_the_owner_approve_then_the_dispatch(client):
    tid = start_ticket(client)
    quote(client, tid)
    approve(client)
    subjects = [m["subject"] for m in feed(client, viewer="technician")]
    assert subjects == [
        f"Quote request for ticket {tid} (emergency)",  # from the owner
        f"Quote for ticket {tid}",                       # the technician's own quote
        f"Quote approved: ticket {tid}",
        f"DISPATCH: ticket {tid}",
    ]


# --- the tenant's own messages after the DIY guide ---------------------------------


def test_the_tenants_feedback_is_recorded_in_their_chat(make_client):
    c = make_client(DRAIN)
    tid = start_ticket(c)
    r = c.post(f"/api/tickets/{tid}/feedback", json={"resolved": False, "note": "Plunger did not help"})
    assert r.status_code == 200
    mine = [m for m in feed(c, viewer="tenant") if m["sender"] == "tenant"]
    assert [m["body"] for m in mine] == ["", "Plunger did not help"]  # the photo, then the reply

    # asking for a technician is recorded as the tenant's own message too
    tid3 = start_ticket(c, tenant_id="tenant-12B")
    c.post(f"/api/tickets/{tid3}/feedback", json={"resolved": False, "want_technician": True})
    assert [m["body"] for m in feed(c, viewer="tenant") if m["sender"] == "tenant"][-1] == "🔧 Please send a technician"
    tid4 = start_ticket(c, tenant_id="tenant-12B")
    c.post(f"/api/tickets/{tid4}/feedback", json={"resolved": False, "want_technician": True, "note": "I'm out until 5"})
    assert [m["body"] for m in feed(c, viewer="tenant") if m["sender"] == "tenant"][-1] == "I'm out until 5"

    tid2 = start_ticket(c, tenant_id="tenant-12B")
    c.post(f"/api/tickets/{tid2}/feedback", json={"resolved": True})
    assert [m["body"] for m in feed(c, viewer="tenant") if m["sender"] == "tenant"][-1] == "✅ It's fixed, thanks!"


# --- the switch ----------------------------------------------------------------


def test_demo_ui_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_UI", False)
    conn = sqlite3.connect(str(tmp_path / "off.sqlite"), check_same_thread=False)
    vault = MockVault({"vault/12B": UnitSecrets("12B", CODE)})
    workflow = build_workflow(
        VisionService([MockProvider([BURST])]), MockNotifier(echo=False), vault, SqliteSaver(conn)
    )
    with TestClient(create_app(workflow=workflow, vault=vault)) as c:
        assert c.get("/api/demo/messages").status_code == 404
        assert c.get("/phone").status_code == 404
        r = c.get("/", follow_redirects=False)
        assert r.status_code in (302, 307) and r.headers["location"] == "/docs"
    conn.close()


def test_plain_tenant_messages_reach_the_owner(client):
    r = client.post("/api/chat", json={"unit_id": "12B", "messages": [
        {"sender": "tenant", "body": "Hi, the heating is making a noise"},
        {"sender": "manager", "body": "Please send a photo so I can assess it."}]})
    assert r.status_code == 201 and r.json() == {"sent": 2}
    owner = feed(client, viewer="manager")
    assert [(m["sender"], m["recipient"], m["conversation"]) for m in owner] == [
        ("tenant", "manager", "tenant:tenant-12B"), ("manager", "tenant", "tenant:tenant-12B")]
    assert feed(client, viewer="technician") == []


def test_chat_rejects_an_unknown_unit_and_empty_input(client):
    assert client.post("/api/chat", json={"unit_id": "99Z", "messages": [{"sender": "tenant", "body": "hi"}]}).status_code == 422
    assert client.post("/api/chat", json={"unit_id": "12B", "messages": []}).status_code == 422
    assert client.post("/api/chat", json={"unit_id": "12B", "messages": [{"sender": "system", "body": "x"}]}).status_code == 422
