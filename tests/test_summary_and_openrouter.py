"""The owner's AI-written summary, and OpenRouter as a last-resort provider."""
import base64
import re
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver

from app.adapters.mock_notifier import MockNotifier
from app.api.main import create_app
from app.core.security import MockVault, UnitSecrets
from app.core.usage import UsageTracker
from app.graph.workflow import build_workflow
from app.services.summary import OwnerSummaryWriter, build_summarizer, clean
from app.services.vision_service import MockProvider, OpenRouterProvider, VisionService

PNG = b"\x89PNG\r\n\x1a\n fake image bytes"
BURST = {
    "severity": "major", "confidence": 0.95, "trade": "plumber",
    "is_safe_for_diy": False, "summary": "Steady leak from a wall pipe",
}
GOOD = "A pipe in the wall is leaking steadily and should be dealt with within a day to limit water damage."


# ---- OpenRouter ----------------------------------------------------------

def test_openrouter_provider_posts_an_image_chat_request(monkeypatch):
    seen = {}

    class Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, json=json, headers=headers)
        return Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    p = OpenRouterProvider("sk-test", "google/gemini-2.5-flash")
    result = p.generate(base64.b64encode(PNG).decode(), "image/png", "tap dripping")

    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert seen["json"]["model"] == "google/gemini-2.5-flash"
    assert "response_format" not in seen["json"]  # not every OpenRouter model supports JSON mode
    parts = seen["json"]["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert (result.usage.input_tokens, result.usage.output_tokens) == (7, 3)
    assert p.name == "openrouter"


def test_openrouter_is_used_when_the_other_providers_fail():
    down = MockProvider([RuntimeError("429 quota")], name="gemini")
    down2 = MockProvider([RuntimeError("503")], name="groq")
    ok = MockProvider([BURST], name="openrouter")
    a = VisionService([down, down2, ok]).triage(base64.b64encode(PNG).decode(), "image/png")
    assert a.summary == "Steady leak from a wall pipe"
    assert (down.calls, down2.calls, ok.calls) == (1, 1, 1)


def test_openrouter_is_only_built_when_it_has_a_key(monkeypatch):
    from app import config
    from app.services import vision_service as vs

    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    with pytest.raises(vs.TriageError):
        vs.build_default_service(["openrouter"])
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-x")
    service = vs.build_default_service(["openrouter"])
    assert [p.name for p in service._providers] == ["openrouter"]


# ---- the summary writer --------------------------------------------------

def test_the_summary_never_shows_the_ai_the_price():
    seen = []
    w = OwnerSummaryWriter([("groq", lambda p: seen.append(p) or GOOD)])
    out = w.write({"severity": "major", "quote_usd": "175.00", "eta": "2 PM", "tenant_note": "leak"})
    assert out == GOOD
    assert "175" not in seen[0] and "2 PM" not in seen[0]


def test_a_reply_that_names_a_price_or_is_unusable_is_rejected():
    assert clean("The repair will cost $175 and is urgent, so please act today.") is None
    assert clean("ok") is None
    assert clean("x" * 600) is None
    assert clean("**Leak:** a pipe is leaking in the wall, needing a plumber soon.") == (
        "Leak: a pipe is leaking in the wall, needing a plumber soon."
    )


def test_a_failing_provider_falls_through_and_all_failing_gives_none():
    def boom(prompt):
        raise httpx.ConnectError("down")

    assert OwnerSummaryWriter([("groq", boom), ("gemini", lambda p: GOOD)]).write({"a": 1}) == GOOD
    assert OwnerSummaryWriter([("groq", boom)]).write({"a": 1}) is None


def test_summary_mode_off_builds_no_writer():
    assert build_summarizer("off") is None


# ---- on the real approval card ------------------------------------------

class Stack:
    def __init__(self, tmp_path, summarizer):
        self.conn = sqlite3.connect(str(tmp_path / "s.sqlite"), check_same_thread=False)
        self.notifier = MockNotifier(echo=False)
        vault = MockVault({"vault/12B": UnitSecrets("12B", "4821#")})
        vision = VisionService([MockProvider([BURST])])
        workflow = build_workflow(vision, self.notifier, vault, SqliteSaver(self.conn), summarizer)
        self.client = TestClient(create_app(
            workflow=workflow, vault=vault, usage=UsageTracker(), notifier=self.notifier)).__enter__()

    def card(self):
        r = self.client.post("/api/tickets/intake", files={"image": ("p.png", PNG, "image/png")},
                             data={"tenant_id": "tenant-12B", "unit_id": "12B", "description": "pipe leaking"})
        tid = r.json()["ticket_id"]
        q = self.client.post("/api/webhooks/technician-quote", json={"ticket_id": tid, "quote": 175.0, "eta": "2 PM"})
        assert q.status_code == 200, q.text
        msgs = self.notifier.since(0, "manager", None)
        return next(m for m in msgs if m["subject"].startswith("Approval needed"))["body"]

    def close(self):
        self.client.__exit__(None, None, None)
        self.conn.close()


def test_the_approval_card_uses_the_ai_summary_and_prints_the_quote_itself(tmp_path):
    s = Stack(tmp_path, OwnerSummaryWriter([("groq", lambda p: GOOD)]))
    try:
        body = s.card()
    finally:
        s.close()
    assert GOOD in body
    assert "Quote: $175.00, ETA 2 PM." in body
    assert "4821" not in body  # the access code is never on the card


def test_the_card_falls_back_to_the_template_when_the_ai_is_down(tmp_path):
    def boom(prompt):
        raise httpx.ConnectError("down")

    s = Stack(tmp_path, OwnerSummaryWriter([("groq", boom)]))
    try:
        body = s.card()
    finally:
        s.close()
    assert "Major plumber issue: Steady leak from a wall pipe." in body
    assert re.search(r"Quote: \$175\.00, ETA 2 PM\.", body)


def test_without_a_summarizer_the_card_is_unchanged(tmp_path):
    s = Stack(tmp_path, None)
    try:
        body = s.card()
    finally:
        s.close()
    assert "Major plumber issue: Steady leak from a wall pipe." in body
