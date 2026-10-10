"""Work moves off a provider that is out of quota, and comes back to it later."""
import base64

import httpx

from app.core.health import ProviderHealth, is_quota_error
from app.services.summary import OwnerSummaryWriter
from app.services.vision_service import MockProvider, VisionService

GOOD = {"severity": "minor", "confidence": 0.95, "trade": "plumber", "is_safe_for_diy": True, "summary": "Dripping tap"}
IMG = base64.b64encode(b"\x89PNG\r\n\x1a\n fake").decode()


def http_error(code):
    req = httpx.Request("POST", "https://x")
    return httpx.HTTPStatusError("err", request=req, response=httpx.Response(code, request=req))


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def test_quota_errors_are_recognised():
    assert is_quota_error(http_error(429)) and is_quota_error(http_error(402))
    assert not is_quota_error(http_error(500))
    assert is_quota_error(RuntimeError("429 RESOURCE_EXHAUSTED quota"))


def test_a_cooling_provider_goes_to_the_back_then_returns():
    clock = Clock()
    h = ProviderHealth(clock=clock, quota_cooldown=300, error_cooldown=30)
    items = ["gemini", "groq", "openrouter"]
    assert h.order(items, lambda x: x) == items
    h.failed("gemini", http_error(429))
    assert h.order(items, lambda x: x) == ["groq", "openrouter", "gemini"]
    clock.t = 301
    assert h.order(items, lambda x: x) == items


def test_a_plain_error_cools_down_for_less_than_a_quota_error():
    clock = Clock()
    h = ProviderHealth(clock=clock, quota_cooldown=300, error_cooldown=30)
    h.failed("a", http_error(500)); h.failed("b", http_error(429))
    clock.t = 31
    assert not h.cooling("a") and h.cooling("b")


def test_when_everything_is_cooling_all_are_still_tried_soonest_first():
    clock = Clock()
    h = ProviderHealth(clock=clock)
    h.failed("a", http_error(429)); clock.t = 10; h.failed("b", http_error(429))
    assert h.order(["b", "a"], lambda x: x) == ["a", "b"]


def test_the_next_photo_skips_the_provider_that_just_ran_out_of_quota():
    first = MockProvider([RuntimeError("429 RESOURCE_EXHAUSTED")], name="gemini")
    second = MockProvider([GOOD], name="groq")
    service = VisionService([first, second])
    service.triage(IMG, "image/png")
    service.triage(IMG, "image/png")
    assert first.calls == 1  # not asked again for the second ticket
    assert second.calls == 2


def test_summaries_move_to_the_next_provider_and_stay_there():
    calls = {"or": 0, "groq": 0}
    text = "A pipe in the wall is leaking steadily and needs a plumber within a day."

    def openrouter(prompt):
        calls["or"] += 1
        raise http_error(402)

    def groq(prompt):
        calls["groq"] += 1
        return text

    w = OwnerSummaryWriter([("openrouter", openrouter), ("groq", groq)])
    assert w.write({"a": 1}) == text and w.write({"a": 1}) == text
    assert calls == {"or": 1, "groq": 2}


def test_a_server_hiccup_is_retried_once_before_giving_up_on_the_provider():
    from app.core.health import is_transient

    assert is_transient(RuntimeError("500 INTERNAL. Internal error encountered"))
    assert is_transient(http_error(503))
    assert not is_transient(http_error(429))

    flaky = MockProvider([RuntimeError("500 INTERNAL"), GOOD], name="gemini")
    other = MockProvider([GOOD], name="groq")
    VisionService([flaky, other], retry_delay=0).triage(IMG, "image/png")
    assert (flaky.calls, other.calls) == (2, 0)  # the second try on the same provider worked


def test_json_wrapped_in_prose_is_still_read():
    from app.services.vision_service import parse_assessment
    import json

    obs = {"summary": "Dripping tap", "trade": "plumber", "confidence": 0.95, "evidence": {
        "water": "dripping", "damage": "none", "electrical_hazard": False, "gas_or_fire": False,
        "structural_risk": False, "sewage": False, "simple_fix": True, "note_matches_photo": None}}
    wrapped = "Here is my assessment:\n" + json.dumps(obs) + "\nHope that helps."
    assert parse_assessment(wrapped, 0.75, True).is_safe_for_diy
