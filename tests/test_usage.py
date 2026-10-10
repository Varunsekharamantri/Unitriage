import json
import threading

import pytest

from app.core.usage import Usage, UsageTracker
from app.services.vision_service import MockProvider, VisionService

IMG = "aGVsbG8="
DRAIN = {
    "severity": "minor", "confidence": 0.9, "trade": "plumber",
    "is_safe_for_diy": True, "summary": "x",
}


def test_cost_uses_per_million_prices():
    t = UsageTracker({"m": (1.0, 2.0)})
    assert t.estimate_cost("m", Usage(500_000, 250_000)) == pytest.approx(1.0)


def test_unpriced_model_is_free():
    t = UsageTracker()
    t.record("p", "free-model", Usage(10_000, 500))
    s = t.summary()
    assert s["cost_usd"] == 0 and s["input_tokens"] == 10_000 and s["calls"] == 1


def test_summary_groups_by_provider_and_model():
    t = UsageTracker()
    t.record("gemini", "g1", Usage(100, 10))
    t.record("gemini", "g1", Usage(200, 20))
    t.record("groq", "l1", Usage(50, 5))
    s = t.summary()
    assert s["calls"] == 3 and s["input_tokens"] == 350 and s["output_tokens"] == 35
    g = next(r for r in s["by_model"] if r["provider"] == "gemini")
    assert g["calls"] == 2 and g["input_tokens"] == 300


def test_tokens_are_counted_even_when_output_is_invalid():
    t = UsageTracker()
    provider = MockProvider(["garbage", json.dumps(DRAIN)], usage=Usage(100, 10))
    VisionService([provider], retries_per_provider=2, tracker=t).triage(IMG)
    assert t.summary()["calls"] == 2  # the wasted call and the good one


def test_tracker_is_thread_safe():
    t = UsageTracker()

    def work():
        for _ in range(500):
            t.record("p", "m", Usage(1, 1))

    threads = [threading.Thread(target=work) for _ in range(8)]
    [th.start() for th in threads]
    [th.join() for th in threads]
    assert t.summary()["calls"] == 4000
