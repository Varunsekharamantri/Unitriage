"""Screening the tenant's note for dangers a photo can't show (a gas smell, sparks heard...)."""
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.adapters.mock_notifier import MockNotifier
from app.core.schemas import TicketState
from app.core.security import MockVault, UnitSecrets
from app.graph.workflow import build_workflow
from app import config
from app.services.note_screen import (
    HAZARDS,
    KeywordNoteScreener,
    build_note_screener,
    check_note,
    describe,
    flags_for,
)
from app.services.vision_service import MockProvider, VisionService

IMG = "aGVsbG8="
DRAIN = {  # a calm, minor, DIY-able photo
    "severity": "minor", "confidence": 0.92, "trade": "plumber",
    "is_safe_for_diy": True, "summary": "Slow kitchen drain",
}
EMERGENCY = {**DRAIN, "severity": "emergency", "is_safe_for_diy": False, "summary": "Flooding"}


class FakeScreener:
    """Returns fixed probabilities when the note contains a trigger word."""

    name = "fake"

    def __init__(self, triggers=None):
        self.triggers = triggers or {}
        self.calls = []

    def screen(self, note):
        self.calls.append(note)
        probs = {k: 0.0 for k in HAZARDS}
        for word, (hazard, p) in self.triggers.items():
            if word in note.lower():
                probs[hazard] = p
        return probs


class BrokenScreener:
    name = "broken"

    def screen(self, note):
        raise RuntimeError("model could not be loaded")


GAS = {"gas": ("gas", 0.9)}


# --- the keyword check ----------------------------------------------------------------


@pytest.mark.parametrize(
    "note, expected",
    [
        ("I can smell gas in the kitchen", {"gas"}),
        ("There is a gas leak near the stove", {"gas"}),
        ("smells like something is burning from the socket", {"fire_smoke"}),
        ("smoke is coming out of the wall", {"fire_smoke"}),
        ("the socket gave me a shock", {"electrical"}),
        ("sparks flew from the switch", {"electrical"}),
        ("water is pouring through the ceiling", {"flooding"}),
        ("a pipe burst and the floor is flooded", {"flooding"}),
        ("the ceiling is sagging and about to fall", {"collapse"}),
        ("my mother is trapped in the bathroom", {"danger"}),
        ("the tap drips a bit at night", set()),
        ("the light bulb in the hallway is out", set()),
        ("paint is peeling on the bedroom wall", set()),
        ("URGENT URGENT EMERGENCY please fix my dripping tap now", set()),  # shouting is not a hazard
    ],
)
def test_keyword_check_finds_the_obvious_dangers(note, expected):
    probs = KeywordNoteScreener().screen(note)
    assert {k for k, p in probs.items() if p >= 0.5} == expected


@pytest.mark.parametrize(
    "note",
    [
        "no smell of gas, just the igniter clicking",
        "it is not flooding, only a small drip",
        "I don't smell any gas",
        "there is no smoke, only dust",
    ],
)
def test_keyword_check_ignores_simple_negations(note):
    assert not any(KeywordNoteScreener().screen(note).values())


# --- helpers --------------------------------------------------------------------------


def test_flags_come_out_most_urgent_first_and_respect_the_threshold():
    probs = {"flooding": 0.9, "gas": 0.6, "danger": 0.2, "electrical": 0.5}
    assert flags_for(probs, 0.5) == ["gas", "electrical", "flooding"]
    assert flags_for(probs, 0.95) == []
    assert flags_for({}, 0.5) == []


def test_describe_reads_naturally():
    assert describe([]) == ""
    assert describe(["gas"]) == "a smell of gas"
    assert describe(["gas", "flooding"]) == "a smell of gas and flooding or a burst pipe"
    assert describe(["danger", "gas", "flooding"]).count(",") == 1
    assert describe(["no_water", "no_power"]) == "no water supply and a power cut"
    assert describe(["no_water", "vulnerable"]) == "no water supply"   # the modifier isn't read out as a problem


def test_a_blank_note_or_no_screener_means_no_flags_and_no_call():
    screener = FakeScreener(GAS)
    assert check_note(screener, "   ", 0.5).flags == [] and screener.calls == []
    assert check_note(None, "gas smell", 0.5).flags == []


def test_a_crashing_screener_falls_back_to_keywords_rather_than_failing_silently():
    check = check_note(BrokenScreener(), "I smell gas", 0.5)
    assert check.flags == ["gas"] and check.method == "keywords (fallback)"
    assert check_note(BrokenScreener(), "the tap drips", 0.5).flags == []


def test_choosing_a_screener(monkeypatch):
    assert build_note_screener("off") is None
    assert isinstance(build_note_screener("keywords"), KeywordNoteScreener)
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    assert isinstance(build_note_screener("llm"), KeywordNoteScreener)  # no key: keywords, not a crash
    for gone in ("laya", "laya-choice", "auto", "magic"):               # Laya was evaluated and dropped
        with pytest.raises(ValueError):
            build_note_screener(gone)


# --- inside the vision service ----------------------------------------------------------


def service(response=DRAIN, screener=None, threshold=0.5):
    return VisionService([MockProvider([response])], note_screener=screener, note_threshold=threshold)


def test_a_reported_gas_smell_turns_a_calm_minor_photo_into_an_emergency():
    a = service(screener=FakeScreener(GAS)).triage(IMG, "image/jpeg", "I think I smell gas")
    assert a.severity == "emergency" and a.is_safe_for_diy is False
    assert a.note_flags == ["gas"]
    assert "tenant's note reports a smell of gas" in a.reasons[-1] and "fake" in a.reasons[-1]


def test_a_note_below_the_threshold_changes_nothing():
    a = service(screener=FakeScreener({"gas": ("gas", 0.3)})).triage(IMG, "image/jpeg", "maybe gas?")
    assert (a.severity, a.is_safe_for_diy, a.note_flags) == ("minor", True, [])


def test_a_note_can_only_raise_severity_never_lower_it():
    reassuring = FakeScreener()  # sees no hazard in "it's nothing, don't worry"
    a = service(EMERGENCY, reassuring).triage(IMG, "image/jpeg", "it's nothing, don't worry")
    assert a.severity == "emergency"
    both = service(EMERGENCY, FakeScreener(GAS)).triage(IMG, "image/jpeg", "gas")
    assert both.severity == "emergency" and both.note_flags == ["gas"]


def test_no_note_means_the_screener_is_never_asked():
    screener = FakeScreener(GAS)
    a = service(screener=screener).triage(IMG)
    assert screener.calls == [] and a.severity == "minor"


def test_if_the_screener_crashes_the_service_still_catches_an_obvious_danger():
    a = service(screener=BrokenScreener()).triage(IMG, "image/jpeg", "there is a strong smell of gas")
    assert a.severity == "emergency" and a.note_flags == ["gas"]
    assert "keywords (fallback)" in a.reasons[-1]


def test_the_note_can_be_screened_with_no_photo_analysis_at_all():
    assert service(screener=FakeScreener(GAS)).screen_note("gas smell").flags == ["gas"]
    assert service().screen_note("gas smell").flags == []  # no screener configured


# --- through the whole graph -------------------------------------------------------------


class Env:
    def __init__(self, tmp_path, provider_response, screener):
        self.notifier = MockNotifier(echo=False)
        self.conn = sqlite3.connect(str(tmp_path / "n.sqlite"), check_same_thread=False)
        vault = MockVault({"vault/12B": UnitSecrets("12B", "4821#")})
        vision = VisionService([MockProvider([provider_response])], note_screener=screener)
        self.graph = build_workflow(vision, self.notifier, vault, SqliteSaver(self.conn))
        self.config = {"configurable": {"thread_id": "T-1"}}

    def start(self, note):
        self.graph.invoke(
            TicketState(ticket_id="T-1", tenant_id="t1", tenant_name="Priya", unit_id="12B",
                        image_b64=IMG, vault_key="vault/12B", description=note),
            self.config,
        )

    @property
    def paused(self):
        tasks = self.graph.get_state(self.config).tasks
        return tasks[0].interrupts[0].value["step"] if tasks and tasks[0].interrupts else None


@pytest.fixture
def env(tmp_path):
    made = []

    def factory(response=DRAIN, screener=None):
        e = Env(tmp_path, response, screener)
        made.append(e)
        return e

    yield factory
    for e in made:
        e.conn.close()


def test_a_gas_note_skips_the_diy_path_and_the_tenant_gets_gas_safety_advice(env):
    e = env(DRAIN, FakeScreener(GAS))
    e.start("I can smell gas")
    assert e.paused == "wait_for_quote"                      # not "wait_for_feedback": no DIY
    tenant = e.notifier.to("tenant")[0].body
    assert "Because you mentioned a smell of gas" in tenant
    assert "Don't use switches" in tenant and "leave the building" in tenant
    assert "emergency number" in tenant
    rfq = e.notifier.to("technician")[0]
    assert "⚠️ The tenant reports a smell of gas (not verified from the photo)" in rfq.body


def test_without_a_hazard_in_the_note_the_ticket_takes_the_normal_diy_path(env):
    e = env(DRAIN, FakeScreener(GAS))
    e.start("the sink drains slowly")
    assert e.paused == "wait_for_feedback"


def test_when_the_photo_ai_is_down_a_gas_note_still_raises_the_alarm(env):
    e = env(RuntimeError("quota exhausted"), FakeScreener(GAS))
    e.start("I smell gas")
    snapshot = e.graph.get_state(e.config).values
    assert snapshot["status"] == "manual_review" and snapshot["note_flags"] == ["gas"]
    manager = e.notifier.to("manager")[0].body
    assert "URGENT" in manager and "a smell of gas" in manager
    tenant = e.notifier.to("tenant")[0].body
    assert "treating this as urgent" in tenant and "Don't use switches" in tenant


def test_when_the_photo_ai_is_down_and_there_is_no_hazard_it_is_an_ordinary_manual_review(env):
    e = env(RuntimeError("quota exhausted"), FakeScreener(GAS))
    e.start("tap drips")
    assert "URGENT" not in e.notifier.to("manager")[0].body
    assert "treating this as urgent" not in e.notifier.to("tenant")[0].body


def test_the_most_urgent_reported_danger_decides_the_advice(env):
    both = FakeScreener({"flood": ("flooding", 0.9), "hurt": ("danger", 0.9)})
    e = env(DRAIN, both)
    e.start("water is flooding in and my dad is hurt")
    tenant = e.notifier.to("tenant")[0].body
    assert "hurt or in danger" in tenant and "main water valve" not in tenant
