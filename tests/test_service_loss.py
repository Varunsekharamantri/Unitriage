"""Urgent (same-day, not an emergency) versus emergency, and the two new trades."""
import sqlite3
from datetime import date

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from app.adapters.mock_notifier import MockNotifier
from app.core.schemas import TicketState, VisualTriageAssessment
from app.core.security import MockVault, UnitSecrets
from app.graph.nodes import _who
from app.graph.workflow import build_workflow
from app.services.note_screen import (
    HAZARDS,
    LLM_PROMPT,
    KeywordNoteScreener,
    NoteCheck,
    assess_note,
    parse_hazards,
)
from app.services.vision_service import MockProvider, VisionService

IMG = "aGVsbG8="
DRAIN = {  # a calm, minor, DIY-able photo
    "severity": "minor", "confidence": 0.92, "trade": "plumber",
    "is_safe_for_diy": True, "summary": "Slow kitchen drain",
}
WINTER, SUMMER = date(2026, 1, 15), date(2026, 7, 15)


class Fake:
    name = "fake"

    def __init__(self, **triggers):  # word -> list of hazard keys it makes the screener report
        self.triggers = triggers

    def screen(self, note):
        probs = {k: 0.0 for k in HAZARDS}
        for word, keys in self.triggers.items():
            if word in note.lower():
                for k in keys:
                    probs[k] = 1.0
        return probs


def service(screener, today=WINTER, response=DRAIN):
    return VisionService([MockProvider([response])], note_screener=screener, today=lambda: today)


# --- the keyword check for the new situations ---------------------------------------------


@pytest.mark.parametrize(
    "note, expected",
    [
        ("There has been no water in the whole flat since this morning", {"no_water"}),
        ("we have no running water", {"no_water"}),
        ("the water has been cut off", {"no_water"}),
        ("No electricity since last night", {"no_power"}),
        ("there was a power cut and the lights are all out", {"no_power"}),
        ("no heating and it is freezing", {"no_heat"}),
        ("the boiler has stopped working", {"no_heat"}),
        ("the front door lock is broken", {"no_lock"}),
        ("I can't lock the back door", {"no_lock"}),
        ("someone broke into the flat last night", {"no_lock"}),
        ("we have no working toilet", {"no_toilet"}),
        ("there is no water and no power", {"no_water", "no_power"}),   # two losses in one note
        ("no heating and my newborn baby is cold", {"no_heat", "vulnerable"}),
        ("the tap drips a bit", set()),
        ("the hot water is lukewarm", set()),
        ("no hot water this morning", set()),          # a loss of HOT water is an ordinary repair
        ("there is no water leak, just a damp patch", set()),
        ("the door handle is a bit loose", set()),
        ("my baby's room needs repainting", {"vulnerable"}),  # the modifier alone is harmless: see assess_note
    ],
)
def test_keyword_check_for_service_losses(note, expected):
    probs = KeywordNoteScreener().screen(note)
    assert {k for k, p in probs.items() if p >= 0.5} == expected


# --- what a screened note means ----------------------------------------------------------------


def flags(*names, vulnerable=False):
    return NoteCheck(list(names), "fake", vulnerable)


def test_a_service_loss_is_urgent_not_an_emergency():
    a = assess_note(flags("no_water"), in_heating_season=True)
    assert (a.urgency, a.flags, a.vulnerable) == ("urgent", ["no_water"], False)


@pytest.mark.parametrize("hazard", ["gas", "fire_smoke", "electrical", "flooding", "collapse", "danger"])
def test_a_danger_is_an_emergency(hazard):
    assert assess_note(flags(hazard), True).urgency == "emergency"


@pytest.mark.parametrize("loss", ["no_water", "no_power", "no_heat"])
def test_losing_water_power_or_heat_with_a_vulnerable_person_is_an_emergency(loss):
    assert assess_note(flags(loss, vulnerable=True), True).urgency == "emergency"


def test_a_vulnerable_person_does_not_escalate_a_lock_or_toilet_problem_or_anything_alone():
    assert assess_note(flags("no_lock", vulnerable=True), True).urgency == "urgent"
    assert assess_note(flags("no_toilet", vulnerable=True), True).urgency == "urgent"
    assert assess_note(flags(vulnerable=True), True) .urgency == "none"   # "my baby's room needs painting"


def test_no_heating_only_counts_in_the_cold_season_unless_someone_is_vulnerable():
    assert assess_note(flags("no_heat"), in_heating_season=False).urgency == "none"
    assert assess_note(flags("no_heat"), in_heating_season=True).urgency == "urgent"
    assert assess_note(flags("no_heat", vulnerable=True), in_heating_season=False).urgency == "emergency"


# --- inside the vision service ------------------------------------------------------------------


def test_no_water_turns_a_calm_minor_photo_into_an_urgent_same_day_ticket():
    a = service(Fake(water=["no_water"])).triage(IMG, "image/jpeg", "no water at all")
    assert a.severity == "major" and a.note_urgency == "urgent"      # raised to major, not emergency
    assert a.is_safe_for_diy is False and a.note_flags == ["no_water"]
    assert "no water supply" in a.reasons[-1]


def test_an_urgent_note_never_lowers_a_photos_emergency():
    emergency = {**DRAIN, "severity": "emergency", "is_safe_for_diy": False}
    a = service(Fake(water=["no_water"])).triage(IMG, "image/jpeg", "no water")
    assert a.severity == "major"
    b = service(Fake(water=["no_water"]), response=emergency).triage(IMG, "image/jpeg", "no water")
    assert b.severity == "emergency" and b.note_urgency == "urgent"


def test_a_vulnerable_person_makes_no_water_an_emergency():
    a = service(Fake(water=["no_water", "vulnerable"])).triage(IMG, "image/jpeg", "no water, my father is elderly")
    assert a.severity == "emergency" and a.note_urgency == "emergency" and a.note_vulnerable is True
    assert "vulnerable person" in a.reasons[-1]


def test_the_season_comes_from_the_date_the_service_is_given():
    summer = service(Fake(heat=["no_heat"]), today=SUMMER).triage(IMG, "image/jpeg", "no heat")
    winter = service(Fake(heat=["no_heat"]), today=WINTER).triage(IMG, "image/jpeg", "no heat")
    assert summer.note_urgency == "none" and summer.severity == "minor"
    assert winter.note_urgency == "urgent" and winter.severity == "major"


# --- the messages ---------------------------------------------------------------------------------


class Env:
    def __init__(self, tmp_path, screener, response=DRAIN, today=WINTER):
        self.notifier = MockNotifier(echo=False)
        self.conn = sqlite3.connect(str(tmp_path / "s.sqlite"), check_same_thread=False)
        vault = MockVault({"vault/12B": UnitSecrets("12B", "4821#")})
        vision = VisionService([MockProvider([response])], note_screener=screener, today=lambda: today)
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

    def factory(screener, response=DRAIN, today=WINTER):
        e = Env(tmp_path, screener, response, today)
        made.append(e)
        return e

    yield factory
    for e in made:
        e.conn.close()


def test_no_water_gets_the_urgent_message_not_the_emergency_one(env):
    e = env(Fake(water=["no_water"]))
    e.start("There is no water in the whole flat")
    assert e.paused == "wait_for_quote"                       # straight to a technician, no DIY
    tenant = e.notifier.to("tenant")[0].body
    assert "we're treating this as urgent" in tenant and "to come today" in tenant
    assert "isn't an emergency" in tenant
    assert "This looks like an emergency" not in tenant and "treating this as an emergency" not in tenant
    assert "drinking water" in tenant                         # practical advice for this situation


def test_the_technician_sees_same_day_in_the_subject_and_the_reason(env):
    e = env(Fake(water=["no_water"]))
    e.start("no water")
    rfq = e.notifier.to("technician")[0]
    assert rfq.subject == "Quote request for ticket T-1 (major, SAME DAY)"
    assert "⚠️ SAME-DAY: The tenant reports no water supply (not verified from the photo)" in rfq.body


def test_an_ordinary_ticket_is_not_labelled_same_day(env):
    e = env(Fake(), {**DRAIN, "severity": "major", "is_safe_for_diy": False})
    e.start("a steady leak under the sink")
    assert "SAME DAY" not in e.notifier.to("technician")[0].subject


def test_a_vulnerable_person_turns_it_into_a_full_emergency_message(env):
    e = env(Fake(water=["no_water", "vulnerable"]))
    e.start("no water and my elderly father needs his medication")
    tenant = e.notifier.to("tenant")[0].body
    assert "Because you mentioned no water supply, we're treating this as an emergency" in tenant
    assert "emergency number" in tenant
    rfq = e.notifier.to("technician")[0]
    assert "(emergency)" in rfq.subject and "A vulnerable person" in rfq.body


def test_a_photo_emergency_is_not_credited_to_an_urgent_note(env):
    flood = {**DRAIN, "severity": "emergency", "is_safe_for_diy": False, "summary": "Burst pipe flooding the floor"}
    e = env(Fake(water=["no_water"]), flood)
    e.start("no water")
    tenant = e.notifier.to("tenant")[0].body
    assert "This looks like an emergency: Burst pipe flooding the floor" in tenant
    assert "Because you mentioned" not in tenant


def test_when_the_photo_ai_is_down_a_service_loss_is_still_urgent_for_a_person(env):
    e = env(Fake(water=["no_water"]), RuntimeError("quota exhausted"))
    e.start("no water")
    values = e.graph.get_state(e.config).values
    assert values["status"] == "manual_review" and values["note_urgency"] == "urgent"
    assert "SAME-DAY" in e.notifier.to("manager")[0].body
    tenant = e.notifier.to("tenant")[0].body
    assert "look at it today" in tenant and "isn't an emergency" in tenant


# --- the two new trades --------------------------------------------------------------------------


@pytest.mark.parametrize("trade, person", [
    ("plumber", "A plumber"), ("electrician", "An electrician"), ("hvac", "A heating and cooling technician"),
    ("pest", "A pest controller"), ("restoration", "A restoration specialist"),
    ("general", "A technician"), ("none", "A technician"),
])
def test_the_person_is_named_correctly(trade, person):
    assert _who(trade) == person


@pytest.mark.parametrize("trade", ["pest", "restoration"])
def test_the_new_trades_are_accepted_and_read_naturally_in_messages(env, trade):
    major = {**DRAIN, "severity": "major", "is_safe_for_diy": False, "trade": trade, "summary": "Needs attention"}
    VisualTriageAssessment(**major)                                   # the schema accepts it
    e = env(Fake(), major)
    e.start("see photo")
    rfq = e.notifier.to("technician")[0].body
    assert f"Trade: {'pest control' if trade == 'pest' else trade}." in rfq
    assert f"major {'pest control' if trade == 'pest' else trade} issue" in e.notifier.to("tenant")[0].body


def test_the_photo_prompt_describes_the_new_trades():
    from app.services.vision_service import TRIAGE_PROMPT

    assert '"pest" | "restoration"' in TRIAGE_PROMPT
    assert "rats, mice" in TRIAGE_PROMPT and "mould" in TRIAGE_PROMPT


# --- the AI prompt and its reply format -----------------------------------------------------------


def test_the_ai_prompt_defines_every_new_problem_and_its_limits():
    for key in ("no_water", "no_power", "no_heat", "no_lock", "no_toilet", "vulnerable"):
        assert f"- {key}:" in LLM_PROMPT
    assert "not just no hot water" in LLM_PROMPT and "not a loose handle" in LLM_PROMPT


def test_new_names_are_accepted_in_the_ai_reply():
    probs = parse_hazards('{"hazards": ["no_water", "vulnerable"]}')
    assert probs["no_water"] == 1.0 and probs["vulnerable"] == 1.0 and probs["gas"] == 0.0
