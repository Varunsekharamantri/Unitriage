import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from app.adapters.mock_notifier import MockNotifier
from app.core.schemas import TicketState
from app.core.security import MockVault, UnitSecrets
from app.graph.workflow import build_workflow
from app.services.vision_service import MockProvider, VisionService

IMG = "aGVsbG8="  # the mock provider never inspects the image
SECRET_CODE = "9137#"

DRAIN = {
    "severity": "minor", "confidence": 0.92, "trade": "plumber",
    "is_safe_for_diy": True, "summary": "Slow kitchen drain",
}
BURST = {
    "severity": "emergency", "confidence": 0.97, "trade": "plumber",
    "is_safe_for_diy": False, "summary": "Burst pipe flooding the floor",
}


class Env:
    """Wires a graph over a sqlite file; can be rebuilt to simulate a restart."""

    def __init__(self, tmp_path, responses):
        self.db = str(tmp_path / "cp.sqlite")
        self.responses = responses
        self.notifier = MockNotifier(echo=False)
        self.vault = MockVault({"vault/12B": UnitSecrets("12B", SECRET_CODE)})
        self.config = {"configurable": {"thread_id": "T-1"}}
        self.open()

    def open(self):
        self.conn = sqlite3.connect(self.db, check_same_thread=False)
        self.provider = MockProvider(self.responses)
        vision = VisionService([self.provider])
        self.graph = build_workflow(vision, self.notifier, self.vault, SqliteSaver(self.conn))

    def restart(self):
        self.conn.close()
        self.open()

    def start(self, description=""):
        state = TicketState(
            ticket_id="T-1", tenant_id="tenant-1", unit_id="12B",
            image_b64=IMG, vault_key="vault/12B", description=description,
        )
        self.graph.invoke(state, self.config)

    def resume(self, value):
        self.graph.invoke(Command(resume=value), self.config)

    @property
    def snapshot(self):
        return self.graph.get_state(self.config)

    @property
    def paused_step(self):
        tasks = self.snapshot.tasks
        return tasks[0].interrupts[0].value["step"] if tasks and tasks[0].interrupts else None


@pytest.fixture
def make_env(tmp_path):
    envs = []

    def factory(responses):
        env = Env(tmp_path, responses)
        envs.append(env)
        return env

    yield factory
    for e in envs:
        e.conn.close()


def test_minor_issue_takes_diy_path_and_waits_for_feedback(make_env):
    env = make_env([DRAIN])
    env.start()
    assert env.paused_step == "wait_for_feedback"
    assert env.snapshot.values["status"] == "awaiting_feedback"
    assert len(env.notifier.to("tenant")) == 1
    assert env.notifier.to("technician") == []


def test_diy_resolved_closes_the_ticket(make_env):
    env = make_env([DRAIN])
    env.start()
    env.resume({"resolved": True})
    assert env.snapshot.next == ()
    assert env.snapshot.values["status"] == "resolved"
    assert env.notifier.to("technician") == []
    assert len(env.notifier.to("tenant")) == 2  # guide + closing message


def test_diy_not_resolved_escalates_with_the_tenants_note(make_env):
    env = make_env([DRAIN])
    env.start()
    env.resume({"resolved": False, "note": "Tried the plunger, water still backs up"})

    assert env.paused_step == "wait_for_quote"
    rfq = env.notifier.to("technician")[0]
    assert "Tried the plunger" in rfq.body
    assert "DIY steps didn't solve it" in env.notifier.to("tenant")[-1].body
    # the rest of the escalation path works as normal from here
    env.resume({"quote": 120.0, "eta": "tomorrow"})
    env.resume("approved")
    assert env.snapshot.values["status"] == "dispatched"


def test_diy_is_optional_the_tenant_can_ask_for_a_technician_straight_away(make_env):
    env = make_env([DRAIN])
    env.start()
    guide = env.notifier.to("tenant")[0]
    # a technician is always on offer, listed first; and we don't promise a guide we don't have
    assert [a["kind"] for a in guide.actions] == ["technician", "fixed", "not_fixed"]
    assert all(len(a["label"]) <= 20 for a in guide.actions)  # WhatsApp's button-label limit
    assert "step-by-step" not in guide.body and "Ask a technician" in guide.body

    env.resume({"resolved": False, "want_technician": True})
    assert env.paused_step == "wait_for_quote"
    rfq = env.notifier.to("technician")[0]
    assert "chose to have a technician instead of trying the DIY steps" in rfq.body
    tenant_reply = env.notifier.to("tenant")[-1].body
    assert tenant_reply.startswith("No problem") and "didn't solve it" not in tenant_reply
    # and the rest of the escalation works as usual
    env.resume({"quote": 90.0, "eta": "tomorrow"})
    env.resume("approved")
    assert env.snapshot.values["status"] == "dispatched"


def test_asking_for_a_technician_can_carry_a_note(make_env):
    env = make_env([DRAIN])
    env.start()
    env.resume({"resolved": False, "want_technician": True, "note": "I'm away until Friday"})
    rfq = env.notifier.to("technician")[0].body
    assert "Tenant's note: I'm away until Friday" in rfq and "after trying the DIY steps" not in rfq


def test_feedback_survives_a_restart(make_env):
    env = make_env([DRAIN])
    env.start()
    env.restart()
    assert env.paused_step == "wait_for_feedback"
    env.resume({"resolved": True})
    assert env.snapshot.values["status"] == "resolved"


def test_tenant_description_reaches_the_ai_and_the_technician(make_env):
    env = make_env([BURST])
    provider = env.provider
    env.start(description="Water is coming out of the ceiling")
    assert provider.notes == ["Water is coming out of the ceiling"]
    assert "Water is coming out of the ceiling" in env.notifier.to("technician")[0].body


def test_emergency_advice_matches_the_trade(make_env):
    env = make_env([{**BURST, "trade": "electrician", "summary": "Sparking socket"}])
    env.start()
    body = env.notifier.to("tenant")[0].body
    assert "breaker" in body and "water valve" not in body


def test_major_but_not_emergency_gets_no_emergency_alarm(make_env):
    env = make_env([{**BURST, "severity": "major", "summary": "Steady leak under the sink"}])
    env.start()
    body = env.notifier.to("tenant")[0].body
    assert "emergency" not in body.lower() and "water valve" not in body
    assert "asked a technician for a quote" in body


def test_tenant_hears_first_then_technician_gets_the_enquiry_with_photo(make_env):
    env = make_env([BURST])
    env.start()
    assert [m.recipient for m in env.notifier.sent] == ["tenant", "technician"]
    assert env.notifier.to("tenant")[0].attachments == []
    photo = env.notifier.to("technician")[0].attachments
    assert len(photo) == 1 and photo[0]["kind"] == "image" and "/photo?token=" in photo[0]["url"]


def test_major_issue_pauses_at_gate_1(make_env):
    env = make_env([BURST])
    env.start()
    assert env.paused_step == "wait_for_quote"
    assert env.snapshot.next == ("wait_for_quote",)
    assert len(env.notifier.to("technician")) == 1


def test_quote_resumes_to_gate_2(make_env):
    env = make_env([BURST])
    env.start()
    env.resume({"quote": 175.0, "eta": "2 PM"})
    assert env.paused_step == "wait_for_approval"
    assert env.snapshot.values["quote"] == 175.0
    # the owner gets the technician's quote, then their own approval card (owner-only)
    assert [m.sender for m in env.notifier.to("manager")] == ["technician", "system"]
    # Resuming must not replay the RFQ side effect.
    assert len(env.notifier.to("technician")) == 1


def test_approval_dispatches_with_hydrated_secrets(make_env):
    env = make_env([BURST])
    env.start()
    env.resume({"quote": 175.0, "eta": "2 PM"})
    env.resume("approved")

    assert env.snapshot.next == ()
    assert env.snapshot.values["status"] == "dispatched"
    dispatch = env.notifier.to("technician")[-1]
    assert SECRET_CODE in dispatch.body and "Unit 12B" in dispatch.body
    assert "access_code" not in env.snapshot.values["dispatch_packet"]


def test_rejection_notifies_vendor_and_reroutes(make_env):
    env = make_env([BURST])
    env.start()
    env.resume({"quote": 900.0, "eta": "next week"})
    env.resume("rejected")

    assert env.snapshot.values["status"] == "rejected_rerouted"
    assert all(SECRET_CODE not in m.body for m in env.notifier.sent)
    assert "not approved" in env.notifier.to("technician")[-1].subject


def test_secret_never_reaches_checkpoint_database(make_env, tmp_path):
    env = make_env([BURST])
    env.start()
    env.resume({"quote": 175.0, "eta": "2 PM"})
    env.resume("approved")
    env.conn.commit()

    for f in tmp_path.iterdir():  # includes any -wal / -shm files
        assert SECRET_CODE.encode() not in f.read_bytes(), f.name


def test_pause_survives_process_restart(make_env):
    env = make_env([BURST])
    env.start()
    env.restart()  # fresh connection and graph; only the sqlite file remains
    assert env.paused_step == "wait_for_quote"

    env.resume({"quote": 175.0, "eta": "2 PM"})
    env.restart()
    assert env.paused_step == "wait_for_approval"

    env.resume("approved")
    assert env.snapshot.values["status"] == "dispatched"


def test_ambiguous_minor_issue_escalates_to_contractor(make_env):
    env = make_env([{**DRAIN, "confidence": 0.5}])
    env.start()
    assert env.paused_step == "wait_for_quote"


def test_non_diy_minor_issue_goes_to_contractor(make_env):
    env = make_env([{**DRAIN, "is_safe_for_diy": False}])
    env.start()
    assert env.paused_step == "wait_for_quote"


def test_triage_failure_goes_to_manual_review(make_env):
    env = make_env([RuntimeError("all providers down")])
    env.start()
    assert env.snapshot.values["status"] == "manual_review"
    assert env.snapshot.next == ()
    assert len(env.notifier.to("manager")) == 1
