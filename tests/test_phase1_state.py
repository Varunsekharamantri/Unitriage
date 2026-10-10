import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.core.schemas import TicketState, VisualTriageAssessment


def make_state() -> TicketState:
    assessment = VisualTriageAssessment(
        severity="major",
        confidence=0.9,
        trade="plumber",
        is_safe_for_diy=False,
        summary="Active pipe leak under sink",
    )
    return TicketState(
        ticket_id="T-1",
        tenant_id="tenant-1",
        unit_id="unit-12B",
        vault_key="vault/unit-12B",
        assessment=assessment.model_dump(),
        status="new",
    )


def test_state_conforms_to_ticket_state():
    state = make_state()
    assert set(state) <= set(TicketState.__annotations__)
    assert VisualTriageAssessment(**state["assessment"]).trade == "plumber"


def test_sqlite_checkpoint_roundtrip(tmp_path):
    def passthrough(state: TicketState) -> dict:
        return {"status": "triaged"}

    builder = StateGraph(TicketState)
    builder.add_node("passthrough", passthrough)
    builder.add_edge(START, "passthrough")
    builder.add_edge("passthrough", END)

    conn = sqlite3.connect(str(tmp_path / "cp.sqlite"), check_same_thread=False)
    graph = builder.compile(checkpointer=SqliteSaver(conn))
    config = {"configurable": {"thread_id": "T-1"}}

    graph.invoke(make_state(), config)

    saved = graph.get_state(config).values
    assert saved["ticket_id"] == "T-1"
    assert saved["status"] == "triaged"
    assert saved["assessment"]["severity"] == "major"
    conn.close()
