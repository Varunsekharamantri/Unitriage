"""StateGraph assembly."""
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app import config
from app.adapters.base import Notifier
from app.core.schemas import TicketState
from app.core.security import MockVault
from app.graph.edges import route_after_approval, route_after_feedback, route_after_triage
from app.graph.nodes import TriageNodes
from app.services.vision_service import VisionService


def get_sqlite_checkpointer(path: str = config.CHECKPOINT_DB_PATH) -> SqliteSaver:
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def build_workflow(
    vision: VisionService,
    notifier: Notifier,
    vault: MockVault,
    checkpointer: SqliteSaver,
    summarizer=None,
    clear_image: bool = False,
    retry_window: float = 0.0,
):
    n = TriageNodes(vision, notifier, vault, summarizer, clear_image, retry_window)
    g = StateGraph(TicketState)

    g.add_node("triage", n.triage)
    g.add_node("manual_review", n.manual_review)
    g.add_node("dispatch_diy_guide", n.dispatch_diy_guide)
    g.add_node("await_feedback", n.await_feedback)
    g.add_node("close_ticket", n.close_ticket)
    g.add_node("request_technician_rfq", n.request_technician_rfq)
    g.add_node("wait_for_quote", n.wait_for_quote)
    g.add_node("prepare_pm_approval", n.prepare_pm_approval)
    g.add_node("wait_for_pm_approval", n.wait_for_pm_approval)
    g.add_node("notify_vendor_and_reroute", n.notify_vendor_and_reroute)
    g.add_node("dispatch_packet_assembler", n.dispatch_packet_assembler)

    g.add_edge(START, "triage")
    g.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "diy": "dispatch_diy_guide",
            "contractor": "request_technician_rfq",
            "manual_review": "manual_review",
        },
    )
    g.add_edge("manual_review", END)

    g.add_edge("dispatch_diy_guide", "await_feedback")
    g.add_conditional_edges(
        "await_feedback",
        route_after_feedback,
        {"close": "close_ticket", "escalate": "request_technician_rfq"},
    )
    g.add_edge("close_ticket", END)

    g.add_edge("request_technician_rfq", "wait_for_quote")
    g.add_edge("wait_for_quote", "prepare_pm_approval")
    g.add_edge("prepare_pm_approval", "wait_for_pm_approval")
    g.add_conditional_edges(
        "wait_for_pm_approval",
        route_after_approval,
        {"dispatch": "dispatch_packet_assembler", "reroute": "notify_vendor_and_reroute"},
    )
    g.add_edge("notify_vendor_and_reroute", END)
    g.add_edge("dispatch_packet_assembler", END)

    return g.compile(checkpointer=checkpointer)
