"""Deterministic routing. The model only perceives; these functions decide."""
from app.core.schemas import TicketState


def route_after_triage(state: TicketState) -> str:
    assessment = state.get("assessment")
    if not assessment:
        return "manual_review"
    # Only a confident, minor, DIY-safe ticket skips the contractor path.
    if (
        assessment["severity"] == "minor"
        and assessment["is_safe_for_diy"]
        and not assessment["ambiguous"]
    ):
        return "diy"
    return "contractor"


def route_after_feedback(state: TicketState) -> str:
    return "close" if (state.get("feedback") or {}).get("resolved") else "escalate"


def route_after_approval(state: TicketState) -> str:
    return "dispatch" if state.get("approval") == "approved" else "reroute"
