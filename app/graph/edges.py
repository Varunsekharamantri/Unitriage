"""Deterministic routing. The model only perceives; these functions decide."""
from app.core.schemas import TicketState


def route_after_triage(state: TicketState) -> str:
    assessment = state.get("assessment")
    if not assessment:
        return "manual_review"
    # A photo that is not about a home problem never becomes a ticket, unless the tenant's own words
    # report a danger (the note screener found hazards) or, for a spotless photo, describe the problem.
    subject = assessment.get("subject", "home_issue")
    if subject != "home_issue" and not state.get("note_flags"):
        if subject == "not_a_home" or not (state.get("description") or "").strip():
            return "not_relevant"
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
