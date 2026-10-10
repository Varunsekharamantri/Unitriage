"""Domain models: the AI's assessment, the dispatch packet, and graph state."""
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field

Severity = Literal["minor", "major", "emergency"]
Trade = Literal["plumber", "electrician", "hvac", "general", "pest", "restoration", "none"]


class Evidence(BaseModel):
    """What the vision model can SEE in the photo. It reports facts, never urgency:
    severity is derived from these by app.core.policy, so it can be tuned and tested."""

    water: Literal[
        "none", "stain_or_damp", "dripping", "steady_flow_or_spray", "flooding", "contained_in_fixture"
    ]
    damage: Literal["none", "cosmetic", "localized", "extensive"]
    electrical_hazard: bool
    gas_or_fire: bool
    structural_risk: bool
    sewage: bool
    simple_fix: bool  # a layperson could fix it with basic tools in under an hour
    # Is the tenant's note consistent with the photo? None when there was no note.
    note_matches_photo: Optional[bool] = None


class TriageObservation(BaseModel):
    """The vision model's raw output: what it sees, not what to do about it."""

    summary: str
    trade: Trade
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Evidence


class VisualTriageAssessment(BaseModel):
    """The triage result the rest of the system acts on."""

    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    trade: Trade
    is_safe_for_diy: bool
    summary: str
    ambiguous: bool = False
    # Present when severity was derived from evidence (always, in production).
    evidence: Optional[dict] = None
    reasons: list[str] = Field(default_factory=list)  # why the rules chose this severity
    # Dangers the tenant's own note reports that a photo can't show (a gas smell, sparks heard...).
    # They can only raise the severity, never lower it. See app.services.note_screen.
    note_flags: list[str] = Field(default_factory=list)
    # What the note alone means: "urgent" = same-day but not an emergency (no water, a lock that
    # won't lock...); "emergency" = a danger, or a service loss affecting someone vulnerable.
    note_urgency: Literal["none", "urgent", "emergency"] = "none"
    note_vulnerable: bool = False  # the note says a baby, elderly or sick person is affected


class DispatchPacket(BaseModel):
    """Final packet sent to the technician after manager approval.

    unit_number and access_code stay empty until the dispatch node hydrates them.
    """

    ticket_id: str
    trade: Trade
    summary: str
    quote: float
    eta: str
    unit_number: Optional[str] = None
    access_code: Optional[str] = None


class TicketState(TypedDict, total=False):
    """LangGraph state. Only plain, serializable values live here."""

    ticket_id: str
    tenant_id: str
    tenant_name: str  # display name from the intake channel; falls back to tenant_id
    unit_id: str
    image_b64: str
    image_mime: str
    description: str  # optional note the tenant typed with the photo
    feedback: Optional[dict]  # {"resolved": bool, "note": str} after a DIY guide
    note_flags: list  # problems reported in the tenant's note (also set when the photo AI is down)
    note_urgency: str  # "none", "urgent" or "emergency", from the note alone
    note_vulnerable: bool
    assessment: Optional[dict]  # VisualTriageAssessment.model_dump()
    quote: Optional[float]
    eta: Optional[str]
    approval: Optional[str]  # "approved" | "rejected"
    vault_key: str  # reference to the secret, never the secret itself
    dispatch_packet: Optional[dict]
    status: str
    error: Optional[str]
