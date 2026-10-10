"""Intake and resumption endpoints."""
import base64
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

from app import config
from app.adapters.base import tenant_chat
from app.core.schemas import TicketState
from app.core.security import photo_attachment, verify_token

router = APIRouter(prefix="/api")

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
HIDDEN_KEYS = {"image_b64", "vault_key"}


class QuoteIn(BaseModel):
    ticket_id: str
    quote: float = Field(gt=0)
    eta: str = Field(min_length=1)


class FeedbackIn(BaseModel):
    resolved: bool
    note: str = Field("", max_length=500)
    # The tenant would rather not do it themselves: send a technician now, without trying DIY.
    want_technician: bool = False

    @model_validator(mode="after")
    def _not_both(self):
        if self.resolved and self.want_technician:
            raise ValueError("a ticket can't be both resolved and in need of a technician")
        return self


def _config(ticket_id: str) -> dict:
    return {"configurable": {"thread_id": ticket_id}}


def _summary(workflow, ticket_id: str) -> dict:
    """Public view of a ticket. Never exposes the image or the vault reference."""
    snap = workflow.get_state(_config(ticket_id))
    if not snap.values:
        raise HTTPException(404, f"Unknown ticket {ticket_id!r}")
    paused = None
    if snap.tasks and snap.tasks[0].interrupts:
        paused = snap.tasks[0].interrupts[0].value["step"]
    values = {k: v for k, v in snap.values.items() if k not in HIDDEN_KEYS}
    return {"ticket_id": ticket_id, "awaiting": paused, **values}


def _require_gate(workflow, ticket_id: str, step: str) -> dict:
    """Raise 409 unless the ticket is paused at this step; returns the ticket summary."""
    ticket = _summary(workflow, ticket_id)
    current = ticket["awaiting"]
    if current != step:
        raise HTTPException(
            409, f"Ticket {ticket_id!r} is not waiting on {step!r} (currently: {current or 'not paused'})"
        )
    return ticket


@router.post("/tickets/intake", status_code=201)
def intake(
    request: Request,
    image: UploadFile = File(...),
    tenant_id: str = Form(...),
    unit_id: str = Form(...),
    description: str = Form("", max_length=500),
    tenant_name: str = Form("", max_length=100),
):
    workflow, vault = request.app.state.workflow, request.app.state.vault
    if image.content_type not in ALLOWED_MIME:
        raise HTTPException(415, f"Image must be one of {sorted(ALLOWED_MIME)}")
    data = image.file.read(config.MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(422, "Image is empty")
    if len(data) > config.MAX_IMAGE_BYTES:
        raise HTTPException(413, f"Image exceeds {config.MAX_IMAGE_BYTES} bytes")

    vault_key = f"vault/{unit_id}"
    try:
        vault.get(vault_key)
    except KeyError:
        raise HTTPException(422, f"Unknown unit {unit_id!r}") from None

    ticket_id = f"T-{uuid.uuid4().hex[:8]}"
    state = TicketState(
        ticket_id=ticket_id,
        tenant_id=tenant_id,
        tenant_name=tenant_name.strip() or tenant_id,
        unit_id=unit_id,
        image_b64=base64.b64encode(data).decode(),
        image_mime=image.content_type,
        description=description.strip(),
        vault_key=vault_key,
    )
    request.app.state.photos.put(ticket_id, data, image.content_type)
    # The tenant's own message goes into their chat with the owner first, so the owner is
    # notified straight away, before the AI has finished looking at the photo.
    request.app.state.notifier.send(
        "manager",
        "",
        state["description"],
        ticket_id=ticket_id,
        attachments=[photo_attachment(ticket_id)],
        sender="tenant",
        conversation=tenant_chat(tenant_id),
    )
    workflow.invoke(state, _config(ticket_id))
    return _summary(workflow, ticket_id)


@router.get("/usage")
def usage(request: Request):
    return request.app.state.usage.summary()


@router.get("/tickets/{ticket_id}")
def get_ticket(request: Request, ticket_id: str):
    return _summary(request.app.state.workflow, ticket_id)


@router.get("/tickets/{ticket_id}/photo")
def ticket_photo(request: Request, ticket_id: str, token: str = Query(...)):
    """The tenant's photo, for the technician and owner. Needs the signed link from the message."""
    # Check the token before looking the ticket up, so ids can't be probed for existence.
    if not verify_token(ticket_id, "photo", token):
        raise HTTPException(403, "Invalid photo token")
    cached = request.app.state.photos.get(ticket_id)
    if cached:
        content, media_type = cached
    else:  # e.g. after a restart: the photo is still in the ticket's checkpoint
        snap = request.app.state.workflow.get_state(_config(ticket_id))
        if not snap.values or not snap.values.get("image_b64"):
            raise HTTPException(404, f"Unknown ticket {ticket_id!r}")
        content = base64.b64decode(snap.values["image_b64"])
        media_type = snap.values.get("image_mime", "image/jpeg")
    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )


@router.post("/tickets/{ticket_id}/feedback")
def tenant_feedback(request: Request, ticket_id: str, body: FeedbackIn):
    """Tenant's reply to a DIY guide: resolved closes the ticket, otherwise it escalates."""
    workflow = request.app.state.workflow
    ticket = _require_gate(workflow, ticket_id, "wait_for_feedback")
    # What the tenant said, as a message in their chat with the owner.
    note = body.note.strip()
    request.app.state.notifier.send(
        "manager",
        "",
        "✅ It's fixed, thanks!" if body.resolved
        else note or ("🔧 Please send a technician" if body.want_technician else "❌ Still not fixed"),
        ticket_id=ticket_id,
        sender="tenant",
        conversation=tenant_chat(ticket["tenant_id"]),
    )
    workflow.invoke(
        Command(resume={
            "resolved": body.resolved,
            "note": body.note.strip(),
            "want_technician": body.want_technician,
        }),
        _config(ticket_id),
    )
    return _summary(workflow, ticket_id)


@router.post("/webhooks/technician-quote")
def technician_quote(request: Request, body: QuoteIn):
    workflow = request.app.state.workflow
    _require_gate(workflow, body.ticket_id, "wait_for_quote")
    workflow.invoke(
        Command(resume={"quote": body.quote, "eta": body.eta}), _config(body.ticket_id)
    )
    return _summary(workflow, body.ticket_id)


@router.get("/webhooks/manager-approval")
def manager_approval(
    request: Request,
    ticket_id: str = Query(...),
    decision: Literal["approved", "rejected"] = Query(...),
    token: str = Query(...),
):
    workflow = request.app.state.workflow
    if not verify_token(ticket_id, decision, token):
        raise HTTPException(403, "Invalid approval token")
    _require_gate(workflow, ticket_id, "wait_for_approval")  # also makes links single-use
    workflow.invoke(Command(resume=decision), _config(ticket_id))
    return _summary(workflow, ticket_id)
