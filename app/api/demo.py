"""Endpoints that exist only for the chat simulator (enabled by DEMO_UI)."""
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from app import config

router = APIRouter(prefix="/api/demo")


@router.get("/config")
def demo_config():
    return {
        "owner_name": config.OWNER_NAME,
        "technician_name": config.TECHNICIAN_NAME,
        "tenants": config.DEMO_TENANTS,
    }


@router.get("/messages")
def messages(
    request: Request,
    after: int = Query(0, ge=0),
    viewer: Optional[Literal["tenant", "technician", "manager"]] = None,
    conversation: Optional[str] = None,
):
    """Messages with id > after, as one participant's phone receives them.

    viewer limits the feed to messages that person sent or was sent (the property owner,
    "manager", is a party to every conversation); conversation narrows it to one chat.
    """
    return request.app.state.notifier.since(after, viewer, conversation)


class CallAction(BaseModel):
    action: Literal["start", "answer", "decline", "end"]
    unit: Optional[str] = None
    name: Optional[str] = None


IDLE = {"id": 0, "state": "idle", "unit": None, "name": None}


@router.get("/call")
def get_call(request: Request):
    """The one tenant-to-owner voice call (the demo has a single tenant calling at a time).

    state: idle | ringing | active | declined | ended. The owner's phone polls this to show an
    incoming call; the tenant's phone polls it to see whether the owner declined."""
    return getattr(request.app.state, "call", IDLE)


@router.post("/call")
def post_call(body: CallAction, request: Request):
    call = getattr(request.app.state, "call", IDLE)
    if body.action == "start":
        call = {"id": call["id"] + 1, "state": "ringing", "unit": body.unit, "name": body.name or "Tenant"}
    elif call["state"] == "idle":
        raise HTTPException(409, "No call in progress")
    elif body.action == "answer" and call["state"] == "ringing":
        call = {**call, "state": "active"}
    elif body.action == "decline" and call["state"] == "ringing":
        call = {**call, "state": "declined"}
    elif body.action == "end" and call["state"] in ("ringing", "active"):
        call = {**call, "state": "ended"}
    request.app.state.call = call
    return call
