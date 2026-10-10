"""Endpoints that exist only for the chat simulator (enabled by DEMO_UI)."""
from typing import Literal, Optional

from fastapi import APIRouter, Query, Request

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
