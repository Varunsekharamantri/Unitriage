"""Messaging abstraction. Swap in Telegram, email or SMS by implementing Notifier."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# An action is a button a chat UI can render, e.g.
#   {"kind": "approve", "label": "Approve", "url": "..."}   (one-click link)
#   {"kind": "quote", "label": "Reply with quote"}           (opens a quote form)
# Channels without buttons (SMS, plain text) can ignore actions: links stay in the body.
Action = Dict[str, Any]

# An attachment is media to show with the message, e.g. {"kind": "image", "url": "..."}.
Attachment = Dict[str, Any]

# Every message belongs to one conversation between two parties, and the property owner
# (role "manager") is always one of them:
#   tenant:<tenant_id>  tenant <-> owner       technician  owner <-> technician
# The AI assistant speaks as the owner, so its messages have sender "manager". Sender
# "system" is for notices that only the recipient may see (e.g. the approval card).
TECHNICIAN_CHAT = "technician"


def tenant_chat(tenant_id: str) -> str:
    return f"tenant:{tenant_id}"


class Notifier(ABC):
    @abstractmethod
    def send(
        self,
        recipient: str,
        subject: str,
        body: str,
        ticket_id: Optional[str] = None,
        actions: Optional[List[Action]] = None,
        attachments: Optional[List[Attachment]] = None,
        sender: str = "manager",
        conversation: Optional[str] = None,
    ) -> None:
        """Deliver a message. Roles are: tenant, technician, manager (the property owner),
        and system for sender only."""
