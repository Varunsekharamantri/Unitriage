import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.adapters.base import TECHNICIAN_CHAT, Action, Attachment, Notifier

logger = logging.getLogger(__name__)


@dataclass
class SentMessage:
    id: int
    ts: str
    sender: str
    recipient: str
    conversation: str
    subject: str
    body: str
    ticket_id: Optional[str] = None
    actions: List[Action] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)


class MockNotifier(Notifier):
    """Prints messages to stdout and keeps them, so tests and the demo UI can read them.

    In the demo it also doubles as the chat store: the API records the tenant's own
    messages here too, so every conversation can be replayed from one place.
    """

    def __init__(self, echo: bool = True, store=None):
        """store: optional object with load_messages() and add_message(dict), e.g. app.core.db.Database,
        so the conversation survives a restart."""
        self._store = store
        self.sent: List[SentMessage] = []
        if store is not None:
            self.sent = [
                SentMessage(**{**row, "actions": list(row["actions"]), "attachments": list(row["attachments"])})
                for row in store.load_messages()
            ]
        self._echo = echo
        self._lock = threading.Lock()

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
        if conversation is None:
            conversation = TECHNICIAN_CHAT if "technician" in (sender, recipient) else "tenant"
        with self._lock:
            msg = SentMessage(
                id=len(self.sent) + 1,
                ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                sender=sender,
                recipient=recipient,
                conversation=conversation,
                subject=subject,
                body=body,
                ticket_id=ticket_id,
                actions=list(actions or []),
                attachments=list(attachments or []),
            )
            self.sent.append(msg)
            if self._store is not None:
                try:
                    self._store.add_message(asdict(msg))
                except Exception:  # a storage hiccup must not stop a notification
                    logger.exception("could not save message %s", msg.id)
        if self._echo:
            line = f"\n[{sender.upper()} -> {recipient.upper()}] {subject}\n{body}\n"
            try:
                print(line)
            except UnicodeEncodeError:  # a console that can't show emoji must never break a ticket
                print(line.encode("ascii", "replace").decode("ascii"))

    def to(self, recipient: str) -> List[SentMessage]:
        return [m for m in self.sent if m.recipient == recipient]

    def since(
        self,
        after_id: int = 0,
        viewer: Optional[str] = None,
        conversation: Optional[str] = None,
    ) -> List[Dict]:
        """Messages newer than after_id as one participant's phone would receive them:
        only those the viewer sent or was sent, optionally within one conversation."""
        with self._lock:
            return [
                asdict(m)
                for m in self.sent
                if m.id > after_id
                and (viewer is None or viewer in (m.sender, m.recipient))
                and (conversation is None or m.conversation == conversation)
            ]
