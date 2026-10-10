"""Mock vault for property secrets.

Graph state only ever holds a `vault_key` reference. The unit number and access
code are fetched here, in the dispatch node, after explicit manager approval, and
are handed to the notifier without being written back into checkpointed state.
"""
import hashlib
import hmac
from dataclasses import dataclass
from typing import Dict

from app import config


def sign_token(ticket_id: str, action: str) -> str:
    """HMAC token for a one-click link; binds the link to one ticket and one action."""
    msg = f"{ticket_id}:{action}".encode()
    return hmac.new(config.VAULT_SECRET.encode(), msg, hashlib.sha256).hexdigest()


def verify_token(ticket_id: str, action: str, token: str) -> bool:
    return hmac.compare_digest(sign_token(ticket_id, action), token)


def photo_attachment(ticket_id: str) -> dict:
    """Signed link to the tenant's photo, served by GET /api/tickets/{id}/photo."""
    url = f"{config.PUBLIC_BASE_URL}/api/tickets/{ticket_id}/photo?token={sign_token(ticket_id, 'photo')}"
    return {"kind": "image", "url": url}


@dataclass(frozen=True)
class UnitSecrets:
    unit_number: str
    access_code: str


class MockVault:
    """In-memory stand-in for an encrypted secrets store (KMS, Vault, etc.)."""

    def __init__(self, secrets: Dict[str, UnitSecrets] | None = None):
        self._secrets = dict(secrets or {})

    def put(self, vault_key: str, secrets: UnitSecrets) -> None:
        self._secrets[vault_key] = secrets

    def get(self, vault_key: str) -> UnitSecrets:
        try:
            return self._secrets[vault_key]
        except KeyError:
            raise KeyError(f"No secrets stored for vault key {vault_key!r}") from None
