"""Runtime guardrails around the chat assistant.

Two checks wrap every AI reply to a tenant:
- screen_input: a message that tries to override the assistant's rules, or asks for codes and
  credentials, is refused outright and never reaches an AI;
- check_reply: whatever an AI wrote is rejected, and replaced by a safe fixed reply, if it leaks a
  secret or its own instructions, contains personal data, links, prices or invented promises,
  gives unsafe do-it-yourself advice, or ignores a danger the tenant reported.

These are plain rules, so they are fast, predictable and testable (see tests/test_guardrails.py and
evals/run_chat_eval.py). Counters in STATS show how often each fired.
"""
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

from app.core.pii import redact

STATS: Counter = Counter()

_INJECTION = re.compile(
    r"ignore\s+(all\s+|any\s+|your\s+|the\s+|previous\s+|prior\s+|above\s+)*(instructions|rules|prompt|guidelines|polic\w+)"
    r"|disregard\s+.{0,25}(instructions|rules|prompt|guidelines)"
    r"|forget\s+(everything|all|your)\b"
    r"|(reveal|show|print|repeat|display|leak|tell me)\s+.{0,25}(system\s+prompt|your\s+(prompt|instructions|rules))"
    r"|\bsystem\s+prompt\b|\bdeveloper\s+mode\b|\bjailbreak|\bDAN\s+mode\b"
    r"|you\s+are\s+now\b|\bpretend\s+(to\s+be|you\s+are|you're)\b|\bact\s+as\s+(an?\s+)?(admin|developer|owner|system|root)\b"
    r"|\bnew\s+instructions?\s*:",
    re.I,
)
_SECRET_PROBE = re.compile(
    r"\b(access|door|gate|lock\s?box|key\s?box|entry|security)\s*(code|pin)\b|\b(passcode|password|passphrase|api\s*key|secret\s*key|token|vault)\b"
    r"|\bother\s+(tenants?|units?|residents?)\b|\bowner'?s\s+(phone|number|address|email)\b",
    re.I,
)
_DANGER = re.compile(
    r"\b(gas|fire|smoke|smoking|flames?|spark\w*|shock\w*|electrocut\w*|flood\w*|burst|exposed wire|live wire|collaps\w*)\b", re.I)

_LEAK = re.compile(r"system\s+prompt|my\s+(instructions|rules|guidelines)|as an ai (language )?model|i (was|am) (told|instructed|programmed)", re.I)
_UNSAFE_ADVICE = re.compile(
    r"touch\s+the\s+(wire|cable)|bare\s+hands?|re-?light|turn\s+(the\s+)?gas\s+(back\s+)?on|open\s+(up\s+)?the\s+(fuse|breaker|electrical)\s+(box|panel)"
    r"|bypass|remove\s+the\s+(cover|panel)|climb\s+(up|onto)|use\s+a\s+(lighter|match|candle)|try\s+to\s+(fix|repair)\s+(the\s+)?(wiring|gas|electric\w*)",
    re.I,
)
_STATUS_PROMISE = re.compile(
    r"\b(technician|plumber|electrician|engineer|someone|they)\b.{0,40}\b(will\s+(arrive|be\s+there|come)|is\s+(coming|on\s+the\s+way)|arrives?)\b", re.I)
_SAFETY_WORDS = re.compile(r"emergency|stay\s+safe|keep\s+clear|stay\s+away|leave|evacuat|call\s+(your|the)\s+(local\s+)?(emergency|fire|gas)", re.I)

REFUSAL = ("I can't help with that, and I can't share codes or personal details. "
           "I can help you report a maintenance problem: describe it or send a photo.")
SAFE_DANGER = ("Please stay safe and keep clear of it. If anyone is in danger, call your local emergency number. "
               "If it is safe, send a photo of the problem so I can assess it.")


@dataclass
class InputCheck:
    injection: bool
    secret_probe: bool
    danger: bool

    @property
    def refuse(self) -> bool:
        return self.injection or self.secret_probe


def screen_input(message: str) -> InputCheck:
    return InputCheck(bool(_INJECTION.search(message)), bool(_SECRET_PROBE.search(message)), bool(_DANGER.search(message)))


def check_reply(reply: str, message: str, secrets: Iterable[str] = ()) -> Optional[str]:
    """Why a reply must not be sent, or None when it is fine."""
    low = reply.lower()
    for s in secrets:
        if s and s.lower() in low:
            return "secret_leak"
    if _LEAK.search(reply):
        return "prompt_leak"
    if redact(reply) != reply:
        return "pii_in_reply"
    if re.search(r"https?://|www\.|[$€£₹]\s?\d|\b\d{1,2}[:.]\d{2}\s?(am|pm)?\b|\b\d{1,2}\s?(am|pm)\b", reply, re.I):
        return "link_price_or_time"
    if _UNSAFE_ADVICE.search(reply):
        return "unsafe_advice"
    if _STATUS_PROMISE.search(reply):
        return "invented_status"
    if _DANGER.search(message) and not _SAFETY_WORDS.search(reply):
        return "missing_safety"
    return None
