"""Graph nodes.

Each interrupt() lives in its own node with no side effects before it: LangGraph
re-runs a paused node from the top on resume, so anything before interrupt()
would fire twice.

Messages: the assistant speaks as the property owner (sender "manager"), so the tenant and
technician see the owner's name and the owner sees everything in both conversations. Notices
only the owner should see (the approval card with its links) use sender "system".
"""
import time

from langgraph.types import interrupt

from app import config
from app.adapters.base import TECHNICIAN_CHAT, Notifier, tenant_chat
from app.core.schemas import DispatchPacket, TicketState
from app.core.security import MockVault, photo_attachment, sign_token
from app.services.note_screen import PRIORITY, describe
from app.services.vision_service import TriageError, VisionService


_TRADE_LABEL = {"hvac": "heating and cooling", "pest": "pest control", "none": "general"}
_TRADE_PERSON = {
    "plumber": "A plumber",
    "electrician": "An electrician",
    "hvac": "A heating and cooling technician",
    "pest": "A pest controller",
    "restoration": "A restoration specialist",
}


def _trade(trade: str) -> str:
    """The trade as it reads in a message: 'heating and cooling', 'pest control', 'general'..."""
    return _TRADE_LABEL.get(trade, trade)


def _issue(a: dict) -> str:
    """'major plumber issue', or 'minor issue' when the model named no trade."""
    return f"{a['severity']} {_trade(a['trade'])} issue" if a["trade"] != "none" else f"{a['severity']} issue"


def _who(trade: str) -> str:
    """'A plumber', 'An electrician', or 'A technician' when no specific trade applies."""
    return _TRADE_PERSON.get(trade, "A technician")


def _summary(a: dict) -> str:
    return a["summary"].strip().rstrip(".")


# Canned first-aid advice for emergencies, by trade. Deliberately not model-written:
# safety guidance should be predictable and reviewed by a person.
_SAFETY = {
    "plumber": "If you can do it safely, turn off the main water valve and move belongings away from the water.",
    "electrician": "If it is safe to reach, switch off the power at the breaker, and don't touch exposed wiring or wet fittings.",
    "hvac": "Turn the unit off at its switch. If you smell gas, leave the area and call your gas provider's emergency line.",
}
_SAFETY_DEFAULT = "Please keep clear of the affected area until the technician arrives."

# Advice for dangers the tenant reports in their note, most urgent first (see note_screen.PRIORITY).
_FLAG_ADVICE = {
    "danger": "If anyone is hurt or in danger, call your local emergency number now.",
    "gas": "Don't use switches, lighters or flames. Open the windows, leave the building, and call "
           "your gas provider's emergency line or your local emergency number from outside.",
    "fire_smoke": "If there is fire or smoke, leave the building now and call your local emergency "
                  "number. Don't go back inside for belongings.",
    "electrical": "Stay away from the socket, switch or wiring and don't touch it. If it is safe to "
                  "reach, switch the power off at the breaker.",
    "collapse": "Keep everyone out from under the area, and don't try to prop it up yourself.",
    "flooding": _SAFETY["plumber"],
    # Service losses are urgent, not emergencies: practical advice, no alarm.
    "no_heat": "Wrap up warmly and stay in one warm room if you can. Don't use a gas cooker, barbecue "
               "or open flame to heat the home.",
    "no_water": "If you can, keep some drinking water aside while we sort this out.",
    "no_power": "Keep the fridge and freezer closed and unplug sensitive appliances until the power is "
                "back. Use a torch rather than candles.",
    "no_lock": "If you feel unsafe or there has been a break-in, call your local emergency number. "
               "Keep your valuables with you for now.",
    "no_toilet": "Please use another toilet if you can while we arrange the repair.",
}


def _flag_advice(flags) -> str:
    """The advice for the most urgent reported danger."""
    for key in PRIORITY:
        if key in flags:
            return _FLAG_ADVICE[key]
    return _SAFETY_DEFAULT


def _tenant(state: TicketState) -> str:
    return state.get("tenant_name") or state["tenant_id"]


class TriageNodes:
    def __init__(
        self, vision: VisionService, notifier: Notifier, vault: MockVault, summarizer=None, clear_image: bool = False,
        retry_window: float = 0.0, retry_pause: float = 12.0,
    ):
        # retry_window: seconds to keep retrying while the AI providers are busy. 0 = try once.
        self._retry_window = retry_window
        self._retry_pause = retry_pause
        # clear_image: the photo is saved elsewhere (the database), so don't carry it in every
        # checkpoint of the ticket after triage has looked at it.
        self._clear_image = clear_image
        self._summarizer = summarizer  # writes the owner's plain-language summary; None = template
        self._vision = vision
        self._notifier = notifier
        self._vault = vault

    def _send(
        self,
        to: str,
        subject: str,
        body: str,
        state: TicketState,
        *,
        sender: str = "manager",
        chat: str | None = None,
        actions=None,
        attachments=None,
    ):
        if chat is None:
            chat = TECHNICIAN_CHAT if "technician" in (to, sender) else tenant_chat(state["tenant_id"])
        self._notifier.send(
            to,
            subject,
            body,
            ticket_id=state["ticket_id"],
            actions=actions,
            attachments=attachments,
            sender=sender,
            conversation=chat,
        )

    # --- triage ---------------------------------------------------------
    def triage(self, state: TicketState) -> dict:
        deadline = time.monotonic() + self._retry_window
        try:
            while True:
                try:
                    assessment = self._vision.triage(
                        state["image_b64"],
                        state.get("image_mime", "image/jpeg"),
                        state.get("description", ""),
                    )
                    break
                except TriageError as exc:
                    # Busy or rate-limited providers usually recover within a minute: wait and try
                    # again instead of handing the ticket to a person straight away.
                    if not exc.retryable or time.monotonic() + self._retry_pause > deadline:
                        raise
                    time.sleep(self._retry_pause)
        except TriageError as exc:
            # The photo AI is out, but the tenant's own words can still raise the alarm.
            found = self._vision.screen_note(state.get("description", ""))
            return {
                "assessment": None,
                "status": "triage_failed",
                "error": str(exc),
                "note_flags": found.flags,
                "note_urgency": found.urgency,
                "note_vulnerable": found.vulnerable,
                **self._image_update(),
            }
        return {
            "assessment": assessment.model_dump(),
            "status": "triaged",
            "note_flags": assessment.note_flags,
            "note_urgency": assessment.note_urgency,
            "note_vulnerable": assessment.note_vulnerable,
            **self._image_update(),
        }

    def _image_update(self) -> dict:
        """Empty the photo out of the ticket state when it is kept in the database instead."""
        return {"image_b64": ""} if self._clear_image else {}

    def manual_review(self, state: TicketState) -> dict:
        flags = state.get("note_flags") or []
        urgency = state.get("note_urgency", "none")
        label = {"emergency": "URGENT", "urgent": "SAME-DAY"}.get(urgency, "")
        who = " A vulnerable person is affected." if state.get("note_vulnerable") else ""
        urgent = (
            f"\n⚠️ {label}: the tenant's note reports {describe(flags)}.{who} Please look at this first."
            if flags else ""
        )
        self._send(
            "manager",
            f"Ticket {state['ticket_id']} needs manual review",
            f"Automatic photo triage failed. Please review the ticket by hand.{urgent}",
            state,
            sender="system",
        )
        if flags and urgency == "emergency":
            tenant_text = (
                f"We couldn't analyse your photo automatically, but because you mentioned "
                f"{describe(flags)}, we're treating this as urgent. {_flag_advice(flags)} "
                "If anyone is in danger, call your local emergency number. "
                "A person is looking at it right now."
            )
        elif flags:  # a service loss: urgent, not an emergency
            tenant_text = (
                f"We couldn't analyse your photo automatically, but because you mentioned "
                f"{describe(flags)}, we're treating this as urgent and a person will look at it today. "
                f"{_flag_advice(flags)} This isn't an emergency, but if anyone is in danger, "
                "call your local emergency number."
            )
        else:
            tenant_text = "We couldn't analyse your photo automatically, so a person will review it shortly."
        self._send("tenant", f"Ticket {state['ticket_id']}: under review", tenant_text, state)
        return {"status": "manual_review"}

    # --- DIY path -------------------------------------------------------
    def dispatch_diy_guide(self, state: TicketState) -> dict:
        a = state["assessment"]
        self._send(
            "tenant",
            f"Ticket {state['ticket_id']}: you can fix this yourself",
            f"We think this is a {_issue(a)}: {_summary(a)}. "
            "It may be something you can fix yourself, but that's entirely up to you: "
            "tap 'Ask a technician' at any time and we'll arrange one.",
            state,
            # Doing it yourself is always optional: the first choice is to hand it to a technician.
            actions=[
                {"kind": "technician", "label": "🔧 Ask a technician"},
                {"kind": "fixed", "label": "✅ It's fixed"},
                {"kind": "not_fixed", "label": "❌ Still not fixed"},
            ],
        )
        return {"status": "awaiting_feedback"}

    def await_feedback(self, state: TicketState) -> dict:
        reply = interrupt({"step": "wait_for_feedback", "ticket_id": state["ticket_id"]})
        return {
            "feedback": {
                "resolved": bool(reply.get("resolved")),
                "note": str(reply.get("note", ""))[:500],
                # True when the tenant skipped the DIY steps and asked for a technician straight away.
                "want_technician": bool(reply.get("want_technician")),
            }
        }

    def close_ticket(self, state: TicketState) -> dict:
        self._send(
            "tenant",
            f"Ticket {state['ticket_id']}: resolved",
            "Glad that fixed it! We've closed this ticket. Message us any time if it comes back.",
            state,
        )
        return {"status": "resolved"}

    # --- contractor path --------------------------------------------------
    def request_technician_rfq(self, state: TicketState) -> dict:
        a = state["assessment"]
        feedback = state.get("feedback") or {}
        skipped_diy = bool(feedback.get("want_technician"))
        # Words the tenant typed (with the photo, or after the DIY guide) go to the technician.
        notes = "".join(
            f"\n{label}: {text}"
            for label, text in (
                ("Tenant's description", state.get("description")),
                ("Tenant's note" if skipped_diy else "Tenant after trying the DIY steps", feedback.get("note")),
            )
            if text
        )
        if skipped_diy:
            notes += "\nThe tenant chose to have a technician instead of trying the DIY steps."
        same_day = a.get("note_urgency") == "urgent"
        if a.get("note_flags"):
            tag = "SAME-DAY: " if same_day else ""
            notes += f"\n⚠️ {tag}The tenant reports {describe(a['note_flags'])} (not verified from the photo)."
            if a.get("note_vulnerable"):
                notes += " A vulnerable person (baby, elderly or medical need) is affected."
        # The tenant hears first. What they are told depends on how serious the issue is.
        if skipped_diy:
            tenant_text = (
                "No problem. We've asked a technician for a quote and will keep you updated."
            )
        elif feedback:
            tenant_text = (
                "Sorry the DIY steps didn't solve it. We've asked a technician for a quote "
                "and will keep you updated."
            )
        elif a["severity"] == "emergency":
            # Only when the NOTE made it an emergency does the message credit the note.
            reported = (a.get("note_flags") or []) if a.get("note_urgency") == "emergency" else []
            if reported:  # the tenant told us about a danger the photo can't show
                lead = f"Because you mentioned {describe(reported)}, we're treating this as an emergency."
                safety = _flag_advice(reported)
            else:
                lead = f"This looks like an emergency: {_summary(a)}."
                safety = _SAFETY.get(a["trade"], _SAFETY_DEFAULT)
            tenant_text = (
                f"{lead} {safety} "
                "If anyone is in danger, call your local emergency number. "
                "We're contacting a technician right now and will update you here."
            )
        elif a.get("note_urgency") == "urgent":
            # A service loss (no water, a lock that won't lock...): same day, but not an emergency.
            tenant_text = (
                f"Because you mentioned {describe(a['note_flags'])}, we're treating this as urgent. "
                f"{_flag_advice(a['note_flags'])} We've asked a technician to come today and will keep "
                "you updated. This isn't an emergency, but if anyone is in danger, call your local "
                "emergency number."
            )
        else:
            caution = (
                " We weren't fully sure from the photo, so we're being cautious."
                if a["ambiguous"]
                else ""
            )
            tenant_text = (
                f"We assessed this as a {_issue(a)}: {_summary(a)}.{caution} "
                "We've asked a technician for a quote and will keep you updated."
            )
        self._send(
            "tenant", f"Ticket {state['ticket_id']}: technician requested", tenant_text, state
        )
        # In the background, the technician gets an enquiry with the tenant's photo.
        # Deliberately no unit number, tenant name or access code at this stage.
        self._send(
            "technician",
            f"Quote request for ticket {state['ticket_id']} ({a['severity']}{', SAME DAY' if same_day else ''})",
            f"Trade: {_trade(a['trade'])}. Issue: {_summary(a)}.{notes}\n"
            "Please reply with a quote and ETA.",
            state,
            actions=[{"kind": "quote", "label": "Reply with quote"}],
            attachments=[photo_attachment(state["ticket_id"])],
        )
        return {"status": "rfq_sent"}

    def wait_for_quote(self, state: TicketState) -> dict:
        reply = interrupt({"step": "wait_for_quote", "ticket_id": state["ticket_id"]})
        return {
            "quote": float(reply["quote"]),
            "eta": str(reply["eta"]),
            "status": "quote_received",
        }

    def _owner_summary(self, state: TicketState) -> str:
        """The paragraph on the approval card. AI-written when a summarizer is set, and the
        fixed template otherwise (or if the AI is down). The quote and ETA lines are never
        AI-written: the app adds them itself."""
        a = state["assessment"]
        template = f"{_issue(a).capitalize()}: {_summary(a)}."
        if self._summarizer is None:
            return template
        facts = {
            "severity": a["severity"],
            "trade": _trade(a["trade"]),
            "what_the_photo_shows": _summary(a),
            "tenant_note": state.get("description", ""),
            "note_reports": describe(a.get("note_flags") or []),
            "vulnerable_person": bool(a.get("note_vulnerable")),
            "quote_usd": f"{state['quote']:.2f}",
            "eta": state["eta"],
            "unit": state["unit_id"],
        }
        return self._summarizer.write(facts) or template

    def prepare_pm_approval(self, state: TicketState) -> dict:
        a = state["assessment"]
        tid = state["ticket_id"]

        def link(decision: str) -> str:
            return (
                f"{config.PUBLIC_BASE_URL}/api/webhooks/manager-approval"
                f"?ticket_id={tid}&decision={decision}&token={sign_token(tid, decision)}"
            )

        # The technician's quote, as both of them see it. It must never carry the approval
        # links, or the technician could approve their own quote.
        self._send(
            "manager",
            f"Quote for ticket {tid}",
            f"Quote: ${state['quote']:.2f} · ETA {state['eta']}",
            state,
            sender="technician",
        )
        # The owner's card, visible to the owner only, with the tenant context and the buttons.
        self._send(
            "manager",
            f"Approval needed: ticket {tid}",
            f"Tenant: {_tenant(state)} · Unit {state['unit_id']}\n"
            f"{self._owner_summary(state)}\n"
            f"Quote: ${state['quote']:.2f}, ETA {state['eta']}.\n"
            f"Approve: {link('approved')}\n"
            f"Reject:  {link('rejected')}",
            state,
            sender="system",
            chat=TECHNICIAN_CHAT,
            actions=[
                {"kind": "approve", "label": "Approve", "url": link("approved")},
                {"kind": "reject", "label": "Reject", "url": link("rejected")},
            ],
        )
        return {"status": "awaiting_approval"}

    def wait_for_pm_approval(self, state: TicketState) -> dict:
        decision = interrupt(
            {
                "step": "wait_for_approval",
                "ticket_id": state["ticket_id"],
                "quote": state["quote"],
                "eta": state["eta"],
            }
        )
        return {"approval": "approved" if decision == "approved" else "rejected"}

    def notify_vendor_and_reroute(self, state: TicketState) -> dict:
        self._send(
            "technician",
            f"Ticket {state['ticket_id']}: quote not approved",
            "❌ The property owner declined this quote. No dispatch will happen.",
            state,
        )
        self._send(
            "manager",
            f"Ticket {state['ticket_id']} needs a new vendor",
            "The quote was rejected. Please choose another technician.",
            state,
            sender="system",
            chat=TECHNICIAN_CHAT,
        )
        self._send(
            "tenant",
            f"Ticket {state['ticket_id']}: finding another technician",
            "The first quote wasn't approved, so we're arranging a different technician.",
            state,
        )
        return {"status": "rejected_rerouted"}

    def dispatch_packet_assembler(self, state: TicketState) -> dict:
        a = state["assessment"]
        secrets = self._vault.get(state["vault_key"])
        packet = DispatchPacket(
            ticket_id=state["ticket_id"],
            trade=a["trade"],
            summary=a["summary"],
            quote=state["quote"],
            eta=state["eta"],
            unit_number=secrets.unit_number,
            access_code=secrets.access_code,
        )
        self._send(
            "technician",
            f"Quote approved: ticket {packet.ticket_id}",
            "✅ The property owner approved your quote.",
            state,
        )
        self._send(
            "technician",
            f"DISPATCH: ticket {packet.ticket_id}",
            f"Unit {packet.unit_number} (tenant: {_tenant(state)}), access code {packet.access_code}. "
            f"{_trade(packet.trade).capitalize()}: {_summary(a)}. Agreed ${packet.quote:.2f}, ETA {packet.eta}.",
            state,
        )
        # The tenant is told who is coming, never the access code or the price.
        self._send(
            "tenant",
            f"Ticket {packet.ticket_id}: technician on the way",
            f"Good news: the property owner approved the repair. "
            f"{_who(packet.trade)} is on the way, arriving {packet.eta}.",
            state,
        )
        # Secrets go to the technician only; the checkpointed state keeps a redacted copy.
        redacted = packet.model_dump(exclude={"unit_number", "access_code"})
        return {"dispatch_packet": redacted, "status": "dispatched"}
