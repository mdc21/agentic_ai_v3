"""
contact_history.py — writes a contact history record to SoR_1 on session close.

Called as the final action before every session terminates (resolved or escalated).
A write failure is logged in the audit record but does not alter the session outcome.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

SOR1_BASE = os.getenv("SOR1_API_BASE_URL", "https://sor1.example.com/api/v1")


# ── Payload ────────────────────────────────────────────────────────────────────

@dataclass
class ContactHistoryPayload:
    policy_number:        str
    contact_date:         str
    contact_timestamp:    str
    channel:              str
    session_id:           str
    session_outcome:      str           # resolved | escalated
    escalation_reason:    Optional[str]
    caller_type:          Optional[str]
    verification_passed:  bool
    call_intent:          Optional[str]
    intents_served:       list
    conversation_summary: str
    transcript:           list

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}


# ── Builder ────────────────────────────────────────────────────────────────────

def build_contact_history(ctx) -> ContactHistoryPayload:
    """Build a ContactHistoryPayload from a ConversationContext."""
    now = datetime.now(timezone.utc)

    # Collect the list of action_intents that were actually executed this session
    intents_served = _extract_intents_from_history(ctx.turn_history)

    # Build a concise human-readable summary
    summary_parts = []
    if ctx.policy_number:
        summary_parts.append(f"Policy {ctx.policy_number}.")
    if ctx.policy_record:
        summary_parts.append(
            f"Found on {getattr(ctx, 'heritage_brand', 'SoR_1')} "
            f"({getattr(ctx, 'product_type', 'unknown product type')})."
        )
    if ctx.caller_type:
        summary_parts.append(f"Caller identified as {ctx.caller_type.value}.")
    if ctx.verification_result:
        outcome = "passed" if ctx.verification_result.verified else "failed"
        summary_parts.append(f"Identity verification {outcome}.")
    if ctx.call_intent:
        summary_parts.append(f"Call intent: {ctx.call_intent}.")
    if ctx.state.value == "escalated":
        summary_parts.append(f"Escalated — reason: {ctx.escalation_reason}.")
    else:
        summary_parts.append("Session resolved successfully.")

    return ContactHistoryPayload(
        policy_number=ctx.policy_number or "UNKNOWN",
        contact_date=now.strftime("%Y-%m-%d"),
        contact_timestamp=now.isoformat(),
        channel=ctx.channel,
        session_id=ctx.session_id,
        session_outcome=ctx.state.value,
        escalation_reason=ctx.escalation_reason,
        caller_type=ctx.caller_type.value if ctx.caller_type else None,
        verification_passed=(ctx.verification_result.verified
                             if ctx.verification_result else False),
        call_intent=ctx.call_intent,
        intents_served=intents_served,
        conversation_summary=" ".join(summary_parts),
        transcript=list(ctx.turn_history),
    )


def _extract_intents_from_history(turn_history: list) -> list[str]:
    """
    Best-effort extraction of action_intents from the assistant turns
    in the turn history (assistant turns are stored as JSON strings by the LLM).
    Falls back gracefully if turns are plain text.
    """
    intents = []
    for turn in turn_history:
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "")
        try:
            data = json.loads(content)
            intent = data.get("action_intent") or data.get("action")
            if intent and intent not in intents:
                intents.append(intent)
        except (json.JSONDecodeError, TypeError):
            pass
    return intents


# ── Writer ─────────────────────────────────────────────────────────────────────

class ContactHistoryClient:
    def __init__(self) -> None:
        self._mock = os.getenv("USE_MOCK_POLICY_API", "false").lower() == "true"
        if not self._mock:
            self._http = httpx.Client(base_url=SOR1_BASE, timeout=15.0)

    def write(self, payload: ContactHistoryPayload, session_id: str = "") -> bool:
        """
        POST the contact history record to SoR_1.
        Returns True on success. A failure is logged but never raises.
        """
        if self._mock:
            logger.info("[%s] Contact history (mock write): %s",
                        session_id, payload.conversation_summary[:80])
            return True

        try:
            r = self._http.post(
                f"/policies/{payload.policy_number}/contact-history",
                json=payload.to_dict(),
            )
            r.raise_for_status()
            data = r.json()
            logger.info("[%s] Contact history written — confirmation_id=%s",
                        session_id, data.get("confirmation_id"))
            return True
        except httpx.HTTPError as exc:
            logger.error("[%s] Contact history write failed: %s", session_id, exc)
            return False
