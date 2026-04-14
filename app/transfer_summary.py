"""
transfer_summary.py v3 — generates the structured handoff document
placed on the human-agent queue on escalation.

Updated to include:
  - product_type, heritage_brand, sor_system
  - policy_flags and caller_flags
  - intents_served list
  - RAG queries answered (from intents_served)
"""
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class TransferSummary:
    session_id: str
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    channel: str = "chat"
    # Escalation
    escalation_reason: str = "UNKNOWN"
    # Policy routing
    policy_number: Optional[str] = None
    heritage_brand: Optional[str] = None
    product_type: Optional[str] = None
    sor_system: Optional[str] = None
    policy_found: bool = False
    policy_flags: list = field(default_factory=list)
    # Caller
    caller_type: Optional[str] = None
    verification_attempted: bool = False
    verification_passed: bool = False
    fields_verified: list = field(default_factory=list)
    fields_failed: list = field(default_factory=list)
    caller_flags: list = field(default_factory=list)
    # Session summary
    call_intent: Optional[str] = None
    intents_served: list = field(default_factory=list)
    transcript_summary: str = ""
    full_transcript: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), default=str, indent=indent)

    def human_readable(self) -> str:
        lines = [
            "=" * 65,
            "TRANSFER SUMMARY — AGENTIC AI ESCALATION",
            "=" * 65,
            f"Session:           {self.session_id}",
            f"Time:              {self.timestamp_utc}",
            f"Channel:           {self.channel.upper()}",
            f"Escalation reason: {self.escalation_reason}",
            "",
            f"Policy number:     {self.policy_number or 'Not captured'}",
            f"Policy found:      {'Yes' if self.policy_found else 'No'}",
            f"Heritage brand:    {self.heritage_brand or 'Unknown'}",
            f"Product type:      {self.product_type or 'Unknown'}",
            f"SoR system:        {self.sor_system or 'Unknown'}",
            f"Policy flags:      {', '.join(self.policy_flags) or 'None'}",
            "",
            f"Caller type:       {self.caller_type or 'Not identified'}",
            f"Caller flags:      {', '.join(self.caller_flags) or 'None'}",
            f"Call intent:       {self.call_intent or 'Not determined'}",
            f"Intents served:    {', '.join(self.intents_served) or 'None'}",
            "",
            "Identity verification:",
            f"  Attempted:       {'Yes' if self.verification_attempted else 'No'}",
            f"  Passed:          {'Yes' if self.verification_passed else 'No'}",
            f"  Fields verified: {', '.join(self.fields_verified) or 'None'}",
            f"  Fields failed:   {', '.join(self.fields_failed) or 'None'}",
            "",
            "Summary:",
            f"  {self.transcript_summary}",
            "",
            "NOTE: Human agent must independently re-verify caller before sharing any data.",
            "=" * 65,
        ]
        return "\n".join(lines)


def build_transfer_summary(ctx) -> TransferSummary:
    """Build a TransferSummary from a ConversationContext."""
    fields_verified, fields_failed = [], []
    verification_attempted = False

    if ctx.verification_result:
        verification_attempted = True
        fields_verified = [f for f, r in ctx.verification_result.results.items() if r.passed]
        fields_failed   = ctx.verification_result.failed_fields

    # Collect policy flags and caller flags from context
    policy_flags = [k for k, v in ctx.policy_flags.items() if v] if ctx.policy_flags else []
    caller_flags = []
    for flag in ("vulnerable_customer", "fcu_flag"):
        if ctx.policy_flags.get(flag):
            caller_flags.append(flag)

    # Build short summary
    parts = [f"Call of {len(ctx.turn_history)//2} turns."]
    if ctx.policy_number:
        parts.append(f"Policy {ctx.policy_number}"
                     f"{' found' if ctx.policy_record else ' not found'}.")
    if ctx.product_type:
        parts.append(f"Product: {ctx.product_type} ({ctx.heritage_brand or 'unknown brand'}).")
    if ctx.caller_type:
        parts.append(f"Caller: {ctx.caller_type.value}.")
    if ctx.call_intent:
        parts.append(f"Intent: {ctx.call_intent}.")
    parts.append(f"Escalation: {ctx.escalation_reason}.")

    return TransferSummary(
        session_id=ctx.session_id,
        channel=ctx.channel,
        escalation_reason=ctx.escalation_reason or "UNKNOWN",
        policy_number=ctx.policy_number,
        heritage_brand=getattr(ctx, "heritage_brand", None),
        product_type=getattr(ctx, "product_type", None),
        sor_system=getattr(ctx, "sor_system", None),
        policy_found=ctx.policy_record is not None,
        policy_flags=policy_flags,
        caller_type=ctx.caller_type.value if ctx.caller_type else None,
        verification_attempted=verification_attempted,
        verification_passed=(ctx.verification_result.verified
                             if ctx.verification_result else False),
        fields_verified=fields_verified,
        fields_failed=fields_failed,
        caller_flags=caller_flags,
        call_intent=ctx.call_intent,
        intents_served=list(getattr(ctx, "intents_served", [])),
        transcript_summary=" ".join(parts),
        full_transcript=list(ctx.turn_history),
    )
