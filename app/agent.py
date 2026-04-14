"""
Agent Orchestrator v3 — full spec implementation.

New vs v2:
  - Platform Directory System check (formerly OCIS) before SoR_1
  - Policy flags: value_restriction, share_restriction checked pre-serve
  - Caller flags: vulnerable_customer, fcu_flag checked post-verification
  - RAG / FAQ knowledge base for general process queries
  - TTS adapter for voice channel
  - Contact history written to SoR_1 on every session close
  - Loop detection (same intent >= 3 consecutive turns)
  - Duress signal escalation
  - Caller-requests-human escalation (any state)
  - Product-type contextual responses
  - product_type, heritage_brand, sor_system stored on ConversationContext
"""

import logging, os, re, uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.llm_client import AgentTurn, Entities, LLMClient
# from app.asr_client import ASRClient
from tools.policy import PolicyAPIClient, PolicyRecord
from tools.fuzzy import verify_caller, VerificationResult
from app.session_cache import SessionCache, AuditLogger
from app.tool_registry import ToolRegistry
from app.transfer_summary import build_transfer_summary, TransferSummary
from app.contact_history import ContactHistoryClient, build_contact_history
from app.rag_client import RAGClient
from app.prompts import build_messages, SYSTEM_PROMPT

logger = logging.getLogger(__name__)
MAX_POLICY_RETRIES = 3
LOOP_THRESHOLD     = int(os.getenv("LOOP_DETECTION_THRESHOLD", "3"))


class AgentState(Enum):
    GREET                   = "greet"
    COLLECT_POLICY          = "collect_policy"
    CONFIRM_POLICY          = "confirm_policy"
    PLATFORM_DIRECTORY_CHECK= "platform_directory_check"
    SOR_CHECK               = "sor_check"
    IDENTIFY_CALLER         = "identify_caller"
    VERIFY_POLICYHOLDER     = "verify_policyholder"
    VERIFY_ADVISER          = "verify_adviser"
    CHECK_CALLER_FLAGS      = "check_caller_flags"
    SERVE_INTENT            = "serve_intent"
    CREATE_CONTACT_HISTORY  = "create_contact_history"
    RESOLVED                = "resolved"
    ESCALATED               = "escalated"


class CallerType(Enum):
    POLICY_HOLDER    = "policy_holder"
    FA_REPRESENTATIVE= "fa_representative"
    TRUSTEE          = "trustee"
    EMPLOYER         = "employer"
    THIRD_PARTY      = "third_party"
    UNKNOWN          = "unknown"


_ESCALATE_TYPES = {CallerType.TRUSTEE, CallerType.EMPLOYER, CallerType.THIRD_PARTY}


@dataclass
class ConversationContext:
    session_id: str       = field(default_factory=lambda: str(uuid.uuid4()))
    channel: str          = "chat"
    state: AgentState     = AgentState.GREET
    turn_history: list    = field(default_factory=list)
    cache: SessionCache   = field(default_factory=SessionCache)
    # Policy routing
    product_type: Optional[str]     = None   # pension|life|protection|annuity
    heritage_brand: Optional[str]   = None
    sor_system: Optional[str]       = None   # SoR_1 | ...
    policy_flags: dict              = field(default_factory=dict)
    # Caller
    caller_type: Optional[CallerType] = None
    policy_number: Optional[str]      = None
    policy_record: Optional[PolicyRecord] = None
    caller_entities: Entities           = field(default_factory=Entities)
    # Tracking
    policy_retry_count: int   = 0
    call_intent: Optional[str] = None
    intents_served: list       = field(default_factory=list)
    # Verification
    verification_result: Optional[VerificationResult]         = None
    adviser_verification_result: Optional[VerificationResult] = None
    # Escalation
    escalation_reason: Optional[str] = None
    # Loop detection: (action_intent, frozenset(entity_vals)) repeated N times
    _last_intent_key: Optional[tuple] = field(default=None, repr=False)
    # Model tracking
    active_model: Optional[str] = None
    # Analytics tracking
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    turn_number: int = 0


class AgentOrchestrator:
    def __init__(self, channel: str = "chat") -> None:
        self._channel  = channel
        self._llm      = LLMClient()
        self._asr      = self._init_asr(channel)
        self._policy   = PolicyAPIClient()
        self._audit    = AuditLogger()
        self._contact  = ContactHistoryClient()
        self._tts      = self._init_tts(channel)

    def _init_asr(self, channel):
        if channel == "voice":
            from app.asr_client import ASRClient
            return ASRClient()
        return None

    def _init_tts(self, channel):
        if channel == "voice":
            from app.tts_client import TTSClient
            return TTSClient()
        return None

    def new_session(self) -> ConversationContext:
        ctx = ConversationContext(channel=self._channel)
        self._audit.log_session_open(ctx.session_id, ctx.channel)
        logger.info("New session %s (%s)", ctx.session_id, ctx.channel)
        return ctx

    # ── Entry point ───────────────────────────────────────────────────────────

    def process_turn(self, ctx: ConversationContext,
                     audio_bytes=None, text_input=None) -> str:
        if ctx.state in (AgentState.RESOLVED, AgentState.ESCALATED):
            return "This conversation has now concluded. Thank you for your time today, and please feel free to start a new chat if you need further assistance."

        ctx.turn_number += 1

        # 1. ASR (voice) or pass-through (chat)
        if self._asr:
            user_text = self._asr.transcribe(audio_bytes=audio_bytes, text_input=text_input,
                                             session_id=ctx.session_id)
        else:
            user_text = text_input or ""

        user_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", user_text)[:2000]
        logger.info("[%s T%d] User: %r", ctx.session_id, ctx.turn_number, user_text)

        ctx.turn_history.append({"role": "user", "content": user_text})

        # 1.5 Hardcoded escalation catch (Defense in depth)
        import re as _re_esc
        if _re_esc.search(r'\b(human|operator|real person|speak to someone|representative|specialist)\b', user_text.lower()):
            logger.warning("[%s T%d] Caller explicitly requested human: %s", ctx.session_id, ctx.turn_number, user_text)
            response = self._escalate(ctx, "CALLER_REQUESTED_HUMAN")
            ctx.turn_history.append({"role": "assistant", "content": response})
            return self._speak(response, ctx)

        # 2. Build state context and call LLM
        from app.prompts import build_messages
        state_ctx = self._state_context(ctx)
        messages  = build_messages(list(ctx.turn_history), state_ctx)
        
        import time
        start_t = time.perf_counter()
        turn: AgentTurn = self._llm.call_with_messages(messages)
        ctx.active_model = turn.model_name
        llm_latency_ms = int((time.perf_counter() - start_t) * 1000)

        logger.info("[%s T%d] LLM Turn: %s", ctx.session_id, ctx.turn_number, turn)

        # 3b. Sanitize entities (convert string "null"/"none" to None)
        for field_name in turn.entities.__dataclass_fields__:
            val = getattr(turn.entities, field_name)
            if isinstance(val, str) and val.lower() in ("null", "none"):
                setattr(turn.entities, field_name, None)
        
        # 3c. Strict Field-Level Validation (Regex/Heuristics)
        # UK Postcode validation
        for field in ("postcode", "adviser_postcode"):
            val = getattr(turn.entities, field)
            if val:
                import re as _re_p
                # Standard UK Postcode regex
                if not _re_p.match(r'^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$', val.strip().upper()):
                    logger.warning("[%s T%d] Rejecting invalid postcode: %s", ctx.session_id, ctx.turn_number, val)
                    setattr(turn.entities, field, None)
        
        # Name validation (letters, spaces, hyphens, apostrophes)
        for field in ("first_name", "last_name", "adviser_rep_name"):
            val = getattr(turn.entities, field)
            if val:
                import re as _re_n
                if not _re_n.match(r"^[A-Za-z\s'\-]+$", val):
                    logger.warning("[%s T%d] Rejecting invalid name chars: %s", ctx.session_id, ctx.turn_number, val)
                    setattr(turn.entities, field, None)
        
        # Address Heuristics (at least one digit and letters)
        for field in ("address_line1", "adviser_address_line1"):
            val = getattr(turn.entities, field)
            if val:
                import re as _re_a
                if not (_re_a.search(r'\d', val) and _re_a.search(r'[A-Za-z]', val)):
                    logger.warning("[%s T%d] Rejecting invalid address (heuristics): %s", ctx.session_id, ctx.turn_number, val)
                    setattr(turn.entities, field, None)

        # 4. Audit LLM turn
        ctx.total_input_tokens += (turn.input_tokens or 0)
        ctx.total_output_tokens += (turn.output_tokens or 0)
        ctx.total_cost += (turn.token_cost or 0.0)
        
        self._audit.log_llm_turn(
            ctx.session_id, ctx.turn_number, ctx.state.value,
            turn.action, turn.intent, turn.confidence,
            {f: getattr(turn.entities, f) for f in turn.entities.__dataclass_fields__},
            turn.caller_response, duress_signal=turn.duress_signal,
            itok=turn.input_tokens, otok=turn.output_tokens,
            input_text=turn.prompt_used, raw_llm_out=turn.raw_response, 
            token_cost=turn.token_cost, latency_ms=llm_latency_ms
        )

        # 4. Merge entities, update caller type and intent
        ctx.caller_entities.update(turn.entities)

        # 4a. Server-side entity fallbacks when LLM misses extraction
        if ctx.state == AgentState.VERIFY_POLICYHOLDER:
            # DOB fallback — only activate if user text looks like it contains a year
            import re as _re_dob
            if not ctx.caller_entities.date_of_birth and user_text and _re_dob.search(r'\b(19|20)\d{2}\b', user_text):
                try:
                    from dateutil import parser as _dateutil
                    parsed_dt = _dateutil.parse(user_text.strip(), dayfirst=True, fuzzy=True)
                    ctx.caller_entities.date_of_birth = parsed_dt.strftime("%Y-%m-%d")
                    logger.info("[%s T%d] DOB fallback extracted: %s", ctx.session_id, ctx.turn_number, ctx.caller_entities.date_of_birth)
                except (ValueError, OverflowError):
                    pass

            # Name mislabelling correction: LLM sometimes puts policyholder name in adviser_firm_name
            if (ctx.caller_entities.adviser_firm_name
                    and (not ctx.caller_entities.first_name or not ctx.caller_entities.last_name)):
                raw_name = ctx.caller_entities.adviser_firm_name.strip()
                parts = raw_name.split()
                if 1 <= len(parts) <= 3 and all(p.replace("'", "").isalpha() for p in parts):
                    ctx.caller_entities.first_name = parts[0].title()
                    ctx.caller_entities.last_name = " ".join(parts[1:]).title() if len(parts) > 1 else parts[0].title()
                    ctx.caller_entities.adviser_firm_name = None  # clear the mislabelled field
                    logger.info("[%s T%d] Name mislabelling corrected: first=%s last=%s",
                                ctx.session_id, ctx.turn_number,
                                ctx.caller_entities.first_name, ctx.caller_entities.last_name)

            # Name fallback: parse from raw text if one or both name fields still empty and text looks like a name
            if ((not ctx.caller_entities.first_name or not ctx.caller_entities.last_name) and user_text):
                import re as _re2
                stripped = user_text.strip()
                if (not _re2.search(r'\d', stripped)                     # no digits
                        and len(stripped.split()) in (2, 3)              # 2 or 3 words
                        and all(p.replace("'","").isalpha() for p in stripped.split())):
                    parts = stripped.split()
                    ctx.caller_entities.first_name = parts[0].title()
                    ctx.caller_entities.last_name = " ".join(parts[1:]).title()
                    logger.info("[%s T%d] Name fallback extracted: first=%s last=%s",
                                ctx.session_id, ctx.turn_number,
                                ctx.caller_entities.first_name, ctx.caller_entities.last_name)

        if ctx.state == AgentState.VERIFY_ADVISER and not ctx.caller_entities.adviser_postcode and user_text:
            import re as _re
            m = _re.search(r'[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}', user_text.strip().upper())
            if m:
                ctx.caller_entities.adviser_postcode = m.group(0).upper()
                logger.info("[%s T%d] Adviser postcode fallback extracted: %s", ctx.session_id, ctx.turn_number, ctx.caller_entities.adviser_postcode)

        # 4b. Server-side auto-completion: trigger verification if all required fields collected
        e = ctx.caller_entities
        ph_fields_done = all([e.first_name, e.last_name, e.address_line1, e.postcode, e.date_of_birth])
        # adviser_rep_name is audit-only — do NOT require it to trigger compare_adviser_verification
        adv_fields_done = all([e.adviser_firm_name, e.adviser_address_line1, e.adviser_postcode])
        if ctx.state == AgentState.VERIFY_POLICYHOLDER and ph_fields_done and turn.action != "compare_verification":
            logger.info("[%s T%d] All policyholder fields collected — forcing compare_verification", ctx.session_id, ctx.turn_number)
            turn.action = "compare_verification"
        elif ctx.state == AgentState.VERIFY_ADVISER and adv_fields_done and turn.action != "compare_adviser_verification":
            logger.info("[%s T%d] All adviser fields collected — forcing compare_adviser_verification", ctx.session_id, ctx.turn_number)
            turn.action = "compare_adviser_verification"

        # Update call intent if confidence is high or not yet set
        # Exclude internal orchestration/verification actions from being saved as the user's "purpose"
        INTERNAL_ACTIONS = {
            "greet", "ask_purpose", "request_policy_number", "confirm_policy_number", 
            "retry_policy_number", "policy_exist_platform_directory_check", 
            "policy_exist_SOR_check", "identify_caller_type", "request_verification", 
            "continue_verification", "compare_verification", "request_adviser_verification", 
            "compare_adviser_verification", "resolve_session", "create_contact_history"
        }
        if turn.intent and turn.intent not in INTERNAL_ACTIONS:
            if not ctx.call_intent or turn.confidence > 0.8:
                ctx.call_intent = turn.intent
        raw_type = turn.raw.get("caller_type","unknown")
        if raw_type and raw_type != "unknown":
            try: ctx.caller_type = CallerType(raw_type)
            except ValueError: pass

        # 5. Duress check — any state
        if turn.duress_signal:
            response = self._escalate(ctx, "CALLER_EMOTIONAL_DURESS")
            return self._speak(response, ctx)

        # 6. Loop detection
        if self._detect_loop(ctx, turn):
            response = self._escalate(ctx, "LOOP_DETECTED")
            return self._speak(response, ctx)

        # 7. Dispatch
        response = self._dispatch(ctx, turn)
        ctx.turn_history.append({"role": "assistant", "content": response})
        logger.info("[%s T%d] Agent (%s): %r", ctx.session_id, ctx.turn_number, ctx.state.value, response[:80])

        # 8. TTS for voice
        response_text = self._speak(response, ctx)
        self._audit.log_user_bot_interaction(ctx.session_id, ctx.turn_number, text_input, response_text)
        return response_text

    # ── Loop detection ────────────────────────────────────────────────────────

    def _detect_loop(self, ctx: ConversationContext, turn: AgentTurn) -> bool:
        e = turn.entities
        # Differentiate keys based on the actual question asked, so back-to-back
        # different RAG queries don't accidentally trigger the loop detector!
        key = (turn.action, turn.rag_query, frozenset(v for v in [
            e.policy_number, e.first_name, e.last_name, e.postcode,
            e.adviser_postcode, e.adviser_firm_name] if v))
        if key == ctx._last_intent_key:
            ctx._loop_count += 1
        else:
            ctx._last_intent_key = key
            ctx._loop_count = 1
        return ctx._loop_count >= LOOP_THRESHOLD

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _dispatch(self, ctx: ConversationContext, turn: AgentTurn) -> str:
        a = turn.action

        if a == "request_policy_number": ctx.state = AgentState.COLLECT_POLICY; return turn.caller_response
        if a == "retry_policy_number":
            ctx.policy_retry_count += 1
            if ctx.policy_retry_count >= MAX_POLICY_RETRIES:
                return self._escalate(ctx, "POLICY_CAPTURE_RETRIES_EXCEEDED")
            ctx.state = AgentState.COLLECT_POLICY; return turn.caller_response
        if a == "confirm_policy_number":
            ctx.policy_number = ctx.caller_entities.policy_number
            if ctx.channel == "chat":
                # Short-circuit: skip the confirmation turn for chat
                return self._platform_directory_check(ctx)
            ctx.state = AgentState.CONFIRM_POLICY
            return turn.caller_response

        # Policy lookup chain
        if a == "policy_exist_platform_directory_check": return self._platform_directory_check(ctx)
        if a == "policy_exist_SOR_check":                return self._sor_check(ctx)
        if a in ("party_role_address_details","policy_basic_details"): return self._serve_intent(ctx, a, turn)

        # Caller identification
        if a == "identify_caller_type":    return self._handle_caller_type(ctx, turn)

        # Policyholder verification
        if a in ("request_verification", "continue_verification"):
            # Force transition to policyholder verification if we just identified an FA caller
            if (ctx.state in (AgentState.IDENTIFY_CALLER, AgentState.SOR_CHECK)
                    and ctx.caller_type == CallerType.FA_REPRESENTATIVE):
                ctx.state = AgentState.VERIFY_POLICYHOLDER
                return self._handle_caller_type(ctx, turn)

            if ctx.state != AgentState.VERIFY_ADVISER:
                ctx.state = AgentState.VERIFY_POLICYHOLDER
            return turn.caller_response
        if a == "compare_verification":    return self._verify_policyholder(ctx, turn)

        # Adviser verification
        if a == "request_adviser_verification":
            ctx.state = AgentState.VERIFY_ADVISER; return turn.caller_response
        if a == "compare_adviser_verification": return self._verify_adviser(ctx, turn)

        # Intent serving (Data + RAG Coordination)
        rag_context = ""
        sor_data = None
        
        # 1. Selective RAG lookup: Only run if LLM specifies rag_query
        if turn.rag_query or a == "rag_query":
            rag_result = self._get_rag_context(ctx, turn)
            if rag_result:
                if not rag_result.answerable:
                    return self._escalate(ctx, "UNANSWERABLE_QUERY")
                rag_context = rag_result.context_for_llm()
        
        # 2. Data Intent lookup
        data_actions = ("policy_valuation","policy_exist_SOR_check","policy_basic_details",
                       "party_role_address_details","policy_benefits","policy_status", "return_details")
        if a in data_actions:
            if a == "return_details":
                # Special case: return_details generates its own full summary
                return self._return_details(ctx, turn)
            
            sor_data = self._get_sor_data(ctx, a, turn)
            if sor_data is None:
                return self._escalate(ctx, "SOR_API_ERROR")

        # 3. Unified Synthesis: Combine responses if we have either RAG or SOR info
        if rag_context or sor_data:
            return self._synthesize_unified_response(ctx, turn, rag_context, sor_data)

        # Terminal
        if a == "create_contact_history":  return self._close_session(ctx, resolved=True, message="Is there anything else I can help you with today?")
        if a == "resolve_session":         return self._close_session(ctx, resolved=True, message="Thank you for contacting us today. We appreciate your business and hope you have a wonderful day. Goodbye!")
        if a == "escalate":                return self._escalate(ctx, "LLM_DIRECTED")

        return turn.caller_response

    # ── Platform Directory System check ───────────────────────────────────────

    def _platform_directory_check(self, ctx: ConversationContext) -> str:
        num = ctx.caller_entities.policy_number or ctx.policy_number
        if not num:
            ctx.state = AgentState.COLLECT_POLICY
            return "I'm sorry, I didn't catch the policy number. Could you say it again?"
        ctx.policy_number = num
        ctx.state = AgentState.PLATFORM_DIRECTORY_CHECK

        ck = f"platform_dir:{num}"
        cached = ctx.cache.get(ck)
        if not cached:
            # In mock mode, policy tool acts as PDS too
            pol = self._policy.get_policy_details(num)
            if not pol:
                return self._escalate(ctx, "POLICY_NOT_FOUND_PLATFORM_DIR",
                    f"I'm sorry, I wasn't able to locate a policy with number {num} "
                    "in our systems. Let me connect you with a specialist.")
            cached = {
                "exists": True, 
                "heritage_brand": getattr(pol, 'heritage_brand', 'Brand_A'),
                "product_type": getattr(pol, 'product_type', 'life'),
                "sor_system": "SoR_1", 
                "policy_number": num
            }
            ctx.cache.set(ck, cached)

        self._audit.log_tool_call(ctx.session_id, ctx.turn_number,
            "policy_exist_platform_directory_check", {"policy_number": num},
            ["heritage_brand","product_type","sor_system"], hit=bool(ctx.cache.get(ck)))

        ctx.heritage_brand = cached.get("heritage_brand","Brand_A")
        ctx.product_type   = cached.get("product_type","life")
        ctx.sor_system     = cached.get("sor_system","SoR_1")

        # Proceed to SoR check
        return self._sor_check(ctx)

    # ── SoR_1 policy check ────────────────────────────────────────────────────

    def _sor_check(self, ctx: ConversationContext) -> str:
        ctx.state = AgentState.SOR_CHECK
        ck = f"sor1:{ctx.policy_number}"
        cached = ctx.cache.get(ck)

        if not cached:
            pol = self._policy.get_policy_details(ctx.policy_number)
            if not pol:
                return self._escalate(ctx, "POLICY_NOT_FOUND_SOR",
                    "I'm sorry, I wasn't able to retrieve the details for that policy. "
                    "Let me connect you with a specialist who can help.")
            ctx.policy_record = pol
            ctx.cache.set(ck, pol)
            ctx.cache.set(f"party:{ctx.policy_number}", pol.parties)
        else:
            ctx.policy_record = cached

        # Check policy-level flags
        for flag in ("value_restriction","share_restriction"):
            flag_val = getattr(ctx.policy_record, flag, False)
            self._audit.log_flag_check(ctx.session_id, ctx.turn_number, flag, bool(flag_val))
            if flag_val:
                ctx.policy_flags[flag] = True
                return self._escalate(ctx, "POLICY_FLAG_RESTRICTION")

        # Only go to IDENTIFY_CALLER if we don't already know the caller type
        if not ctx.caller_type or ctx.caller_type == CallerType.UNKNOWN:
            ctx.state = AgentState.IDENTIFY_CALLER
            return ("I've located your policy. Before I can share any details, "
                    "could you tell me your relationship to this policy? Are you the "
                    "policy holder, or are you calling on behalf of a financial adviser?")
        
        # If already known, proceed to verification or serving
        if ctx.verification_result and ctx.verification_result.passed:
            ctx.state = AgentState.SERVE_INTENT
            return "I've located your policy and confirmed your details. How can I help you further?"
        
        ctx.state = AgentState.IDENTIFY_CALLER
        return ("I've located your policy. Before I can share any details, "
                "could you tell me your relationship to this policy? "
                "Are you the policy holder, or are you calling on behalf of a financial adviser?")

    # ── Caller type ───────────────────────────────────────────────────────────

    def _handle_caller_type(self, ctx: ConversationContext, turn: AgentTurn) -> str:
        if ctx.caller_type in _ESCALATE_TYPES:
            return self._escalate(ctx, "CALLER_TYPE_NOT_AUTHORISED")
        if ctx.caller_type and ctx.caller_type != CallerType.UNKNOWN:
            ctx.state = AgentState.VERIFY_POLICYHOLDER
            if ctx.caller_type == CallerType.FA_REPRESENTATIVE:
                # FA callers: must verify the POLICYHOLDER identity first before adviser firm details
                return ("Thank you. Before I can assist, I need to first verify the identity of "
                        "the policy holder. Could you please provide the policy holder's "
                        "first and last name?")
            return ("Thank you. For security purposes I need to verify the policy holder's details. "
                    "Could you please provide the policy holder's first and last name?")
        return turn.caller_response

    # ── Policyholder verification ─────────────────────────────────────────────

    def _verify_policyholder(self, ctx: ConversationContext, turn: AgentTurn) -> str:
        if not ctx.policy_record:
            return self._escalate(ctx, "SOR_API_ERROR")
        party = (self._policy.get_party_for_role(ctx.policy_record,"policy_owner") or
                 self._policy.get_party_for_role(ctx.policy_record,"life_insured"))
        if not party:
            return self._escalate(ctx, "SOR_API_ERROR")

        sor = {"first_name":party.first_name,"last_name":party.last_name,
               "address_line1":party.address_line1,"postcode":party.postcode,
               "date_of_birth":party.date_of_birth}
        caller = {"first_name":ctx.caller_entities.first_name,"last_name":ctx.caller_entities.last_name,
                  "address_line1":ctx.caller_entities.address_line1,"postcode":ctx.caller_entities.postcode,
                  "date_of_birth":ctx.caller_entities.date_of_birth}

        result = verify_caller(caller, sor)
        ctx.verification_result = result
        self._audit.log_verification(ctx.session_id, ctx.turn_number,
            list(result.results.keys()),
            [f for f,r in result.results.items() if r.passed],
            result.failed_fields, result.verified)

        if not result.verified:
            return self._escalate(ctx, "VERIFICATION_FAILED_POLICYHOLDER")

        # Check caller-level flags
        for flag in ("vulnerable_customer","fcu_flag"):
            flag_val = getattr(party, flag, False)
            self._audit.log_flag_check(ctx.session_id, ctx.turn_number, flag, bool(flag_val))
            if flag_val:
                ctx.policy_flags[flag] = True
                return self._escalate(ctx, f"{flag.upper()}_FLAG")

        if ctx.caller_type == CallerType.FA_REPRESENTATIVE:
            ctx.state = AgentState.VERIFY_ADVISER
            return ("Thank you, the policy holder's details are confirmed. "
                    "Could you please give me the name of your adviser firm?")

        ctx.state = AgentState.SERVE_INTENT
        brand = ctx.heritage_brand or "your"
        ptype = ctx.product_type or "policy"
        if ctx.call_intent:
            return (f"Thank you, I've verified your identity. "
                    f"I see you need help with {ctx.call_intent.replace('_', ' ')} for your {brand} {ptype} policy. "
                    f"Please go ahead.")
        return (f"Thank you, I've verified your identity. "
                f"I can now help you with your {brand} {ptype} policy. "
                f"What would you like to know?")

    # ── Adviser verification ──────────────────────────────────────────────────

    def _verify_adviser(self, ctx: ConversationContext, turn: AgentTurn) -> str:
        if not ctx.policy_record:
            return self._escalate(ctx, "SOR_API_ERROR")
        ck = f"adviser:{ctx.policy_number}"
        adviser = ctx.cache.get(ck) or self._policy.get_party_for_role(ctx.policy_record, "financial_adviser")
        if adviser: ctx.cache.set(ck, adviser)
        if not adviser:
            return self._escalate(ctx, "CALLER_TYPE_NOT_AUTHORISED",
                "I'm sorry, there doesn't appear to be a financial adviser registered on this policy. "
                "Let me connect you with a specialist.")

        # Check that the FA firm's servicing role is still active on this policy
        if getattr(adviser, "role_status", "active") != "active":
            return self._escalate(ctx, "ADVISER_ROLE_INACTIVE",
                "I'm sorry, the financial adviser registration for this policy is no longer active. "
                "Let me connect you with a specialist.")

        sor = {"adviser_firm_name":      adviser.firm_name or adviser.first_name,
               "adviser_address_line1":  adviser.address_line1,
               "adviser_postcode":       adviser.postcode}
        caller = {"adviser_firm_name":     ctx.caller_entities.adviser_firm_name,
                  "adviser_address_line1": ctx.caller_entities.adviser_address_line1,
                  "adviser_postcode":      ctx.caller_entities.adviser_postcode}

        result = verify_caller(caller, sor)
        ctx.adviser_verification_result = result
        self._audit.log_verification(ctx.session_id, ctx.turn_number,
            list(result.results.keys()),
            [f for f,r in result.results.items() if r.passed],
            result.failed_fields, result.verified)

        if not result.verified:
            return self._escalate(ctx, "VERIFICATION_FAILED_ADVISER")

        ctx.state = AgentState.SERVE_INTENT
        if ctx.call_intent:
            return ("Thank you, your firm details are confirmed. "
                    f"I see you are calling about {ctx.call_intent.replace('_', ' ')}. Please go ahead.")
        return ("Thank you, your firm details are confirmed. "
                "How can I assist you with this policy today?")

    # ── RAG / FAQ intent serving ──────────────────────────────────────────────

    # ── Context Gatherers (Internal) ──────────────────────────────────────────
    
    def _get_rag_context(self, ctx: ConversationContext, turn: AgentTurn):
        """Fetch FAQ context only when strictly needed."""
        question = turn.rag_query or turn.intent or ""
        if not question: return None
        
        rag = RAGClient(ctx.cache)
        return rag.query(question, product_type=ctx.product_type,
                        heritage_brand=ctx.heritage_brand,
                        session_id=ctx.session_id, audit_logger=self._audit)

    def _get_sor_data(self, ctx: ConversationContext, action: str, turn: AgentTurn) -> Optional[dict]:
        """Fetch raw SoR data without synthesising yet."""
        if not ctx.policy_number: return None
        registry = ToolRegistry(ctx.cache, self._audit)
        params = {"policy_number": ctx.policy_number, "product_type": ctx.product_type or "life"}
        return registry.call(action, params, ctx.session_id, ctx.turn_number, product_type=ctx.product_type)

    def _synthesize_unified_response(self, ctx: ConversationContext, turn: AgentTurn, 
                                    rag_context: str, sor_data: Optional[dict]) -> str:
        """Unified synthesis for a combined RAG + Data response."""
        brand = ctx.heritage_brand or "our"
        ptype = ctx.product_type or "policy"
        
        # Build prompt sections
        data_block = ""
        if sor_data:
            data_block = f"REAL-TIME POLICY DATA:\n" + \
                        "\n".join(f"- {k.replace('_',' ').title()}: {v}" for k,v in sor_data.items())
        
        knowledge_block = ""
        if rag_context:
            knowledge_block = f"KNOWLEDGE BASE (FAQ) CONTEXT:\n{rag_context}"

        # Combine snippets for the LLM
        synthesis_prompt = (
            f"The caller has a verified {brand} {ptype} policy. "
            f"Help them by synthesizing the following information into a warm, natural, and professional response.\n\n"
            f"{knowledge_block}\n\n"
            f"{data_block}\n\n"
            f"Caller's request keywords: {ctx.call_intent or 'Service query'}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. If both Knowledge and Data are present, integrate them into a holistic answer.\n"
            f"2. Use the FAQ for rules/taxes and the Data for the actual numbers.\n"
            f"3. Always mention 'your {brand} {ptype} policy'.\n"
            f"4. Keep it concise. End with: 'Is there anything else I can help you with today?'"
        )
        
        synth_turn = self._llm.call([], synthesis_prompt)
        return synth_turn.caller_response or turn.caller_response

    def _return_details(self, ctx: ConversationContext, turn: AgentTurn) -> str:
        if not ctx.policy_record:
            return self._escalate(ctx, "SOR_API_ERROR")
            
        summary = self._policy.format_policy_summary(ctx.policy_record)
        synthesis_prompt = (
            f"The caller is verified for policy {ctx.policy_number}. "
            f"Here is the full policy summary:\n\n{summary}\n\n"
            f"Synthesize this into a warm response. Don't repeat the raw table, "
            f"just highlight the key details as a narrative."
        )
        synth_turn = self._llm.call([], synthesis_prompt)
        return synth_turn.caller_response or summary

    # ── Session close (write contact history) ────────────────────────────────

    def _close_session(self, ctx: ConversationContext, resolved: bool, message: str = "") -> str:
        ctx.state = AgentState.CREATE_CONTACT_HISTORY
        payload = build_contact_history(ctx)
        success = self._contact.write(payload, ctx.session_id)
        self._audit.log_contact_history_write(ctx.session_id, ctx.turn_number, success)
        ctx.state = AgentState.RESOLVED if resolved else AgentState.ESCALATED
        self._audit.log_session_close(ctx.session_id, ctx.turn_number, ctx.state.value)
        
        # Write analytics summary
        self._audit.save_analytics_record(
            ctx.session_id, ctx.start_time, datetime.now(timezone.utc),
            ctx.total_input_tokens, ctx.total_output_tokens, ctx.total_cost,
            escalated=(ctx.state == AgentState.ESCALATED),
            state=ctx.state.value
        )
        
        if resolved:
            return message or "Is there anything else I can help you with your policy today?"
        return ""

    # ── Escalation ────────────────────────────────────────────────────────────

    def _escalate(self, ctx: ConversationContext, reason: str, caller_msg: Optional[str]=None) -> str:
        ctx.state = AgentState.ESCALATED
        ctx.escalation_reason = reason
        from app.transfer_summary import build_transfer_summary
        summary = build_transfer_summary(ctx)
        self._audit.log_escalation(ctx.session_id, ctx.turn_number, reason, ctx.state.value)
        
        # Write analytics summary
        self._audit.save_analytics_record(
            ctx.session_id, ctx.start_time, datetime.now(timezone.utc),
            ctx.total_input_tokens, ctx.total_output_tokens, ctx.total_cost,
            escalated=True, state=f"escalated:{reason}"
        )
        
        # Write contact history on escalation too
        payload = build_contact_history(ctx)
        success = self._contact.write(payload, ctx.session_id)
        self._audit.log_contact_history_write(ctx.session_id, ctx.turn_number, success)
        self._audit.log_session_close(ctx.session_id, ctx.turn_number, ctx.state.value)
        logger.warning("[%s] ESCALATED: %s\n%s", ctx.session_id, reason, summary.human_readable())
        return caller_msg or (
            "Thank you for your patience. I'm going to connect you with one of our specialists "
            "who will be happy to help you further. Please bear with me for a moment.")

    # ── TTS wrapper ───────────────────────────────────────────────────────────

    def _speak(self, text: str, ctx: ConversationContext) -> str:
        """For voice channel, synthesise and log. For chat, return text unchanged."""
        if self._tts and ctx.channel == "voice":
            audio = self._tts.synthesise(text, ctx.session_id)
            # In a real system, audio bytes would be streamed to the caller.
            # Here we return text for testability; the caller layer handles audio.
            logger.debug("[%s] TTS synthesised %d bytes", ctx.session_id, len(audio))
        return text

    # ── State context ─────────────────────────────────────────────────────────

    def _state_context(self, ctx: ConversationContext) -> str:
        lines = [f"[State: {ctx.state.value}]", f"[Channel: {ctx.channel}]"]
        if ctx.policy_number:  lines.append(f"[Policy number]: {ctx.policy_number}")
        if ctx.product_type:   lines.append(f"[Product type]: {ctx.product_type}")
        if ctx.heritage_brand: lines.append(f"[Heritage brand]: {ctx.heritage_brand}")
        if ctx.sor_system:     lines.append(f"[SoR system]: {ctx.sor_system}")
        if ctx.policy_record:  lines.append(f"[Policy found]: {ctx.policy_record.product_name}, {ctx.policy_record.status}")
        if ctx.caller_type:    lines.append(f"[Caller type]: {ctx.caller_type.value}")
        if ctx.policy_retry_count: lines.append(f"[Policy retries]: {ctx.policy_retry_count}/3")
        if ctx.call_intent:    lines.append(f"[Call intent]: {ctx.call_intent}")
        e = ctx.caller_entities
        have = [k for k in e.__dataclass_fields__ if getattr(e,k)]
        if have: lines.append(f"[Collected fields]: {have}")
        if ctx.state == AgentState.VERIFY_POLICYHOLDER:
            miss = [f for f in ("first_name","last_name","address_line1","postcode","date_of_birth") if not getattr(e,f)]
            if miss: lines.append(f"[Verification fields still needed]: {miss}")
        if ctx.state == AgentState.VERIFY_ADVISER:
            miss = [f for f in ("adviser_firm_name","adviser_address_line1","adviser_postcode") if not getattr(e,f)]
            if miss: lines.append(f"[Adviser fields still needed]: {miss}")
            lines.append("[Note]: adviser_rep_name is captured for audit only, NOT verified against SoR")
        flags = list(ctx.policy_flags.keys()) if ctx.policy_flags else []
        lines.append(f"[Policy flags]: {flags if flags else 'none'}")
        return "\n".join(lines)
