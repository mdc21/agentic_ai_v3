"""session_cache.py + audit_logger.py — v3 (adds RAG, flag check, contact history events)"""
import hashlib, json, logging, os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

def _redact(v): return "[REDACTED]" if v else None
def _hash(v):   return "sha256:" + hashlib.sha256(str(v).encode()).hexdigest()[:12]

class SessionCache:
    def __init__(self):            self._s = {}
    def get(self, k):              return self._s.get(k)
    def set(self, k, v):           self._s[k] = v; logger.debug("Cache SET %r", k)
    def has(self, k):              return k in self._s
    def clear(self):               self._s.clear()
    def keys(self):                return list(self._s.keys())

@dataclass
class AuditRecord:
    record_type: str; session_id: str; turn: int
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    channel: Optional[str] = None;          agent_state: Optional[str] = None
    input_tokens: Optional[int] = None;     output_tokens: Optional[int] = None
    llm_latency_ms: Optional[int] = None;   action_intent: Optional[str] = None
    confidence: Optional[float] = None
    intent: Optional[str] = None;           duress_signal: Optional[bool] = None
    extracted_entities: Optional[dict] = None; caller_response_hash: Optional[str] = None
    tool_name: Optional[str] = None;        tool_params: Optional[dict] = None
    tool_result_fields: Optional[list] = None; tool_cache_hit: Optional[bool] = None
    rag_query_hash: Optional[str] = None;   rag_product_type: Optional[str] = None
    rag_scores: Optional[list] = None;      rag_cache_hit: Optional[bool] = None
    rag_latency_ms: Optional[int] = None;   flag_name: Optional[str] = None
    flag_detected: Optional[bool] = None
    fields_checked: Optional[list] = None;  fields_passed: Optional[list] = None
    fields_failed: Optional[list] = None;   verification_passed: Optional[bool] = None
    escalation_reason: Optional[str] = None
    contact_history_success: Optional[bool] = None
    contact_confirmation_id: Optional[str] = None

    def to_json(self):
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None}, default=str)

class AuditLogger:
    def __init__(self):
        dest = os.getenv("AUDIT_LOG_DESTINATION", "stdout")
        self._sink = open(Path(dest[5:]), "a") if dest.startswith("file:") else None
        
        # New split loggers
        self._user_bot_log = open("user_bot_interaction.log", "a")
        self._bot_llm_log = open("bot_llm_interaction.log", "a")
        self._bot_sor_log = open("bot_sor_interaction.log", "a")
        self._bot_faq_log = open("bot_faq_interaction.log", "a")
        
        # Analytics CSV initialization
        self._analytics_file = "analytics.csv"
        if not os.path.exists(self._analytics_file):
            with open(self._analytics_file, "w") as f:
                f.write("session_id,start_time,end_time,duration_sec,total_input_tokens,total_output_tokens,total_cost_usd,escalated,final_state\n")

    def __del__(self):
        for f in (self._sink, self._user_bot_log, self._bot_llm_log, self._bot_sor_log, self._bot_faq_log):
            if f and not f.closed and f is not getattr(self, '_sink', None): # stdout check
                f.close()

    def _w(self, r):
        line = r.to_json()
        (self._sink.write(line+"\n") or self._sink.flush()) if self._sink else print(f"[AUDIT] {line}")

    def log_user_bot_interaction(self, sid: str, t: int, caller_input: str, bot_response: str):
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        msg = f"[{now}][{sid} T{t}] Caller: {caller_input}\n[{now}][{sid} T{t}] Bot: {bot_response}\n\n"
        self._user_bot_log.write(msg)
        self._user_bot_log.flush()

    def log_session_open(self, sid, ch):
        self._w(AuditRecord("session_open", sid, 0, channel=ch))
    def log_session_close(self, sid, t, state):
        self._w(AuditRecord("session_close", sid, t, agent_state=state))
    def log_llm_turn(self, sid, t, state, action, intent, conf, ents, resp, duress_signal=None, itok=None, otok=None, input_text=None, raw_llm_out=None, token_cost=None, latency_ms=None):
        self._w(AuditRecord("llm_turn", sid, t, agent_state=state, action_intent=action,
            intent=intent, confidence=conf, duress_signal=duress_signal,
            extracted_entities={k: _redact(v) for k,v in ents.items()},
            caller_response_hash=_hash(resp), input_tokens=itok, output_tokens=otok, llm_latency_ms=latency_ms))
        
        # Log to bot_llm_interaction.log
        if input_text and raw_llm_out:
            cost_str = f" | Cost: ${token_cost:.4f}" if token_cost is not None else ""
            tok_str = f" | Tokens: In={itok or 0} Out={otok or 0}"
            lat_str = f" | Latency: {latency_ms}ms" if latency_ms is not None else ""
            msg = f"[{sid} T{t}]\n-- LLM INPUT --\n{input_text}\n\n-- LLM RESPONSE --\n{raw_llm_out}\n\n[Stats]{tok_str}{lat_str}{cost_str}\n{'-'*60}\n"
            self._bot_llm_log.write(msg)
            self._bot_llm_log.flush()
    def log_tool_call(self, sid, t, name, params, fields, hit):
        safe = {k: (_hash(str(v)) if k=="policy_number" else "[param]") for k,v in params.items()}
        self._w(AuditRecord("tool_call", sid, t, tool_name=name, tool_params=safe,
            tool_result_fields=fields, tool_cache_hit=hit))
            
        # Log to bot_sor_interaction.log
        msg = f"[{sid} T{t}] API CALL to {name}\nParams: {params}\nFields Returned: {fields}\nCache Hit: {hit}\n{'-'*60}\n"
        self._bot_sor_log.write(msg)
        self._bot_sor_log.flush()
    def log_rag_query(self, sid, qhash, ptype, scores, hit, question=None, answerable=None, context=None, latency_ms=None):
        self._w(AuditRecord("rag_query", sid, 0, rag_query_hash=qhash,
            rag_product_type=ptype, rag_scores=[round(s,3) for s in scores], rag_cache_hit=hit, rag_latency_ms=latency_ms))
            
        # Log to bot_faq_interaction.log
        if question:
            msg = f"[{sid}] RAG QUERY\nQuestion: {question}\nProduct Type: {ptype}\nCache Hit: {hit}\nLatency: {latency_ms}ms\nAnswerable: {answerable}\nScores: {scores}\n\nContext Provided:\n{context}\n{'-'*60}\n"
            self._bot_faq_log.write(msg)
            self._bot_faq_log.flush()
    def log_flag_check(self, sid, t, flag, detected):
        self._w(AuditRecord("flag_check", sid, t, flag_name=flag, flag_detected=detected))
    def log_verification(self, sid, t, checked, passed_f, failed_f, passed):
        self._w(AuditRecord("verification", sid, t, fields_checked=checked,
            fields_passed=passed_f, fields_failed=failed_f, verification_passed=passed))
    def log_escalation(self, sid, t, reason, state):
        self._w(AuditRecord("escalation", sid, t, escalation_reason=reason, agent_state=state))
    def log_contact_history_write(self, sid, t, success, conf_id=None):
        self._w(AuditRecord("contact_history_write", sid, t,
            contact_history_success=success, contact_confirmation_id=conf_id))

    def save_analytics_record(self, sid: str, start_time: datetime, end_time: datetime, 
                              itok: int, otok: int, cost: float, escalated: bool, state: str):
        """Append a summary record to analytics.csv, capped at 10,000 records."""
        import csv
        duration = (end_time - start_time).total_seconds()
        row = [sid, start_time.isoformat(), end_time.isoformat(), f"{duration:.1f}", itok, otok, f"{cost:.6f}", str(escalated).lower(), state]
        
        # Read existing to enforce limit
        rows = []
        if os.path.exists(self._analytics_file):
            with open(self._analytics_file, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
        
        rows.append(row)
        if len(rows) > 10000:
            rows = rows[-10000:]
            
        with open(self._analytics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["session_id","start_time","end_time","duration_sec","total_input_tokens","total_output_tokens","total_cost_usd","escalated","final_state"])
            writer.writerows(rows)
        logger.info("[%s] Analytics record saved (total rows: %d)", sid, len(rows))
