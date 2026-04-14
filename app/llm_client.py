"""llm_client.py v3 — OpenAI-compatible + Anthropic SDK. Adds duress_signal parsing."""
import json, logging, os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration is now handled lazily within LLMClient to ensure reliability.


@dataclass
class Entities:
    policy_number: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    address_line1: Optional[str] = None
    postcode: Optional[str] = None
    date_of_birth: Optional[str] = None
    adviser_firm_name: Optional[str] = None
    adviser_address_line1: Optional[str] = None
    adviser_postcode: Optional[str] = None
    adviser_rep_name: Optional[str] = None

    def update(self, other: "Entities"):
        for f in self.__dataclass_fields__:
            val = getattr(other, f)
            if val is not None:
                setattr(self, f, val)


@dataclass
class AgentTurn:
    intent: str = ""
    entities: Entities = field(default_factory=Entities)
    action: str = "respond"
    rag_query: Optional[str] = None
    duress_signal: bool = False
    caller_response: str = ""
    confidence: float = 1.0
    raw: dict = field(default_factory=dict)
    prompt_used: Optional[str] = None
    raw_response: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    token_cost: Optional[float] = None
    model_name: Optional[str] = None


class LLMClient:
    def __init__(self):
        self._load_config()
        # Define priority list of backends based on available keys
        self._backends = []
        if self.groq_key:
            self._backends.append("groq")
        if self.openai_key or self.api_base:
            self._backends.append("openai")
        if self.anthropic_key:
            self._backends.append("anthropic")
        
        if not self._backends:
            logger.error("No LLM backends configured!")
            self._backends = ["mock"]

    def _load_config(self):
        """Fetch configuration lazily from environment or streamlit secrets."""
        # Try environment first (local .env)
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_base = os.getenv("LLM_API_BASE_URL", "")
        self.groq_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

        # Streamlit Secrets Override (for cloud deployment)
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                self.groq_key = st.secrets.get("GROQ_API_KEY", self.groq_key)
                self.groq_model = st.secrets.get("GROQ_MODEL", self.groq_model)
                self.openai_key = st.secrets.get("OPENAI_API_KEY", self.openai_key)
                self.anthropic_key = st.secrets.get("ANTHROPIC_API_KEY", self.anthropic_key)
                self.model = st.secrets.get("LLM_MODEL", self.model)
                self.api_base = st.secrets.get("LLM_API_BASE_URL", self.api_base)
        except:
            pass

    def call_with_messages(self, messages: list) -> AgentTurn:
        errors = []
        for backend in self._backends:
            try:
                logger.info(">>> LLM REQUEST [%s]: %s", backend, messages[-1] if messages else "No messages")
                if backend == "groq":
                    turn = self._call_groq(messages)
                    turn.model_name = f"🚀 Groq ({self.groq_model})"
                elif backend == "anthropic":
                    turn = self._call_anthropic(messages)
                    turn.model_name = f"☁️ Anthropic ({self.model})"
                elif backend == "openai":
                    turn = self._call_openai(messages)
                    turn.model_name = f"☁️ OpenAI ({self.model})"
                else:
                    return AgentTurn(action="escalate", caller_response="No LLM backends available.")
                
                logger.info("<<< LLM RESPONSE [%s]: %s", backend, turn.raw_response[:200] + "..." if turn.raw_response else "Empty")
                return turn
            except Exception as exc:
                # Catch specific rate limit or down errors
                exc_str = str(exc).lower()
                if "rate limit" in exc_str or "429" in exc_str or "insufficient_quota" in exc_str:
                    logger.warning("LLM Backend [%s] rate limited or quota exceeded. Trying failover...", backend)
                    errors.append(f"{backend}: {exc}")
                    continue
                else:
                    logger.error("LLM Backend [%s] fatal error: %s", backend, exc)
                    errors.append(f"{backend}: {exc}")
                    # In case of fatal non-retryable error, we could break or continue. 
                    # For insurance safety, we try the next one.
                    continue
        
        # If all backends fail
        logger.error("ALL LLM backends failed: %s", errors)
        return AgentTurn(action="escalate",
            caller_response="I'm sorry, I'm having a technical issue connecting to my brain. Let me connect you with a specialist.")

    def call(self, turn_history: list, user_input: str) -> AgentTurn:
        from app.prompts import build_messages
        return self.call_with_messages(
            build_messages(list(turn_history) + [{"role":"user","content":user_input}], ""))
    def _call_groq(self, messages: list) -> AgentTurn:
        from groq import Groq
        from app.prompts import SYSTEM_PROMPT
        client = Groq(api_key=self.groq_key)
        
        groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            m for m in messages if m.get("role") != "system"
        ]
        
        prompt_txt = str(groq_messages)
        
        resp = client.chat.completions.create(
            model=self.groq_model,
            messages=groq_messages,
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        raw_out = resp.choices[0].message.content.strip()
        itok = resp.usage.prompt_tokens
        otok = resp.usage.completion_tokens
        # Standard Llama 3 on Groq pricing (approximate)
        cost = (itok * 0.05 / 1000000) + (otok * 0.08 / 1000000)
        
        turn = self._parse(raw_out)
        turn.prompt_used = prompt_txt
        turn.raw_response = raw_out
        turn.input_tokens = itok
        turn.output_tokens = otok
        turn.token_cost = cost
        return turn

    def _call_anthropic(self, messages: list) -> AgentTurn:
        import anthropic
        from app.prompts import SYSTEM_PROMPT
        non_sys = [m for m in messages if m.get("role") != "system"]
        client  = anthropic.Anthropic()
        
        # Approximate input for logging
        prompt_txt = SYSTEM_PROMPT + "\n\n" + str(non_sys)
        
        resp    = client.messages.create(model=self.model, max_tokens=1024,
                                         system=SYSTEM_PROMPT, messages=non_sys)
        
        raw_out = resp.content[0].text.strip()
        itok = resp.usage.input_tokens
        otok = resp.usage.output_tokens
        cost = (itok * 0.003 / 1000) + (otok * 0.015 / 1000) # Claude 3.5 Sonnet approximations
        
        turn = self._parse(raw_out)
        turn.prompt_used = prompt_txt
        turn.raw_response = raw_out
        turn.input_tokens = itok
        turn.output_tokens = otok
        turn.token_cost = cost
        return turn

    def _call_openai(self, messages: list) -> AgentTurn:
        import httpx
        from app.prompts import SYSTEM_PROMPT
        
        openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            m for m in messages if m.get("role") != "system"
        ]
        
        # Approximate input for logging
        prompt_txt = str(openai_messages)

        r = httpx.post(f"{self.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {self.openai_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "temperature": 0.2, "max_tokens": 1024, "messages": openai_messages, "response_format": {"type": "json_object"}},
            timeout=30)
        r.raise_for_status()
        
        data = r.json()
        raw_out = data["choices"][0]["message"]["content"].strip()
        
        itok = data.get("usage", {}).get("prompt_tokens", 0)
        otok = data.get("usage", {}).get("completion_tokens", 0)
        
        # Approximations (e.g. gpt-4o)
        cost = (itok * 0.005 / 1000) + (otok * 0.015 / 1000)

        turn = self._parse(raw_out)
        turn.prompt_used = prompt_txt
        turn.raw_response = raw_out
        turn.input_tokens = itok
        turn.output_tokens = otok
        turn.token_cost = cost
        return turn

    @staticmethod
    def _parse(raw: str) -> AgentTurn:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode LLM JSON: %s. Raw: %r", e, raw)
            return AgentTurn(action="escalate", caller_response="I'm sorry, I'm having a technical issue. Let me connect you with a specialist.")

        # Formal JSON Schema Validation
        try:
            from jsonschema import validate
            from app.schema import AGENT_TURN_SCHEMA
            validate(instance=data, schema=AGENT_TURN_SCHEMA)
        except Exception as e:
            logger.error("LLM JSON schema validation failed: %s. Data: %r", e, data)

        ent_data = data.get("entities", {})
        ent = {k: v for k, v in ent_data.items()
                if k in Entities.__dataclass_fields__ and v is not None}
        return AgentTurn(
            intent=data.get("intent",""),
            entities=Entities(**ent),
            action=data.get("action_intent", data.get("action","respond")),
            rag_query=data.get("rag_query"),
            duress_signal=bool(data.get("duress_signal", False)),
            caller_response=data.get("caller_response",""),
            confidence=float(data.get("confidence",1.0)),
            raw=data,
        )
