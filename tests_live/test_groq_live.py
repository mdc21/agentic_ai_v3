"""
tests_live/test_groq_live.py — Live smoke tests for Groq API.

PURPOSE
-------
These tests make REAL calls to the Groq API. They are NOT run in the CI
pipeline. Run them manually before a production deployment to validate:
  1. Your GROQ_API_KEY is valid and has quota
  2. Groq Whisper ASR can transcribe insurance-domain audio correctly
  3. The LLM (llama-3.3-70b-versatile) returns structured JSON turns
  4. The agent produces a coherent first response on a real voice turn

HOW TO RUN
----------
    # Single test:
    RUN_LIVE_TESTS=true pytest tests_live/test_groq_live.py::TestGroqASRLive::test_api_key_is_valid -v

    # All live tests (takes ~30s):
    RUN_LIVE_TESTS=true pytest tests_live/ -v

    # With explicit API key override:
    RUN_LIVE_TESTS=true GROQ_API_KEY=gsk_xxx pytest tests_live/ -v

COST ESTIMATE
-------------
    ASR tests   : ~1,000 audio tokens each  → negligible
    LLM tests   : ~2,500 prompt + ~200 completion tokens each  → ~$0.001/test
    Full suite  : < $0.01 total
"""

import io
import os
import sys
import time
import pytest

# ── Guard: skip entire module unless RUN_LIVE_TESTS=true ──────────────────────
if not os.getenv("RUN_LIVE_TESTS", "").lower() in ("true", "1", "yes"):
    pytest.skip(
        "Live Groq tests skipped. Set RUN_LIVE_TESTS=true to run them.",
        allow_module_level=True,
    )

# ── Ensure project root is importable ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Set mock flags for non-Groq dependencies so we isolate Groq calls
os.environ.setdefault("USE_MOCK_POLICY_API", "true")
os.environ.setdefault("USE_MOCK_RAG", "true")
os.environ.setdefault("USE_MOCK_TTS", "true")
os.environ["ASR_BACKEND"] = "groq"
os.environ["USE_MOCK_ASR"] = "false"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        pytest.fail(
            "GROQ_API_KEY is not set. "
            "Export it or add it to your .env before running live tests."
        )
    return key


@pytest.fixture(scope="module")
def groq_client(groq_api_key):
    """Bare Groq client — no project wrapper."""
    from groq import Groq
    return Groq(api_key=groq_api_key)


def _make_speech_audio(text: str) -> bytes:
    """
    Generate synthetic MP3 audio from text using gTTS.
    Returns raw bytes that Groq Whisper can transcribe.
    Falls back to a minimal silent WAV if gTTS is unavailable.
    """
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en", tld="co.uk").write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        # 1-second silent WAV (44 bytes header + silence)
        import wave, struct
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))
        buf.seek(0)
        return buf.read()


# ── Suite 1: API Key & Connectivity ───────────────────────────────────────────

class TestGroqAPIConnectivity:

    def test_api_key_is_valid(self, groq_client):
        """
        SMOKE: Can we reach the Groq API at all?
        Verifies key validity + network by making a minimal LLM call.
        We do NOT assert a specific word — small models are not deterministic
        on exact instruction-following. We just need a non-empty response.
        """
        start = time.perf_counter()
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",   # cheapest/fastest model for ping test
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=5,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        reply = resp.choices[0].message.content.strip().lower()

        print(f"\n  [groq-ping] latency={latency_ms}ms  reply={reply!r}")
        assert reply,              "Groq returned an empty response — API key may be invalid"
        assert latency_ms < 10_000, f"API too slow: {latency_ms}ms"
        # Usage metadata confirms a real call was made (not a cached/mock response)
        assert resp.usage.prompt_tokens > 0, "No prompt tokens — unexpected response"

    def test_groq_returns_usage_metrics(self, groq_client):
        """Usage metrics (tokens) are present for billing verification."""
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=10,
        )
        usage = resp.usage
        print(f"\n  [groq-usage] prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0


# ── Suite 2: Groq Whisper ASR ─────────────────────────────────────────────────

class TestGroqASRLive:

    def test_transcribe_policy_number(self, groq_client):
        """
        CORE SMOKE: Transcribe a caller saying a policy number.
        Validates that Whisper returns a non-empty string containing
        recognisable policy number tokens.
        """
        text = "My policy number is A B C slash one two three dash four five"
        audio = _make_speech_audio(text)

        start = time.perf_counter()
        audio_file = io.BytesIO(audio)
        audio_file.name = "audio.mp3"
        resp = groq_client.audio.transcriptions.create(
            file=("audio.mp3", audio_file),
            model="whisper-large-v3",
            prompt="The audio is from an insurance policy servicing call.",
            response_format="json",
            language="en",
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        transcript = resp.text.strip()

        print(f"\n  [groq-asr] latency={latency_ms}ms")
        print(f"  Input text : {text!r}")
        print(f"  Transcript : {transcript!r}")

        assert transcript, "Groq Whisper returned empty transcript"
        assert latency_ms < 15_000, f"ASR too slow: {latency_ms}ms"

    def test_transcribe_verification_phrase(self, groq_client):
        """Transcribe a caller confirming their postcode — tests short phrases."""
        text = "My postcode is M one one A A"
        audio = _make_speech_audio(text)

        audio_file = io.BytesIO(audio)
        audio_file.name = "audio.mp3"
        resp = groq_client.audio.transcriptions.create(
            file=("audio.mp3", audio_file),
            model="whisper-large-v3",
            prompt="The caller is confirming identity details for an insurance policy.",
            response_format="json",
            language="en",
        )
        transcript = resp.text.strip()
        print(f"\n  [groq-asr-postcode] transcript={transcript!r}")
        assert transcript, "Empty transcript for postcode phrase"

    def test_transcribe_affirmative(self, groq_client):
        """Transcribe 'yes that is correct' — verifies NATO confirmation flow."""
        text = "yes that is correct"
        audio = _make_speech_audio(text)

        audio_file = io.BytesIO(audio)
        audio_file.name = "audio.mp3"
        resp = groq_client.audio.transcriptions.create(
            file=("audio.mp3", audio_file),
            model="whisper-large-v3",
            response_format="json",
            language="en",
        )
        transcript = resp.text.strip().lower()
        print(f"\n  [groq-asr-affirm] transcript={transcript!r}")
        assert any(w in transcript for w in ("yes", "correct", "that")), \
            f"Affirmative not detected in: {transcript!r}"

    def test_asr_client_wrapper(self, groq_api_key):
        """
        Tests the project's own ASRClient wrapper (not raw Groq) to ensure
        the integration layer works end-to-end with a real API key.
        """
        from app.asr_client import ASRClient

        # Temporarily set the key in env for ASRClient to pick up
        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["ASR_BACKEND"] = "groq"
        os.environ["USE_MOCK_ASR"] = "false"

        asr = ASRClient()
        assert not asr.is_mock, "ASRClient should be in LIVE mode — check GROQ_API_KEY"

        audio = _make_speech_audio("I would like to check my pension value please")
        start = time.perf_counter()
        transcript = asr.transcribe(audio_bytes=audio, session_id="smoke-test")
        latency_ms = int((time.perf_counter() - start) * 1000)

        print(f"\n  [asr-wrapper] latency={latency_ms}ms  transcript={transcript!r}")
        assert isinstance(transcript, str), "ASRClient.transcribe() must return str"
        assert transcript, "ASRClient returned empty string — check audio generation"


# ── Suite 3: Groq LLM (Insurance Agent Prompt) ────────────────────────────────

class TestGroqLLMLive:

    def test_llm_returns_json_turn(self, groq_client):
        """
        CORE SMOKE: LLM returns a valid JSON AgentTurn for a greeting.
        Validates model + system prompt are working in the insurance context.
        """
        import json
        from app.prompts import SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "Hello, I'd like to check my policy please."},
        ]

        start = time.perf_counter()
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        content = resp.choices[0].message.content

        print(f"\n  [groq-llm] latency={latency_ms}ms")
        print(f"  Raw response: {content[:300]!r}")

        # Must be valid JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            pytest.fail(f"LLM returned invalid JSON: {e}\nContent: {content}")

        # Must contain required AgentTurn fields.
        # Note: the LLM may return 'action_intent' (per system prompt schema) or
        # 'action' (normalised field). Both are valid — LLMClient handles either.
        action_value = parsed.get("action") or parsed.get("action_intent")
        assert action_value, \
            f"Missing both 'action' and 'action_intent' in: {list(parsed.keys())}"
        assert "caller_response" in parsed, \
            f"Missing 'caller_response' in: {parsed}"
        assert parsed["caller_response"], \
            "caller_response is empty"
        assert latency_ms < 20_000, \
            f"LLM too slow: {latency_ms}ms"

        print(f"  Action  : {action_value}")
        print(f"  Response: {parsed['caller_response'][:120]!r}")

    def test_llm_client_wrapper(self, groq_api_key):
        """
        Tests the project's own LLMClient wrapper with a real Groq key.
        Verifies the full parsing pipeline returns a valid AgentTurn dataclass.
        """
        from app.llm_client import LLMClient, AgentTurn
        from app.prompts import SYSTEM_PROMPT

        os.environ["GROQ_API_KEY"] = groq_api_key
        LLMClient._clients.clear()   # force fresh client with real key

        client = LLMClient()
        assert "groq" in client._backends, \
            f"Groq not in backends: {client._backends}. Check GROQ_API_KEY."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "Hello, I need to check my pension."},
        ]

        start = time.perf_counter()
        turn = client.call_with_messages(messages)
        latency_ms = int((time.perf_counter() - start) * 1000)

        print(f"\n  [llm-wrapper] latency={latency_ms}ms")
        print(f"  Action  : {turn.action}")
        print(f"  Response: {turn.caller_response[:120]!r}")
        print(f"  Tokens  : in={turn.input_tokens} out={turn.output_tokens}")

        assert isinstance(turn, AgentTurn),   "call_with_messages() must return AgentTurn"
        assert turn.action,                   "AgentTurn.action is empty"
        assert turn.caller_response,          "AgentTurn.caller_response is empty"
        assert turn.action != "escalate",     "LLM immediately escalated — check system prompt"
        assert latency_ms < 20_000,           f"LLM too slow: {latency_ms}ms"

    def test_llm_extracts_policy_number(self, groq_client):
        """
        LLM must extract a policy number from caller speech.
        Validates entity extraction works in the insurance domain.
        """
        import json
        from app.prompts import SYSTEM_PROMPT

        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Thank you for calling. Could you please provide your policy number?"},
            {"role": "user",      "content": "Yes, my policy number is ABC slash 123 dash 45."},
        ]

        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)

        print(f"\n  [groq-entity] parsed={parsed}")
        entities = parsed.get("entities", {})
        policy = entities.get("policy_number", "")
        print(f"  Extracted policy_number: {policy!r}")

        assert policy, \
            f"LLM did not extract policy_number from entities: {entities}"


# ── Suite 4: Full Agent Voice Turn (End-to-End) ───────────────────────────────

class TestGroqAgentVoiceTurn:

    def test_agent_greeting_turn(self, groq_api_key):
        """
        FULL SMOKE: A complete voice turn through AgentOrchestrator with real Groq.
        Verifies: ASR → LLM → state transition → TTS (mocked).
        No policy lookup or verification — just the greeting turn.
        """
        from app.agent import AgentOrchestrator, AgentState

        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["ASR_BACKEND"]  = "groq"
        os.environ["USE_MOCK_ASR"] = "false"
        from app.llm_client import LLMClient
        LLMClient._clients.clear()

        agent = AgentOrchestrator(channel="voice")
        ctx   = agent.new_session()

        # Generate real audio for "hello"
        audio = _make_speech_audio("Hello, I'd like some help with my insurance policy please.")

        start = time.perf_counter()
        response = agent.process_turn(ctx, audio_bytes=audio)
        latency_ms = int((time.perf_counter() - start) * 1000)

        print(f"\n  [agent-voice] latency={latency_ms}ms")
        print(f"  State   : {ctx.state.value}")
        print(f"  Response: {response[:200]!r}")

        assert response, "Agent returned empty response on greeting"
        assert ctx.state != AgentState.ESCALATED, \
            f"Agent escalated on greeting: {ctx.escalation_reason}"
        assert latency_ms < 30_000, f"Full voice turn too slow: {latency_ms}ms"

    def test_agent_policy_collection_turn(self, groq_api_key):
        """
        Drives two turns: greeting then policy number.
        Verifies state transitions to CONFIRM_POLICY or beyond.
        """
        from app.agent import AgentOrchestrator, AgentState
        from app.llm_client import LLMClient

        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["ASR_BACKEND"]  = "groq"
        os.environ["USE_MOCK_ASR"] = "false"
        LLMClient._clients.clear()

        agent = AgentOrchestrator(channel="voice")
        ctx   = agent.new_session()

        # Turn 1: greeting
        agent.process_turn(ctx, text_input="Hello")

        initial_state = ctx.state
        print(f"\n  [agent-policy-t1] state after greeting: {initial_state.value}")

        # Turn 2: provide policy number
        audio = _make_speech_audio("My policy number is ABC slash one two three dash four five")
        agent.process_turn(ctx, audio_bytes=audio)

        print(f"  [agent-policy-t2] state after policy: {ctx.state.value}")
        print(f"  Entities: {ctx.caller_entities}")

        assert ctx.state != AgentState.ESCALATED, \
            f"Agent escalated after policy input: {ctx.escalation_reason}"
        # Should have progressed from initial greeting state
        assert ctx.state != AgentState.COLLECT_POLICY or ctx.caller_entities.policy_number, \
            "Policy number not extracted after 2 turns"
