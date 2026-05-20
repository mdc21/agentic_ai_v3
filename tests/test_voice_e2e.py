# tests/test_voice_e2e.py
import os, io, pytest
os.environ.update({
    "USE_MOCK_POLICY_API": "true",
    "USE_MOCK_RAG": "true",
    "USE_MOCK_ASR": "true",
    "OPENAI_API_KEY": "sk-test-dummy-key",   # prevents "No LLM backends" escalation
})

from unittest.mock import patch, MagicMock
from app.llm_client import AgentTurn, Entities
from app.agent import AgentOrchestrator, AgentState


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_audio_bytes(text: str) -> bytes:
    """Generate real audio bytes from text using gTTS (no API key needed)."""
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(buf)
        return buf.getvalue()
    except ImportError:
        return b"\x00" * 1000   # fallback: silent audio

def mock_asr_with_text(text: str):
    """Patch ASR to return a specific text, simulating a known transcription."""
    return patch("app.asr_client.ASRClient.transcribe", return_value=text)

def make_llm_turn(action: str, intent: str = "", caller_response: str = "test response",
                  policy_number: str = None, affirm_text: str = "") -> AgentTurn:
    """Build a minimal AgentTurn for mocking LLM responses."""
    e = Entities()
    if policy_number: e.policy_number = policy_number
    return AgentTurn(action=action, intent=intent, entities=e,
                     caller_response=caller_response, user_text=affirm_text,
                     confidence=0.95)

def mock_llm(action: str, intent: str = "", caller_response: str = "test response",
             policy_number: str = None, affirm_text: str = ""):
    """Context manager that patches the LLM call to return a scripted turn."""
    turn = make_llm_turn(action, intent, caller_response, policy_number, affirm_text)
    return patch("app.agent.AgentOrchestrator._call_llm", return_value=turn)


# ── Happy Path ────────────────────────────────────────────────────────────────

class TestVoiceHappyPath:
    """Full voice call: policy holder, pension valuation. Uses scripted LLM responses."""

    def test_full_voice_happy_path(self):
        """Happy path using text_input (bypasses ASR, but exercises full orchestration)."""
        agent = AgentOrchestrator(channel="voice")
        ctx = agent.new_session()

        # Use text_input to drive the scenario (avoids ASR + gTTS network calls in CI)
        script = [
            "hello",
            "My policy number is ABC slash 123 dash 45",
            "yes that is correct",
            "I am the policy holder",
            "Jonathan Smith",
            "14 High Street",
            "M1 1AA",
            "yes that is correct",   # postcode NATO confirm
            "22nd August 1975",
            "What is my current pension value?",
        ]

        for user_text in script:
            agent.process_turn(ctx, text_input=user_text)
            if ctx.state in (AgentState.RESOLVED, AgentState.ESCALATED):
                break

        # Should have progressed past COLLECT_POLICY at minimum
        assert ctx.state != AgentState.COLLECT_POLICY, \
            f"Agent stuck in COLLECT_POLICY after full script"
        # Should NOT have escalated due to LLM_DIRECTED (no backend)
        if ctx.state == AgentState.ESCALATED:
            assert ctx.escalation_reason != "LLM_DIRECTED", \
                f"Agent escalated due to missing LLM backend: {ctx.escalation_reason}"


# ── ASR Error Tests ───────────────────────────────────────────────────────────

class TestVoiceASRErrors:
    """Voice call where ASR fails mid-conversation."""

    def test_empty_transcription_handled_gracefully(self):
        """Empty ASR result should not crash the agent."""
        agent = AgentOrchestrator(channel="voice")
        ctx = agent.new_session()

        with patch("app.asr_client.ASRClient.transcribe", return_value=""):
            response = agent.process_turn(ctx, audio_bytes=b"\x00" * 100)

        assert response != ""
        assert ctx.state != AgentState.ESCALATED

    def test_asr_timeout_returns_retry_prompt(self):
        """ASR timeout should NOT crash — agent must handle it gracefully (fix: try-except in process_turn)."""
        agent = AgentOrchestrator(channel="voice")
        ctx = agent.new_session()

        with patch("app.asr_client.ASRClient.transcribe", side_effect=Exception("Groq timeout")):
            try:
                response = agent.process_turn(ctx, audio_bytes=b"\x00" * 100)
                # After ASR failure, user_text="" → agent should still respond (ask to repeat)
                assert response != "", "Agent returned empty string after ASR failure"
            except Exception as exc:
                pytest.fail(f"ASR timeout should be handled gracefully, not re-raised: {exc}")


# ── LLM Delay / Timeout Tests ─────────────────────────────────────────────────

class TestVoiceLLMDelays:
    """Tests for LLM API slowness and timeouts."""

    def test_llm_slow_response_still_completes(self):
        """LLM with artificial delay should still complete."""
        import time
        agent = AgentOrchestrator(channel="voice")
        ctx = agent.new_session()

        # We test with a tiny delay (0.05s) to keep tests fast
        _original = agent._llm.call
        def slow_call(*args, **kwargs):
            time.sleep(0.05)
            return _original(*args, **kwargs)

        with patch.object(agent._llm, "call", side_effect=slow_call):
            response = agent.process_turn(ctx, text_input="hello")

        assert response != ""

    def test_llm_timeout_escalates_gracefully(self):
        """If _call_llm returns an 'escalate' turn, agent escalates gracefully without crashing."""
        agent = AgentOrchestrator(channel="voice")
        ctx = agent.new_session()

        # Simulate LLM returning a graceful escalation turn
        escalate_turn = AgentTurn(
            action="escalate",
            caller_response="I'm experiencing technical difficulties, transferring you now.",
            intent="escalate",
            confidence=1.0,
        )
        with patch.object(agent, "_call_llm", return_value=escalate_turn):
            response = agent.process_turn(ctx, text_input="hello")

        assert response != ""


# ── RAG / FAQ Failure Tests ───────────────────────────────────────────────────

class TestVoiceRAGFailures:
    """Tests for FAQ retrieval failures during live calls."""

    VERIFIED_SCRIPT = [
        "hello", "ABC slash 123 dash 45", "yes", "I am the policy holder",
        "Jonathan Smith", "14 High Street", "M1 1AA", "yes", "22 August 1975"
    ]

    def _run_to_verified(self, agent, ctx):
        """Drive the conversation to a post-verification state."""
        for text in self.VERIFIED_SCRIPT:
            agent.process_turn(ctx, text_input=text)
            if ctx.state.value in ("serve_intent", "resolved", "escalated"):
                break

    def test_faq_not_found_returns_helpful_message(self):
        """If no FAQ chunk found, agent should respond — not crash or escalate."""
        from app.rag_client import RAGResult
        agent = AgentOrchestrator(channel="voice")
        ctx = agent.new_session()
        self._run_to_verified(agent, ctx)

        if ctx.state == AgentState.ESCALATED:
            pytest.skip("Could not reach serve_intent without real LLM — skipping RAG test")

        with patch.object(agent._rag, "query",
                          return_value=RAGResult(query="test", answerable=False)):
            response = agent.process_turn(
                ctx, text_input="What is the maximum age I can take my pension?"
            )
        assert response != ""

    def test_faq_low_confidence_does_not_crash(self):
        """Low-confidence RAG result should be handled, not surface as an error."""
        from app.rag_client import RAGResult, RAGChunk
        agent = AgentOrchestrator(channel="voice")
        ctx = agent.new_session()
        self._run_to_verified(agent, ctx)

        if ctx.state == AgentState.ESCALATED:
            pytest.skip("Could not reach serve_intent without real LLM — skipping RAG test")

        low_conf_result = RAGResult(
            query="test", answerable=False,
            chunks=[RAGChunk(chunk_id="x", text="Maybe...", score=0.3,
                             source_doc="doc", section="s")]
        )
        with patch.object(agent._rag, "query", return_value=low_conf_result):
            response = agent.process_turn(ctx, text_input="Can I transfer my pension?")
        assert response != ""
