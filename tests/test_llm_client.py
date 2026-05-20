# tests/test_llm_client.py
import os, sys, pytest
from unittest.mock import patch, MagicMock

# Inject a dummy key so LLMClient doesn't fall back to "mock" backend
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key")

from app.llm_client import LLMClient, AgentTurn

VALID_LLM_RESPONSE = """{
  "intent": "request_policy_number",
  "action": "request_policy_number",
  "entities": {},
  "caller_response": "Could you please provide your policy number?",
  "confidence": 0.95,
  "duress_signal": false
}"""


class TestLLMClientParsing:
    def setup_method(self):
        # Clear the cached client between tests so the mock takes effect
        LLMClient._clients.clear()

    def test_valid_response_parsed(self):
        """LLM returns valid JSON → AgentTurn is parsed correctly."""
        mock_choice = MagicMock()
        mock_choice.message.content = VALID_LLM_RESPONSE
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = mock_completion

        # Patch where LLMClient actually calls OpenAI (inside _get_client lazy init)
        with patch("app.llm_client.LLMClient._get_client", return_value=mock_openai_instance):
            client = LLMClient()
            turn = client.call([{"role": "user", "content": "hello"}], "")

        assert turn.action == "request_policy_number"
        assert turn.duress_signal is False

    def test_malformed_json_returns_safe_fallback(self):
        """Malformed JSON from LLM should not crash — returns a safe AgentTurn."""
        mock_choice = MagicMock()
        mock_choice.message.content = "this is not json {"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 50
        mock_completion.usage.completion_tokens = 5

        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.return_value = mock_completion

        with patch("app.llm_client.LLMClient._get_client", return_value=mock_openai_instance):
            client = LLMClient()
            turn = client.call([{"role": "user", "content": "hello"}], "")

        assert isinstance(turn, AgentTurn)

    def test_timeout_returns_safe_fallback(self):
        """LLM API timeout should not crash — returns a safe AgentTurn."""
        mock_openai_instance = MagicMock()
        mock_openai_instance.chat.completions.create.side_effect = TimeoutError("LLM timeout")

        with patch("app.llm_client.LLMClient._get_client", return_value=mock_openai_instance):
            client = LLMClient()
            turn = client.call([{"role": "user", "content": "hello"}], "")

        assert isinstance(turn, AgentTurn)
