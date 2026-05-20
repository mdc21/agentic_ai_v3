# tests/test_asr_client.py
import os, sys, pytest
from unittest.mock import patch, MagicMock

os.environ["USE_MOCK_ASR"] = "true"
from app.asr_client import ASRClient


class TestASRMock:
    def test_text_passthrough(self):
        """Chat mode: text_input is returned unchanged."""
        asr = ASRClient()
        result = asr.transcribe(text_input="My policy is ABC/123")
        assert result == "My policy is ABC/123"

    def test_empty_audio_returns_empty(self):
        asr = ASRClient()
        result = asr.transcribe(audio_bytes=b"")
        assert result == "" or "[Mock ASR" in result

    def test_mock_flag_is_set(self):
        asr = ASRClient()
        assert asr.is_mock is True


class TestASRGroqErrors:
    """
    These tests mock the groq module via sys.modules so they work even
    when the groq package is not installed in the test environment.
    """

    def _make_groq_mock(self):
        """Create a fake groq module with the Groq class stubbed out."""
        mock_groq_module = MagicMock()
        mock_groq_class  = MagicMock()
        mock_groq_module.Groq = mock_groq_class
        return mock_groq_module, mock_groq_class

    @patch.dict(os.environ, {"USE_MOCK_ASR": "false", "ASR_BACKEND": "groq", "GROQ_API_KEY": "test-key"})
    def test_groq_timeout_returns_empty(self):
        """ASR should fail gracefully on API timeout."""
        mock_module, mock_class = self._make_groq_mock()
        mock_class.return_value.audio.transcriptions.create.side_effect = TimeoutError("API timeout")

        with patch.dict(sys.modules, {"groq": mock_module}):
            asr = ASRClient()
            asr._mock = False
            asr._backend = "groq"
            asr._groq_client = mock_class.return_value
            result = asr.transcribe(audio_bytes=b"fake-audio")

        assert result == ""   # graceful fallback

    @patch.dict(os.environ, {"USE_MOCK_ASR": "false", "ASR_BACKEND": "groq", "GROQ_API_KEY": "test-key"})
    def test_groq_rate_limit_returns_empty(self):
        """ASR should return empty on 429 rate limit."""
        mock_module, mock_class = self._make_groq_mock()
        mock_class.return_value.audio.transcriptions.create.side_effect = Exception("429 Rate limit exceeded")

        with patch.dict(sys.modules, {"groq": mock_module}):
            asr = ASRClient()
            asr._mock = False
            asr._backend = "groq"
            asr._groq_client = mock_class.return_value
            result = asr.transcribe(audio_bytes=b"fake-audio")

        assert result == ""

    def test_groq_returns_text_on_success(self):
        """Successful transcription returns the text."""
        mock_module, mock_class = self._make_groq_mock()
        mock_response = MagicMock()
        mock_response.text = "My policy number is ABC slash 123 dash 45"
        mock_class.return_value.audio.transcriptions.create.return_value = mock_response

        with patch.dict(sys.modules, {"groq": mock_module}):
            asr = ASRClient()
            asr._mock = False
            asr._backend = "groq"
            asr._groq_client = mock_class.return_value
            result = asr.transcribe(audio_bytes=b"valid-audio-bytes")

        assert result == "My policy number is ABC slash 123 dash 45"
