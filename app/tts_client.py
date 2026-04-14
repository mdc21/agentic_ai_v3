"""
tts_client.py — Text-to-Speech adapter for the voice channel.

For chat channel, TTS is never called (channel='chat').

Free-tier options for development:
  - Google Cloud Text-to-Speech: 1 million chars/month free
    pip install google-cloud-texttospeech
    Set TTS_BACKEND=google, GOOGLE_APPLICATION_CREDENTIALS=/path/key.json

  - gTTS (offline-capable, simpler quality):
    pip install gTTS
    Set TTS_BACKEND=gtts

  - Mock (returns empty bytes, logs the text):
    Set USE_MOCK_TTS=true  (default in development)
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

TTS_BACKEND = os.getenv("TTS_BACKEND", "mock")


class TTSClient:
    """
    Convert agent response text to audio bytes for the voice channel.
    In chat mode, this client is never instantiated.
    """

    def __init__(self) -> None:
        self._mock = (
            TTS_BACKEND == "mock"
            or os.getenv("USE_MOCK_TTS", "false").lower() == "true"
        )
        if not self._mock:
            self._init_backend()

    def _init_backend(self) -> None:
        if TTS_BACKEND == "google":
            try:
                from google.cloud import texttospeech
                self._client = texttospeech.TextToSpeechClient()
                self._voice  = texttospeech.VoiceSelectionParams(
                    language_code=os.getenv("TTS_LANGUAGE", "en-GB"),
                    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
                )
                self._audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=16000,
                )
                logger.info("Google Cloud TTS client initialised")
            except ImportError:
                logger.warning("google-cloud-texttospeech not installed — mock TTS")
                self._mock = True

        elif TTS_BACKEND == "gtts":
            try:
                import gtts  # noqa: F401  — just checking import
                logger.info("gTTS client ready")
            except ImportError:
                logger.warning("gTTS not installed — mock TTS")
                self._mock = True
        else:
            logger.warning("Unknown TTS_BACKEND %r — mock TTS", TTS_BACKEND)
            self._mock = True

    def synthesise(self, text: str, session_id: str = "") -> bytes:
        """
        Convert text to audio bytes (LINEAR16 PCM, 16 kHz).
        Returns empty bytes in mock mode.
        """
        if self._mock:
            logger.debug("[%s] TTS (mock): %r", session_id, text[:80])
            return b""

        if TTS_BACKEND == "google":
            return self._google_synthesise(text)
        elif TTS_BACKEND == "gtts":
            return self._gtts_synthesise(text)
        return b""

    def _google_synthesise(self, text: str) -> bytes:
        from google.cloud import texttospeech
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self._client.synthesize_speech(
            input=synthesis_input, voice=self._voice, audio_config=self._audio_config
        )
        return response.audio_content

    def _gtts_synthesise(self, text: str) -> bytes:
        import io
        from gtts import gTTS
        buf = io.BytesIO()
        tts = gTTS(text=text, lang=os.getenv("TTS_LANGUAGE", "en"))
        tts.write_to_fp(buf)
        return buf.getvalue()
