"""
tts_client.py — Text-to-Speech adapter for the voice channel.
Supports Google Cloud, OpenAI, and gTTS backends.
"""

import logging
import os
import io
from typing import Optional

logger = logging.getLogger(__name__)

# TTS_BACKEND resolution moved to class __init__ for st.secrets compatibility


class TTSClient:
    """
    Convert agent response text to audio bytes for the voice channel.
    """

    def __init__(self) -> None:
        # Resolve backend with st.secrets fallback
        backend = os.getenv("TTS_BACKEND", "mock")
        if backend == "mock":
            try:
                import streamlit as st
                if hasattr(st, "secrets"):
                    backend = st.secrets.get("TTS_BACKEND", "mock")
            except ImportError:
                pass
        self._backend = backend

        mock_flag = os.getenv("USE_MOCK_TTS", "false").lower() == "true"
        # Check secrets for mock flag too
        if not mock_flag:
            try:
                import streamlit as st
                if hasattr(st, "secrets"):
                    mock_flag = st.secrets.get("USE_MOCK_TTS", False)
            except:
                pass

        self._mock = (self._backend == "mock" or mock_flag)
        if not self._mock:
            self._init_backend()

    def _init_backend(self) -> None:
        if self._backend == "google":
            self._init_google()
        elif self._backend == "openai":
            self._init_openai()
        elif self._backend == "gtts":
            self._init_gtts()
        else:
            logger.warning("Unknown TTS_BACKEND %r — falling back to mock", self._backend)
            self._mock = True

    def _init_google(self) -> None:
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
            logger.warning("google-cloud-texttospeech not installed — falling back to mock TTS")
            self._mock = True

    def _init_openai(self) -> None:
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            
            # Streamlit Cloud secrets fallback
            if not api_key:
                try:
                    import streamlit as st
                    if hasattr(st, "secrets"):
                        api_key = st.secrets.get("OPENAI_API_KEY")
                except ImportError:
                    pass

            if not api_key:
                logger.warning("OPENAI_API_KEY missing in both env and st.secrets. Falling back to mock TTS.")
                self._mock = True
                return
            self._openai_client = OpenAI(api_key=api_key)
            self._voice = os.getenv("OPENAI_TTS_VOICE", "nova")
            logger.info("OpenAI TTS client initialised")
        except ImportError:
            logger.warning("openai not installed — falling back to mock TTS")
            self._mock = True

    def _init_gtts(self) -> None:
        try:
            import gtts
            logger.info("gTTS client initialised")
        except ImportError:
            logger.warning("gTTS not installed — falling back to mock TTS")
            self._mock = True

    def synthesise(self, text: str, session_id: str = "") -> bytes:
        """
        Convert text to audio bytes.
        """
        if self._mock:
            logger.debug("[%s] TTS (mock): %r", session_id, text[:80])
            return b""

        if self._backend == "google":
            return self._google_synthesise(text)
        elif self._backend == "openai":
            return self._openai_synthesise(text)
        elif self._backend == "gtts":
            return self._gtts_synthesise(text)
        return b""

    def _google_synthesise(self, text: str) -> bytes:
        from google.cloud import texttospeech
        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self._client.synthesize_speech(
            input=synthesis_input, voice=self._voice, audio_config=self._audio_config
        )
        return response.audio_content

    def _openai_synthesise(self, text: str) -> bytes:
        try:
            response = self._openai_client.audio.speech.create(
                model="tts-1",
                voice=self._voice,
                input=text
            )
            # Returns binary response content
            return response.content
        except Exception as e:
            logger.error("OpenAI TTS error: %s. Returning empty audio.", e)
            return b""

    def _gtts_synthesise(self, text: str) -> bytes:
        from gtts import gTTS
        buf = io.BytesIO()
        tts = gTTS(text=text, lang=os.getenv("TTS_LANGUAGE", "en")[:2])
        tts.write_to_fp(buf)
        return buf.getvalue()
