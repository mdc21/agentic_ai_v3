"""
ASR client — Supports Google Dialogflow CX and OpenAI Whisper.
Uses the free quota tier (600 mins/month on Dialogflow CX) or OpenAI API.

For testing without billing, set USE_MOCK_ASR=true in your .env
and the client returns the raw text input unchanged.
"""

import logging
import os
import io
from typing import Optional

logger = logging.getLogger(__name__)

ASR_BACKEND = os.getenv("ASR_BACKEND", "mock") # mock | google | openai


class ASRClient:
    """
    Wraps ASR backends for transcribing audio bytes.
    """

    def __init__(self) -> None:
        self._mock = (
            ASR_BACKEND == "mock"
            or os.getenv("USE_MOCK_ASR", "false").lower() == "true"
        )
        if not self._mock:
            self._init_backend()

    def _init_backend(self) -> None:
        if ASR_BACKEND == "google":
            self._init_dialogflow()
        elif ASR_BACKEND == "openai":
            self._init_openai()
        elif ASR_BACKEND == "groq":
            self._init_groq()
        else:
            logger.warning("Unknown ASR_BACKEND %r — falling back to mock", ASR_BACKEND)
            self._mock = True

    def _init_dialogflow(self) -> None:
        try:
            from google.cloud.dialogflowcx_v3 import SessionsClient
            from google.cloud.dialogflowcx_v3.types import (
                AudioEncoding,
                InputAudioConfig,
                QueryInput,
                DetectIntentRequest,
            )
            self._SessionsClient = SessionsClient
            self._AudioEncoding = AudioEncoding
            self._InputAudioConfig = InputAudioConfig
            self._QueryInput = QueryInput
            self._DetectIntentRequest = DetectIntentRequest

            project_id = os.getenv("DIALOGFLOW_PROJECT_ID")
            location = os.getenv("DIALOGFLOW_LOCATION", "global")
            agent_id = os.getenv("DIALOGFLOW_AGENT_ID")

            if not project_id or not agent_id:
                logger.warning("Dialogflow credentials (PROJECT_ID or AGENT_ID) missing. Falling back to mock ASR.")
                self._mock = True
                return

            self._session_prefix = (
                f"projects/{project_id}/locations/{location}/agents/{agent_id}/sessions/"
            )
            self._client = SessionsClient(
                client_options={"api_endpoint": f"{location}-dialogflow.googleapis.com"}
                if location != "global"
                else {}
            )
            logger.info("Dialogflow CX ASR client initialised")
        except ImportError:
            logger.warning("google-cloud-dialogflow-cx not installed, falling back to mock ASR.")
            self._mock = True

    def _init_openai(self) -> None:
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY missing. Falling back to mock ASR.")
                self._mock = True
                return
            self._openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI Whisper ASR client initialised")
        except ImportError:
            logger.warning("openai not installed, falling back to mock ASR.")
            self._mock = True

    def _init_groq(self) -> None:
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY missing. Falling back to mock ASR.")
                self._mock = True
                return
            self._groq_client = Groq(api_key=api_key)
            logger.info("Groq ASR client initialised")
        except ImportError:
            logger.warning("The 'groq' library is not installed. Please run 'pip install groq' to use Groq ASR. Falling back to mock.")
            self._mock = True

    @property
    def is_mock(self) -> bool:
        return self._mock

    def transcribe(
        self,
        audio_bytes: Optional[bytes] = None,
        text_input: Optional[str] = None,
        session_id: str = "default-session",
        language_code: str = "en-GB",
    ) -> str:
        """
        Transcribe audio bytes OR return text_input directly (chat mode / mock).
        """
        if text_input is not None:
            return text_input

        if self._mock:
            if audio_bytes:
                logger.warning("ASR is in MOCK mode but received audio bytes. Returning placeholder.")
                return "[Mock ASR: Audio received but backend is disabled]"
            return ""

        if audio_bytes is None:
            raise ValueError("Either audio_bytes or text_input must be provided")

        logger.info("Transcribing audio using %s backend", ASR_BACKEND)
        if ASR_BACKEND == "google":
            return self._google_transcribe(audio_bytes, session_id, language_code)
        elif ASR_BACKEND == "openai":
            return self._openai_transcribe(audio_bytes)
        elif ASR_BACKEND == "groq":
            return self._groq_transcribe(audio_bytes)
        
        return ""

    def _google_transcribe(self, audio_bytes: bytes, session_id: str, language_code: str) -> str:
        session = f"{self._session_prefix}{session_id}"
        audio_config = self._InputAudioConfig(
            audio_encoding=self._AudioEncoding.AUDIO_ENCODING_LINEAR_16,
            sample_rate_hertz=16000,
            language_code=language_code,
        )
        query_input = self._QueryInput(
            audio=self._QueryInput.AudioInput(config=audio_config, audio=audio_bytes),
            language_code=language_code,
        )
        request = self._DetectIntentRequest(session=session, query_input=query_input)
        response = self._client.detect_intent(request=request)
        return response.query_result.transcript

    def _openai_transcribe(self, audio_bytes: bytes) -> str:
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.webm"
            transcript = self._openai_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
            return transcript.text
        except Exception as e:
            logger.error("OpenAI ASR error: %s. Falling back to empty transcription.", e)
            return ""

    def _groq_transcribe(self, audio_bytes: bytes) -> str:
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.webm"
            
            transcript = self._groq_client.audio.transcriptions.create(
                file=("audio.webm", audio_file),
                model="whisper-large-v3",
                prompt="The audio is from an insurance policy servicing call.", # context aid
                response_format="json",
                language="en",
            )
            return transcript.text
        except Exception as e:
            logger.error("Groq ASR error: %s. Falling back to empty transcription.", e)
            return ""
