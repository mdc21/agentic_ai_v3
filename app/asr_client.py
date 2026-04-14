"""
ASR client — Google Dialogflow CX streaming speech-to-text.
Uses the free quota tier (600 mins/month on Dialogflow CX).

For testing without billing, set USE_MOCK_ASR=true in your .env
and the client returns the raw text input unchanged.

Setup:
  1. Create a Dialogflow CX project at console.dialogflow.cloud.google.com
  2. Enable the Dialogflow CX API
  3. Create a service account key → set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
  4. Set DIALOGFLOW_PROJECT_ID and DIALOGFLOW_LOCATION in .env
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class ASRClient:
    """
    Wraps Dialogflow CX detect_intent for streaming/single-utterance ASR.
    Falls back to pass-through when USE_MOCK_ASR=true (useful for chat-only mode).
    """

    def __init__(self) -> None:
        self._mock = os.getenv("USE_MOCK_ASR", "false").lower() == "true"
        if not self._mock:
            self._init_dialogflow()

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
            logger.info("Dialogflow CX ASR client initialised (project=%s)", project_id)
        except ImportError:
            logger.warning("google-cloud-dialogflow-cx not installed, falling back to mock ASR.")
            self._mock = True

    def transcribe(
        self,
        audio_bytes: Optional[bytes] = None,
        text_input: Optional[str] = None,
        session_id: str = "default-session",
        language_code: str = "en-GB",
    ) -> str:
        """
        Transcribe audio bytes OR return text_input directly (chat mode / mock).
        Returns the best hypothesis as a plain string.
        """
        if self._mock or text_input is not None:
            return text_input or ""

        if audio_bytes is None:
            raise ValueError("Either audio_bytes or text_input must be provided")

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

        transcript = response.query_result.transcript
        logger.debug("ASR transcript: %r", transcript)
        return transcript
